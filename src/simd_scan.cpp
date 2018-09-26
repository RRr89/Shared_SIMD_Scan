#include <iostream>

#if defined(_MSC_VER)
#  include <intrin.h>
#else
#  include <x86intrin.h>
#endif

#include <immintrin.h>
#include <cmath>
#include <vector>
#include <bitset>
#include <algorithm>

#include "simd_scan.hpp"
#include "profiling.hpp"
#include "util.hpp"
#include "simd_scan_commons.hpp"

int scan_unvectorized(int predicate_key, __m128i* input, size_t input_size, std::vector<uint8_t>& output)
{
    uint64_t* in = reinterpret_cast<uint64_t*>(input);
    auto bits_needed = BITS_NEEDED;
    auto mem_size = bits_needed * input_size;
    int array_size = ceil((double)mem_size / 64);

    uint64_t mask = (1 << bits_needed) - 1;

    uint64_t current = 0;
    size_t overflow_bits = 0;

    size_t oi = 0;

    int hits = 0;

    uint8_t output_byte = 0;
    size_t out_bits_used = 0;

    for (size_t i = 0; i < array_size; i++)
    {
        current = in[i] >> overflow_bits;

        size_t unread_bits = 64 - overflow_bits;
        while (unread_bits >= bits_needed)
        {
            uint16_t decompressed_element = current & mask;
            bool match = decompressed_element == predicate_key;
            output_byte |= match << (out_bits_used++);

            if (out_bits_used == 8)
            {
                output[oi++] = output_byte;
                hits += POPCNT(output_byte);
                output_byte = 0;
                out_bits_used = 0;
            }

            if (oi == input_size)
            {
                return hits;
            }

            current = current >> bits_needed;
            unread_bits -= bits_needed;
        }

        // handle overlapping element
        if (unread_bits != 0)
        {
            uint64_t next = in[i + 1];
            current = current | (next << unread_bits);

            uint16_t decompressed_element = current & mask;
            bool match = decompressed_element == predicate_key;
            output_byte |= match << (out_bits_used++);

            if (out_bits_used == 8)
            {
                output[oi++] = output_byte;
                hits += POPCNT(output_byte);
                output_byte = 0;
                out_bits_used = 0;
            }

            overflow_bits = bits_needed - unread_bits;
        }
        else
        {
            overflow_bits = 0;
        }
    }

    if (out_bits_used != 0) 
    {
        output[oi] = output_byte;
        hits += POPCNT(output_byte);
    }

    return hits;
}

// based on decompress_128_unrolled
int scan_128(int predicate_key, __m128i* input, size_t input_size, std::vector<uint8_t>& output)
{
    int hits = 0;

    size_t compression = BITS_NEEDED;

    __m128i source = _mm_loadu_si128(input);

    size_t output_index = 0; // current write index of the output array

    __m128i shuffle_mask[2];
    generate_shuffle_mask_128(compression, shuffle_mask);

    __m128i clean_mask[2];
    generate_clean_masks_128(compression, clean_mask);

    __m128i predicate[2];
    generate_predicate_masks_128(compression, predicate_key, predicate);

    while (8 * output_index < input_size)
    {
        uint8_t out = 0;

        {
            size_t mask_index = 0;
            __m128i b = _mm_shuffle_epi8(source, shuffle_mask[mask_index]);
            __m128i c = _mm_and_si128(b, clean_mask[mask_index]);
            __m128i e = _mm_cmpeq_epi32(c, predicate[mask_index]);

            out |= _mm_movemask_ps(_mm_castsi128_ps(e));

            // load next
            size_t total_processed_bytes = (8 * output_index + 4) * compression / 8;
            source = _mm_loadu_si128((__m128i*)&((uint8_t*)input)[total_processed_bytes]);
        }

        {
            size_t mask_index = 1;
            __m128i b = _mm_shuffle_epi8(source, shuffle_mask[mask_index]);
            __m128i c = _mm_and_si128(b, clean_mask[mask_index]);
            __m128i e = _mm_cmpeq_epi32(c, predicate[mask_index]);

            out |= (_mm_movemask_ps(_mm_castsi128_ps(e)) << 4);

            // load next
            size_t total_processed_bytes = (8 * output_index + 8) * compression / 8;
            source = _mm_loadu_si128((__m128i*)&((uint8_t*)input)[total_processed_bytes]);
        }

        output[output_index] = out;
        output_index += 1;
        hits += POPCNT(out);
    }

    return hits;
}

inline void __scan_128_step(uint32_t& out, size_t const& offset, __m128i const& shuffle_mask, 
    __m128i const& clean_mask, __m128i const& predicate, size_t const& output_index, size_t const& compression, 
    __m128i* input, __m128i& source)
{
    __m128i b = _mm_shuffle_epi8(source, shuffle_mask);
    __m128i c = _mm_and_si128(b, clean_mask);
    __m128i e = _mm_cmpeq_epi32(c, predicate);

    out |= (_mm_movemask_ps(_mm_castsi128_ps(e)) << offset);

    // load next
    size_t total_processed_bytes = (32 * output_index + offset + 4) * compression / 8;
    source = _mm_loadu_si128((__m128i*)&((uint8_t*)input)[total_processed_bytes]);
}

int scan_128_unrolled(int predicate_key, __m128i* input, size_t input_size, std::vector<uint8_t>& output)
{
    int hits = 0;

    size_t compression = BITS_NEEDED;

    __m128i source = _mm_loadu_si128(input);

    uint32_t* output_array = reinterpret_cast<uint32_t*>(output.data());
    size_t output_index = 0; // current write index of the output array

    __m128i shuffle_mask[2];
    generate_shuffle_mask_128(compression, shuffle_mask);

    __m128i clean_mask[2];
    generate_clean_masks_128(compression, clean_mask);

    __m128i predicate[2];
    generate_predicate_masks_128(compression, predicate_key, predicate);

    while (32 * output_index < input_size)
    {
        uint32_t out = 0;

        __scan_128_step(out,  0, shuffle_mask[0], clean_mask[0], predicate[0], output_index, compression, input, source);
        __scan_128_step(out,  4, shuffle_mask[1], clean_mask[1], predicate[1], output_index, compression, input, source);

        __scan_128_step(out,  8, shuffle_mask[0], clean_mask[0], predicate[0], output_index, compression, input, source);
        __scan_128_step(out, 12, shuffle_mask[1], clean_mask[1], predicate[1], output_index, compression, input, source);

        __scan_128_step(out, 16, shuffle_mask[0], clean_mask[0], predicate[0], output_index, compression, input, source);
        __scan_128_step(out, 20, shuffle_mask[1], clean_mask[1], predicate[1], output_index, compression, input, source);

        __scan_128_step(out, 24, shuffle_mask[0], clean_mask[0], predicate[0], output_index, compression, input, source);
        __scan_128_step(out, 28, shuffle_mask[1], clean_mask[1], predicate[1], output_index, compression, input, source);

        output_array[output_index] = out;
        output_index += 1;
        hits += POPCNT(out);
    }

    return hits;
}

#ifdef __AVX__
int scan_256(int predicate_key, __m128i* input, size_t input_size, std::vector<uint8_t>& output)
{
    int hits = 0;

    size_t compression = BITS_NEEDED;

    //avxi_t source = _mm256_loadu_si256(input);
    __m256i source = _mm256_loadu2_m128i(input, input);

    size_t output_index = 0; // current write index of the output array (equals # of decompressed values)

    __m256i shuffle_mask = generate_shuffle_mask_256(compression);

    __m256i clean_mask = generate_clean_mask_256(compression);

    __m256i predicate = generate_predicate_mask_256(compression, predicate_key);

    while (8 * output_index < input_size)
    {
        __m256i b = _mm256_shuffle_epi8(source, shuffle_mask);
        __m256i c = _mm256_and_si256(b, clean_mask);
        __m256i e = _mm256_cmpeq_epi32(c, predicate);

        int matches = _mm256_movemask_ps(_mm256_castsi256_ps(e));
        hits += POPCNT(matches);
        output[output_index] = matches;

        // load next
        output_index += 1;
        size_t total_processed_bytes = (8 * output_index) * compression / 8;
        __m128i* next = (__m128i*)&((uint8_t*)input)[total_processed_bytes];
        source = _mm256_loadu2_m128i(next, next);
    }

    return hits;
}

inline void __scan_256_step(uint32_t& out, size_t const& offset, __m256i const& shuffle_mask,
    __m256i const& clean_mask, __m256i const& predicate, size_t const& output_index, size_t const& compression,
    __m128i* input, __m256i& source)
{
    __m256i b = _mm256_shuffle_epi8(source, shuffle_mask);
    __m256i c = _mm256_and_si256(b, clean_mask);
    __m256i e = _mm256_cmpeq_epi32(c, predicate);

    out |= _mm256_movemask_ps(_mm256_castsi256_ps(e)) << offset;

    // load next
    size_t total_processed_bytes = (32 * output_index + offset + 8) * compression / 8;
    __m128i* next = (__m128i*)&((uint8_t*)input)[total_processed_bytes];
    source = _mm256_loadu2_m128i(next, next);
}

int scan_256_unrolled(int predicate_key, __m128i* input, size_t input_size, std::vector<uint8_t>& output)
{
    int hits = 0;

    size_t compression = BITS_NEEDED;

    //avxi_t source = _mm256_loadu_si256(input);
    __m256i source = _mm256_loadu2_m128i(input, input);

    uint32_t* output_array = reinterpret_cast<uint32_t*>(output.data());
    size_t output_index = 0; // current write index of the output array (equals # of decompressed values)

    __m256i shuffle_mask = generate_shuffle_mask_256(compression);

    __m256i clean_mask = generate_clean_mask_256(compression);

    __m256i predicate = generate_predicate_mask_256(compression, predicate_key);

    while (32 * output_index < input_size)
    {
        uint32_t out = 0;

        __scan_256_step(out,  0, shuffle_mask, clean_mask, predicate, output_index, compression, input, source);
        __scan_256_step(out,  8, shuffle_mask, clean_mask, predicate, output_index, compression, input, source);
        __scan_256_step(out, 16, shuffle_mask, clean_mask, predicate, output_index, compression, input, source);
        __scan_256_step(out, 24, shuffle_mask, clean_mask, predicate, output_index, compression, input, source);

        output_array[output_index] = out;
        output_index += 1;
        hits += POPCNT(out);
    }

    return hits;
}
#endif

