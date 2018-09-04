#include <iostream>
#include <immintrin.h>
#include <cmath>
#include <vector>
#include <bitset>
#include <algorithm>

#include "simd_scan.hpp"
#include "profiling.hpp"

int scan_unvectorized(int predicate_key, __m128i* input, size_t input_size, std::vector<bool>& output)
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

    for (size_t i = 0; i < array_size; i++)
    {
        current = in[i] >> overflow_bits;

        size_t unread_bits = 64 - overflow_bits;
        while (unread_bits >= bits_needed)
        {
            uint16_t decompressed_element = current & mask;
            bool match = decompressed_element == predicate_key;
            output[oi++] = match;
            if (match) hits++;

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
            output[oi++] = match;
            if (match) hits++;

            overflow_bits = bits_needed - unread_bits;
        }
        else 
        {
            overflow_bits = 0;
        }
    }

    return hits;
}

// based on decompress_128_unrolled
int scan_128(int predicate_key, __m128i* input, size_t input_size, std::vector<bool>& output)
{
    int hits = 0;

    size_t compression = BITS_NEEDED;
    size_t free_bits = 32 - compression; // most significant bits in result values that must be 0

    __m128i source = _mm_loadu_si128(input);

    size_t output_index = 0; // current write index of the output array (equals # of decompressed values)
    size_t total_processed_bytes = 0; // holds # of input bytes that have been processed completely

    // shuffle masks
    size_t input_offset[8];
    for (size_t i = 0; i < 8; i++)
    {
        input_offset[i] = (compression * i) / 8;
    }
    size_t correction = input_offset[4];
    for (size_t i = 4; i < 8; i++)
    {
        input_offset[i] -= correction;
    }

    __m128i shuffle_mask[2];
    shuffle_mask[0] = _mm_setr_epi8(
        input_offset[0], input_offset[0] + 1, input_offset[0] + 2, input_offset[0] + 3,
        input_offset[1], input_offset[1] + 1, input_offset[1] + 2, input_offset[1] + 3,
        input_offset[2], input_offset[2] + 1, input_offset[2] + 2, input_offset[2] + 3,
        input_offset[3], input_offset[3] + 1, input_offset[3] + 2, input_offset[3] + 3);
    shuffle_mask[1] = _mm_setr_epi8(
        input_offset[4], input_offset[4] + 1, input_offset[4] + 2, input_offset[4] + 3,
        input_offset[5], input_offset[5] + 1, input_offset[5] + 2, input_offset[5] + 3,
        input_offset[6], input_offset[6] + 1, input_offset[6] + 2, input_offset[6] + 3,
        input_offset[7], input_offset[7] + 1, input_offset[7] + 2, input_offset[7] + 3);

    // shift masks
    size_t padding[8];
    for (size_t i = 0; i < 8; i++)
    {
        padding[i] = (compression * i) % 8;
    }

    __m128i shift_mask[2];
    shift_mask[0] = _mm_setr_epi32(
        1 << (free_bits - padding[0]),
        1 << (free_bits - padding[1]),
        1 << (free_bits - padding[2]),
        1 << (free_bits - padding[3]));
    shift_mask[1] = _mm_setr_epi32(
        1 << (free_bits - padding[4]),
        1 << (free_bits - padding[5]),
        1 << (free_bits - padding[6]),
        1 << (free_bits - padding[7]));

    __m128i predicate = _mm_set1_epi32(predicate_key);

    PROFILE_SAMPLE(whole_loop);
    PROFILE_SAMPLE(write_result);

    while (output_index < input_size)
    {
        {
            PROFILE_BLOCK_START(whole_loop);

            size_t mask_index = 0;
            __m128i b = _mm_shuffle_epi8(source, shuffle_mask[mask_index]);
            __m128i c = _mm_mullo_epi32(b, shift_mask[mask_index]);
            __m128i d = _mm_srli_epi32(c, 32 - compression);
            __m128i e = _mm_cmpeq_epi32(d, predicate);

            PROFILE_BLOCK_START(write_result);
            for (size_t i = 0; i < 4; i++) 
            {
                bool match = e.m128i_u32[i] == 0xFFFFFFFF;
                output[output_index++] = match;
                if (match) hits++;
            }
            PROFILE_BLOCK_END(write_result);

            // load next
            total_processed_bytes = output_index * compression / 8;
            source = _mm_loadu_si128((__m128i*)&((uint8_t*)input)[total_processed_bytes]);

            PROFILE_BLOCK_END(whole_loop);
        }

        {
            PROFILE_BLOCK_START(whole_loop);

            size_t mask_index = 1;
            __m128i b = _mm_shuffle_epi8(source, shuffle_mask[mask_index]);
            __m128i c = _mm_mullo_epi32(b, shift_mask[mask_index]);
            __m128i d = _mm_srli_epi32(c, 32 - compression);
            __m128i e = _mm_cmpeq_epi32(d, predicate);

            PROFILE_BLOCK_START(write_result);
            for (size_t i = 0; i < 4; i++)
            {
                bool match = e.m128i_u32[i] == 0xFFFFFFFF;
                output[output_index++] = match;
                if (match) hits++;
            }
            PROFILE_BLOCK_END(write_result);

            // load next
            total_processed_bytes = output_index * compression / 8;
            source = _mm_loadu_si128((__m128i*)&((uint8_t*)input)[total_processed_bytes]);

            PROFILE_BLOCK_END(whole_loop);
        }
    }

    return hits;
}

int scan_128_alternative(int predicate_key, __m128i* input, size_t input_size, std::vector<bool>& output)
{
    int hits = 0;

    size_t compression = BITS_NEEDED;

    __m128i source = _mm_loadu_si128(input);

    size_t output_index = 0; // current write index of the output array (equals # of decompressed values)
    size_t total_processed_bytes = 0; // holds # of input bytes that have been processed completely

    // shuffle masks
    size_t input_offset[8];
    for (size_t i = 0; i < 8; i++)
    {
        input_offset[i] = (compression * i) / 8;
    }
    size_t correction = input_offset[4];
    for (size_t i = 4; i < 8; i++)
    {
        input_offset[i] -= correction;
    }

    __m128i shuffle_mask[2];
    shuffle_mask[0] = _mm_setr_epi8(
        input_offset[0], input_offset[0] + 1, input_offset[0] + 2, input_offset[0] + 3,
        input_offset[1], input_offset[1] + 1, input_offset[1] + 2, input_offset[1] + 3,
        input_offset[2], input_offset[2] + 1, input_offset[2] + 2, input_offset[2] + 3,
        input_offset[3], input_offset[3] + 1, input_offset[3] + 2, input_offset[3] + 3);
    shuffle_mask[1] = _mm_setr_epi8(
        input_offset[4], input_offset[4] + 1, input_offset[4] + 2, input_offset[4] + 3,
        input_offset[5], input_offset[5] + 1, input_offset[5] + 2, input_offset[5] + 3,
        input_offset[6], input_offset[6] + 1, input_offset[6] + 2, input_offset[6] + 3,
        input_offset[7], input_offset[7] + 1, input_offset[7] + 2, input_offset[7] + 3);

    // clean masks
    size_t padding[8];
    for (size_t i = 0; i < 8; i++)
    {
        padding[i] = (compression * i) % 8;
    }

    __m128i clean_mask[2];
    clean_mask[0] = _mm_setr_epi32(
        ((1 << compression) - 1) << padding[0],
        ((1 << compression) - 1) << padding[1],
        ((1 << compression) - 1) << padding[2],
        ((1 << compression) - 1) << padding[3]);
    clean_mask[1] = _mm_setr_epi32(
        ((1 << compression) - 1) << padding[4],
        ((1 << compression) - 1) << padding[5],
        ((1 << compression) - 1) << padding[6],
        ((1 << compression) - 1) << padding[7]);

    // registers for comparison predicate
    __m128i predicate[2];
    predicate[0] = _mm_setr_epi32(
        predicate_key << padding[0],
        predicate_key << padding[1],
        predicate_key << padding[2],
        predicate_key << padding[3]);
    predicate[1] = _mm_setr_epi32(
        predicate_key << padding[4],
        predicate_key << padding[5],
        predicate_key << padding[6],
        predicate_key << padding[7]);

    while (output_index < input_size)
    {
        {
            size_t mask_index = 0;
            __m128i b = _mm_shuffle_epi8(source, shuffle_mask[mask_index]);
            __m128i c = _mm_and_si128(b, clean_mask[mask_index]);
            __m128i e = _mm_cmpeq_epi32(c, predicate[mask_index]);

            for (size_t i = 0; i < 4; i++)
            {
                bool match = e.m128i_u32[i] == 0xFFFFFFFF;
                output[output_index++] = match;
                if (match) hits++;
            }

            // load next
            total_processed_bytes = output_index * compression / 8;
            source = _mm_loadu_si128((__m128i*)&((uint8_t*)input)[total_processed_bytes]);
        }

        {
            size_t mask_index = 1;
            __m128i b = _mm_shuffle_epi8(source, shuffle_mask[mask_index]);
            __m128i c = _mm_and_si128(b, clean_mask[mask_index]);
            __m128i e = _mm_cmpeq_epi32(c, predicate[mask_index]);

            for (size_t i = 0; i < 4; i++)
            {
                bool match = e.m128i_u32[i] == 0xFFFFFFFF;
                output[output_index++] = match;
                if (match) hits++;
            }

            // load next
            total_processed_bytes = output_index * compression / 8;
            source = _mm_loadu_si128((__m128i*)&((uint8_t*)input)[total_processed_bytes]);
        }
    }

    return hits;
}

#ifdef __AVX__
int scan_256(int predicate_key, __m128i* input, size_t input_size, std::vector<bool>& output)
{
    int hits = 0;

    size_t compression = BITS_NEEDED;
    size_t free_bits = 32 - compression; // most significant bits in result values that must be 0

    //avxi_t source = _mm256_loadu_si256(input);
    __m256i source = _mm256_loadu2_m128i(input, input);

    size_t output_index = 0; // current write index of the output array (equals # of decompressed values)
    size_t total_processed_bytes = 0; // holds # of input bytes that have been processed completely

    // shuffle mask
    size_t input_offset[8];
    for (size_t i = 0; i < 8; i++)
    {
        input_offset[i] = (compression * i) / 8;
    }

    __m256i shuffle_mask = _mm256_setr_epi8(
        input_offset[0], input_offset[0] + 1, input_offset[0] + 2, input_offset[0] + 3,
        input_offset[1], input_offset[1] + 1, input_offset[1] + 2, input_offset[1] + 3,
        input_offset[2], input_offset[2] + 1, input_offset[2] + 2, input_offset[2] + 3,
        input_offset[3], input_offset[3] + 1, input_offset[3] + 2, input_offset[3] + 3,

        input_offset[4], input_offset[4] + 1, input_offset[4] + 2, input_offset[4] + 3,
        input_offset[5], input_offset[5] + 1, input_offset[5] + 2, input_offset[5] + 3,
        input_offset[6], input_offset[6] + 1, input_offset[6] + 2, input_offset[6] + 3,
        input_offset[7], input_offset[7] + 1, input_offset[7] + 2, input_offset[7] + 3);

    // shift mask
    size_t padding[8];
    for (size_t i = 0; i < 8; i++)
    {
        padding[i] = (compression * i) % 8;
    }

    __m256i shift_mask = _mm256_setr_epi32(
        1 << (free_bits - padding[0]),
        1 << (free_bits - padding[1]),
        1 << (free_bits - padding[2]),
        1 << (free_bits - padding[3]),
        1 << (free_bits - padding[4]),
        1 << (free_bits - padding[5]),
        1 << (free_bits - padding[6]),
        1 << (free_bits - padding[7]));

    __m256i predicate = _mm256_set1_epi32(predicate_key);

    while (output_index < input_size)
    {
        __m256i b = _mm256_shuffle_epi8(source, shuffle_mask);
        __m256i c = _mm256_mullo_epi32(b, shift_mask);
        __m256i d = _mm256_srli_epi32(c, 32 - compression);
        __m256i e = _mm256_cmpeq_epi32(d, predicate);

        for (size_t i = 0; i < 8; i++)
        {
            bool match = e.m256i_u32[i] == 0xFFFFFFFF;
            output[output_index++] = match;
            if (match) hits++;
        }

        // load next
        total_processed_bytes = output_index * compression / 8;
        __m128i* next = (__m128i*)&((uint8_t*)input)[total_processed_bytes];
        source = _mm256_loadu2_m128i(next, next);
    }

    return hits;
}
#endif