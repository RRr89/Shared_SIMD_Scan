#include <immintrin.h>
#include <algorithm>
#include <cstring> // memcpy

#include "simd_scan.hpp"
#include "simd_scan_commons.hpp"
#include "util.hpp"

void shared_scan_128_sequential(std::vector<int> const& predicate_keys, __m128i* input, size_t input_size, std::vector<std::vector<uint8_t>>& outputs)
{
    for (size_t i = 0; i < predicate_keys.size(); i++)
    {
        scan_128(predicate_keys[i], input, input_size, outputs[i]);
    }
}

void shared_scan_128_sequential_unrolled(std::vector<int> const& predicate_keys, __m128i* input, size_t input_size, std::vector<std::vector<uint8_t>>& outputs)
{
    for (size_t i = 0; i < predicate_keys.size(); i++)
    {
        scan_128_unrolled(predicate_keys[i], input, input_size, outputs[i]);
    }
}

void shared_scan_128_threaded(std::vector<int> const& predicate_keys, __m128i* input, size_t input_size, std::vector<std::vector<uint8_t>>& outputs)
{
    #pragma omp parallel for
    for (int i = 0; i < predicate_keys.size(); i++)
    {
        scan_128(predicate_keys[i], input, input_size, outputs[i]);
    }
}

void shared_scan_128_standard(std::vector<int> const& predicate_keys, __m128i* input, size_t input_size, std::vector<std::vector<uint8_t>>& outputs)
{
    size_t predicate_key_count = predicate_keys.size();
    size_t compression = BITS_NEEDED;

    __m128i source = _mm_loadu_si128(input);

    size_t output_index = 0; // current write index of the output array

    __m128i shuffle_mask[2];
    generate_shuffle_mask_128(compression, shuffle_mask);
 
    __m128i shift_mask[2];
    generate_shift_masks_128(compression, shift_mask);

    // buffer for partial output bytes
    auto output_bytes = std::make_unique<uint8_t[]>(predicate_key_count);

    while (8 * output_index < input_size)
    {
        {
            size_t mask_index = 0;
            __m128i b = _mm_shuffle_epi8(source, shuffle_mask[mask_index]);
            __m128i c = _mm_mullo_epi32(b, shift_mask[mask_index]);
            __m128i d = _mm_srli_epi32(c, 32 - compression);

            for (size_t key_id = 0; key_id < predicate_key_count; key_id++)
            {
                __m128i predicate = _mm_set1_epi32(predicate_keys[key_id]);
                __m128i e = _mm_cmpeq_epi32(d, predicate);

                uint8_t matches = _mm_movemask_ps(_mm_castsi128_ps(e));

                output_bytes[key_id] = matches;
            }

            // load next
            size_t total_processed_bytes = (8 * output_index + 4) * compression / 8;
            source = _mm_loadu_si128((__m128i*)&((uint8_t*)input)[total_processed_bytes]);
        }

        {
            size_t mask_index = 1;
            __m128i b = _mm_shuffle_epi8(source, shuffle_mask[mask_index]);
            __m128i c = _mm_mullo_epi32(b, shift_mask[mask_index]);
            __m128i d = _mm_srli_epi32(c, 32 - compression);

            for (size_t key_id = 0; key_id < predicate_key_count; key_id++)
            {
                __m128i predicate = _mm_set1_epi32(predicate_keys[key_id]);
                __m128i e = _mm_cmpeq_epi32(d, predicate);

                uint8_t matches = _mm_movemask_ps(_mm_castsi128_ps(e));

                outputs[key_id][output_index] = output_bytes[key_id] | (matches << 4);
            }

            // load next
            size_t total_processed_bytes = (8 * output_index + 8) * compression / 8;
            source = _mm_loadu_si128((__m128i*)&((uint8_t*)input)[total_processed_bytes]);
        }

        output_index += 1;
    }
}


void shared_scan_128_standard_v2(std::vector<int> const& predicate_keys, __m128i* input, size_t input_size, std::vector<std::vector<uint8_t>>& outputs)
{
    size_t predicate_key_count = predicate_keys.size();
    size_t compression = BITS_NEEDED;

    __m128i source = _mm_loadu_si128(input);

    size_t output_index = 0; // current write index of the output array

    __m128i shuffle_mask[2];
    generate_shuffle_mask_128(compression, shuffle_mask);
 
    __m128i shift_mask[2];
    generate_shift_masks_128(compression, shift_mask);

    while (8 * output_index < input_size)
    {
        __m128i d1, d2;

        {
            size_t mask_index = 0;
            __m128i b = _mm_shuffle_epi8(source, shuffle_mask[mask_index]);
            __m128i c = _mm_mullo_epi32(b, shift_mask[mask_index]);
            d1 = _mm_srli_epi32(c, 32 - compression);

            size_t total_processed_bytes = (8 * output_index + 4) * compression / 8;
            source = _mm_loadu_si128((__m128i*)&((uint8_t*)input)[total_processed_bytes]);
        }

        {
            size_t mask_index = 1;
            __m128i b = _mm_shuffle_epi8(source, shuffle_mask[mask_index]);
            __m128i c = _mm_mullo_epi32(b, shift_mask[mask_index]);
            d2 = _mm_srli_epi32(c, 32 - compression);

            size_t total_processed_bytes = (8 * output_index + 8) * compression / 8;
            source = _mm_loadu_si128((__m128i*)&((uint8_t*)input)[total_processed_bytes]);
        }

        for (size_t key_id = 0; key_id < predicate_key_count; key_id++)
        {
            __m128i predicate = _mm_set1_epi32(predicate_keys[key_id]);
            __m128i e1 = _mm_cmpeq_epi32(d1, predicate);
            __m128i e2 = _mm_cmpeq_epi32(d2, predicate);

            uint8_t matches1 = _mm_movemask_ps(_mm_castsi128_ps(e1));
            uint8_t matches2 = _mm_movemask_ps(_mm_castsi128_ps(e2));

            outputs[key_id][output_index] = matches1 | (matches2 << 4);
        }

        output_index += 1;
    }
}

inline void __shared_scan_128_step(std::unique_ptr<uint32_t[]>& output, size_t const& offset, __m128i const& shuffle_mask,
    __m128i const& shift_mask, size_t const& predicate_key_count, std::vector<int> const& predicate_keys, 
    size_t const& output_index, size_t const& compression, __m128i* input, __m128i& source)
{
    size_t mask_index = 0;
    __m128i b = _mm_shuffle_epi8(source, shuffle_mask);
    __m128i c = _mm_mullo_epi32(b, shift_mask);
    __m128i d = _mm_srli_epi32(c, 32 - compression);

    for (size_t key_id = 0; key_id < predicate_key_count; key_id++)
    {
        __m128i predicate = _mm_set1_epi32(predicate_keys[key_id]);
        __m128i e = _mm_cmpeq_epi32(d, predicate);

        output[key_id] |= _mm_movemask_ps(_mm_castsi128_ps(e)) << offset;
    }

    // load next
    size_t total_processed_bytes = (32 * output_index + offset + 4) * compression / 8;
    source = _mm_loadu_si128((__m128i*)&((uint8_t*)input)[total_processed_bytes]);
}

void shared_scan_128_standard_unrolled(std::vector<int> const& predicate_keys, __m128i* input, size_t input_size, std::vector<std::vector<uint8_t>>& outputs)
{
    size_t predicate_key_count = predicate_keys.size();
    size_t compression = BITS_NEEDED;

    __m128i source = _mm_loadu_si128(input);

    size_t output_index = 0; // current write index of the output array

    __m128i shuffle_mask[2];
    generate_shuffle_mask_128(compression, shuffle_mask);

    __m128i shift_mask[2];
    generate_shift_masks_128(compression, shift_mask);

    // buffer for partial outputs
    auto output = std::make_unique<uint32_t[]>(predicate_key_count);

    while (32 * output_index < input_size)
    {
        __shared_scan_128_step(output,  0, shuffle_mask[0], shift_mask[0], predicate_key_count, predicate_keys, output_index, compression, input, source);
        __shared_scan_128_step(output,  4, shuffle_mask[1], shift_mask[1], predicate_key_count, predicate_keys, output_index, compression, input, source);

        __shared_scan_128_step(output,  8, shuffle_mask[0], shift_mask[0], predicate_key_count, predicate_keys, output_index, compression, input, source);
        __shared_scan_128_step(output, 12, shuffle_mask[1], shift_mask[1], predicate_key_count, predicate_keys, output_index, compression, input, source);

        __shared_scan_128_step(output, 16, shuffle_mask[0], shift_mask[0], predicate_key_count, predicate_keys, output_index, compression, input, source);
        __shared_scan_128_step(output, 20, shuffle_mask[1], shift_mask[1], predicate_key_count, predicate_keys, output_index, compression, input, source);

        __shared_scan_128_step(output, 24, shuffle_mask[0], shift_mask[0], predicate_key_count, predicate_keys, output_index, compression, input, source);
        __shared_scan_128_step(output, 28, shuffle_mask[1], shift_mask[1], predicate_key_count, predicate_keys, output_index, compression, input, source);

        for (int key_id = 0; key_id < predicate_key_count; key_id++)
        {
            reinterpret_cast<uint32_t*>(outputs[key_id].data())[output_index] = output[key_id];
            output[key_id] = 0;
        }

        output_index += 1;
    }
}

// based on scan_unvectorized, just 4 times in parallel with SSE!
void shared_scan_128_parallel(std::vector<int> const& predicate_keys, __m128i* input, size_t input_size, std::vector<std::vector<uint8_t>>& outputs)
{
    size_t predicate_key_count = predicate_keys.size();
    uint32_t* in = reinterpret_cast<uint32_t*>(input);
    auto compression = BITS_NEEDED;
    auto mem_size = compression * input_size;
    size_t array_size = ceil((double)mem_size / (8 * sizeof(uint32_t)));

    __m128i mask = _mm_set1_epi32((1 << compression) - 1);

    // process in groups of (maximum) 4 predicates
    for (size_t key_id = 0; key_id < predicate_key_count; key_id += 4)
    {
        __m128i current = _mm_setzero_si128();
        size_t overflow_bits = 0;

        size_t oi = 0; // number of result bits written

        __m128i output = _mm_setzero_si128(); // holds 32 output bits for each compartment
        size_t out_bits_used = 0; // can maybe be optimized away
        __m128i output_match_mask = _mm_set1_epi32(1); // used for masking out one bit in the comparison result register

        __m128i predicate_key = _mm_setr_epi32(
            predicate_keys[std::min(key_id, predicate_key_count - 1)], 
            predicate_keys[std::min(key_id + 1, predicate_key_count - 1)],
            predicate_keys[std::min(key_id + 2, predicate_key_count - 1)],
            predicate_keys[std::min(key_id + 3, predicate_key_count - 1)]);

        for (size_t i = 0; i < array_size; i++)
        {
            __m128i raw_next = _mm_set1_epi32(in[i]);
            __m128i shr_amount = _mm_set_epi64x(0, overflow_bits);
            current = _mm_srl_epi32(raw_next, shr_amount);

            size_t unread_bits = (8 * sizeof(uint32_t)) - overflow_bits;
            while (unread_bits >= compression)
            {
                // decompress and match against predicate
                __m128i decompressed_element = _mm_and_si128(current, mask);
                __m128i match = _mm_cmpeq_epi32(decompressed_element, predicate_key);

                // mask out one bit from the comparison result and OR it to the output registers
                __m128i masked_match = _mm_and_si128(match, output_match_mask);
                output = _mm_or_si128(output, masked_match);
                output_match_mask = _mm_slli_epi32(output_match_mask, 1);
                out_bits_used += 1;

                // if the output register is filled, write results to the output vector
                if (out_bits_used == 32)
                {
                    uint32_t result_reg[4];
                    _mm_storeu_si128((__m128i*)&result_reg, output);

                    for (size_t key_offset = 0; key_offset < 4 && key_id + key_offset < predicate_key_count; key_offset++)
                    {
                        memcpy(outputs[key_id + key_offset].data() + oi, &result_reg[key_offset], sizeof(uint32_t));
                    }
                    oi += 4;

                    // reset these registers
                    output = _mm_setzero_si128();
                    output_match_mask = _mm_set1_epi32(1);
                    out_bits_used = 0;
                }

                if (oi >= input_size)
                {
                    goto nextBatch;
                }

                current = _mm_srli_epi32(current, compression);
                unread_bits -= compression;
            }

            // handle overlapping element
            if (unread_bits != 0)
            {
                __m128i next = _mm_set1_epi32(in[i + 1]);
                __m128i shl_amount = _mm_set_epi64x(0, unread_bits);
                current = _mm_or_si128(current, _mm_sll_epi32(next, shl_amount));

                __m128i decompressed_element = _mm_and_si128(current, mask);
                __m128i match = _mm_cmpeq_epi32(decompressed_element, predicate_key);

                // mask out one bit from the comparison result and OR it to the output registers
                __m128i masked_match = _mm_and_si128(match, output_match_mask);
                output = _mm_or_si128(output, masked_match);
                output_match_mask = _mm_slli_epi32(output_match_mask, 1);
                out_bits_used += 1;

                // if the output register is filled, write results to the output vector
                if (out_bits_used == 32)
                {
                    uint32_t result_reg[4];
                    _mm_storeu_si128((__m128i*)&result_reg, output);

                    for (size_t key_offset = 0; key_offset < 4 && key_id + key_offset < predicate_key_count; key_offset++)
                    {
                        memcpy(outputs[key_id + key_offset].data() + oi, &result_reg[key_offset], sizeof(uint32_t));
                    }
                    oi += 4;

                    // reset these registers
                    output = _mm_setzero_si128();
                    output_match_mask = _mm_set1_epi32(1);
                    out_bits_used = 0;
                }


                overflow_bits = compression - unread_bits;
            }
            else
            {
                overflow_bits = 0;
            }
        }

        if (out_bits_used != 0) 
        {
            uint32_t result_reg[4];
            _mm_storeu_si128((__m128i*)&result_reg, output);

            for (size_t key_offset = 0; key_offset < 4 && key_id + key_offset < predicate_key_count; key_offset++)
            {
                memcpy(outputs[key_id + key_offset].data() + oi, &result_reg[key_offset], sizeof(uint32_t));
            }
        }

        nextBatch:;
    }
}

#ifdef __AVX__
void shared_scan_256_sequential(std::vector<int> const& predicate_keys, __m128i* input, size_t input_size, std::vector<std::vector<uint8_t>>& outputs)
{
    for (size_t i = 0; i < predicate_keys.size(); i++)
    {
        scan_256(predicate_keys[i], input, input_size, outputs[i]);
    }
}

void shared_scan_256_standard(std::vector<int> const& predicate_keys, __m128i* input, size_t input_size, std::vector<std::vector<uint8_t>>& outputs)
{
    size_t predicate_key_count = predicate_keys.size();
    size_t compression = BITS_NEEDED;

    //avxi_t source = _mm256_loadu_si256(input);
    __m256i source = _mm256_loadu2_m128i(input, input);

    size_t output_index = 0; // current write index of the output array

    __m256i shuffle_mask = generate_shuffle_mask_256(compression);

    __m256i shift_mask = generate_shift_mask_256(compression);

    while (8 * output_index < input_size)
    {
        __m256i b = _mm256_shuffle_epi8(source, shuffle_mask);
        __m256i c = _mm256_mullo_epi32(b, shift_mask);
        __m256i d = _mm256_srli_epi32(c, 32 - compression);

        for (size_t key_id = 0; key_id < predicate_key_count; key_id++)
        {
            __m256i predicate = _mm256_set1_epi32(predicate_keys[key_id]);
            __m256i e = _mm256_cmpeq_epi32(d, predicate);

            int matches = _mm256_movemask_ps(_mm256_castsi256_ps(e));
            outputs[key_id][output_index] = matches;
        }

        // load next
        output_index += 1;
        size_t total_processed_bytes = (8 * output_index) * compression / 8;
        __m128i* next = (__m128i*)&((uint8_t*)input)[total_processed_bytes];
        source = _mm256_loadu2_m128i(next, next);
    }
}

// based on scan_unvectorized, just 8 times in parallel with AVX!
void shared_scan_256_parallel(std::vector<int> const& predicate_keys, __m128i* input, size_t input_size, std::vector<std::vector<uint8_t>>& outputs)
{
    size_t predicate_key_count = predicate_keys.size();
    uint32_t* in = reinterpret_cast<uint32_t*>(input);
    auto compression = BITS_NEEDED;
    auto mem_size = compression * input_size;
    size_t array_size = ceil((double)mem_size / (8 * sizeof(uint32_t)));

    __m256i mask = _mm256_set1_epi32((1 << compression) - 1);

    // process in groups of (maximum) 8 predicates
    for (size_t key_id = 0; key_id < predicate_key_count; key_id += 8)
    {
        __m256i current = _mm256_setzero_si256();
        size_t overflow_bits = 0;

        size_t oi = 0; // number of result bits written

        __m256i output = _mm256_setzero_si256(); // holds 32 output bits for each compartment
        size_t out_bits_used = 0; // can maybe be optimized away
        __m256i output_match_mask = _mm256_set1_epi32(1); // used for masking out one bit in the comparison result register

        __m256i predicate_key = _mm256_setr_epi32(
            predicate_keys[std::min(key_id, predicate_key_count - 1)],
            predicate_keys[std::min(key_id + 1, predicate_key_count - 1)],
            predicate_keys[std::min(key_id + 2, predicate_key_count - 1)],
            predicate_keys[std::min(key_id + 3, predicate_key_count - 1)],
            predicate_keys[std::min(key_id + 4, predicate_key_count - 1)],
            predicate_keys[std::min(key_id + 5, predicate_key_count - 1)],
            predicate_keys[std::min(key_id + 6, predicate_key_count - 1)],
            predicate_keys[std::min(key_id + 7, predicate_key_count - 1)]);

        for (size_t i = 0; i < array_size; i++)
        {
            __m256i raw_next = _mm256_set1_epi32(in[i]);
            __m128i shr_amount = _mm_set_epi64x(0, overflow_bits);
            current = _mm256_srl_epi32(raw_next, shr_amount);

            size_t unread_bits = (8 * sizeof(uint32_t)) - overflow_bits;
            while (unread_bits >= compression)
            {
                // decompress and match against predicate
                __m256i decompressed_element = _mm256_and_si256(current, mask);
                __m256i match = _mm256_cmpeq_epi32(decompressed_element, predicate_key);

                // mask out one bit from the comparison result and OR it to the output registers
                __m256i masked_match = _mm256_and_si256(match, output_match_mask);
                output = _mm256_or_si256(output, masked_match);
                output_match_mask = _mm256_slli_epi32(output_match_mask, 1);
                out_bits_used += 1;

                // if the output register is filled, write results to the output vector
                if (out_bits_used == 32)
                {
                    uint32_t result_reg[8];
                    _mm256_storeu_si256((__m256i*)&result_reg, output);

                    for (size_t key_offset = 0; key_offset < 8 && key_id + key_offset < predicate_key_count; key_offset++)
                    {
                        memcpy(outputs[key_id + key_offset].data() + oi, &result_reg[key_offset], sizeof(uint32_t));
                    }
                    oi += 4;

                    // reset these registers
                    output = _mm256_setzero_si256();
                    output_match_mask = _mm256_set1_epi32(1);
                    out_bits_used = 0;
                }


                if (oi >= input_size)
                {
                    goto nextBatch;
                }

                current = _mm256_srli_epi32(current, compression);
                unread_bits -= compression;
            }

            // handle overlapping element
            if (unread_bits != 0)
            {
                __m256i next = _mm256_set1_epi32(in[i + 1]);
                __m128i shl_amount = _mm_set_epi64x(0, unread_bits);
                current = _mm256_or_si256(current, _mm256_sll_epi32(next, shl_amount));

                __m256i decompressed_element = _mm256_and_si256(current, mask);
                __m256i match = _mm256_cmpeq_epi32(decompressed_element, predicate_key);

                // mask out one bit from the comparison result and OR it to the output registers
                __m256i masked_match = _mm256_and_si256(match, output_match_mask);
                output = _mm256_or_si256(output, masked_match);
                output_match_mask = _mm256_slli_epi32(output_match_mask, 1);
                out_bits_used += 1;

                // if the output register is filled, write results to the output vector
                if (out_bits_used == 32)
                {
                    uint32_t result_reg[8];
                    _mm256_storeu_si256((__m256i*)&result_reg, output);

                    for (size_t key_offset = 0; key_offset < 8 && key_id + key_offset < predicate_key_count; key_offset++)
                    {
                        memcpy(outputs[key_id + key_offset].data() + oi, &result_reg[key_offset], sizeof(uint32_t));
                    }

                    // reset these registers
                    output = _mm256_setzero_si256();
                    output_match_mask = _mm256_set1_epi32(1);
                    out_bits_used = 0;
                }

                overflow_bits = compression - unread_bits;
            }
            else
            {
                overflow_bits = 0;
            }
        }

        if (out_bits_used != 0)
        {
            uint32_t result_reg[8];
            _mm256_storeu_si256((__m256i*)&result_reg, output);

            for (size_t key_offset = 0; key_offset < 8 && key_id + key_offset < predicate_key_count; key_offset++)
            {
                memcpy(outputs[key_id + key_offset].data() + oi, &result_reg[key_offset], sizeof(uint32_t));
            }
        }

    nextBatch:;
    }
}
#endif

