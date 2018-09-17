#include <immintrin.h>
#include <algorithm>

#include "simd_scan.hpp"

void shared_scan_128_sequential(std::vector<int> const& predicate_keys, __m128i* input, size_t input_size, std::vector<std::vector<bool>>& outputs)
{
    for (size_t i = 0; i < predicate_keys.size(); i++)
    {
        scan_128(predicate_keys[i], input, input_size, outputs[i]);
    }
}

void shared_scan_128_threaded(std::vector<int> const& predicate_keys, __m128i* input, size_t input_size, std::vector<std::vector<bool>>& outputs)
{
    #pragma omp parallel for
    for (int i = 0; i < predicate_keys.size(); i++)
    {
        scan_128(predicate_keys[i], input, input_size, outputs[i]);
    }
}

// based on scan_unvectorized, just 4 times in parallel with SSE!
void shared_scan_128_horizontal(std::vector<int> const& predicate_keys, __m128i* input, size_t input_size, std::vector<std::vector<bool>>& outputs)
{
    size_t predicate_key_count = predicate_keys.size();
    uint32_t* in = reinterpret_cast<uint32_t*>(input);
    auto compression = BITS_NEEDED;
    auto mem_size = compression * input_size;
    size_t array_size = ceil((double)mem_size / (8 * sizeof(uint32_t)));

    __m128i mask = _mm_set1_epi32((1 << compression) - 1);

    for (size_t key_id = 0; key_id < predicate_key_count; key_id += 4)
    {
        __m128i current = _mm_setzero_si128();
        size_t overflow_bits = 0;

        size_t oi = 0;

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
                __m128i decompressed_element = _mm_and_si128(current, mask);
                __m128i match = _mm_cmpeq_epi32(decompressed_element, predicate_key);

                for (size_t out_offset = 0; out_offset < 4 && key_id + out_offset < predicate_key_count; out_offset++)
                {
                    outputs[key_id + out_offset][oi] = 
#ifdef _MSC_VER
                        match.m128i_u32[out_offset]
#else
                        match[out_offset]
#endif
                    > 0;
                }

                oi++;
                if (oi == input_size)
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

                for (size_t out_offset = 0; out_offset < 4 && key_id + out_offset < predicate_key_count; out_offset++)
                {
                    outputs[key_id + out_offset][oi] = 
#ifdef _MSC_VER
                        match.m128i_u32[out_offset]
#else
                        match[out_offset]
#endif
                    > 0;
                }

                oi++;

                overflow_bits = compression - unread_bits;
            }
            else
            {
                overflow_bits = 0;
            }
        }

        nextBatch:;
    }
}

void shared_scan_128_vertical(std::vector<int> const& predicate_keys, __m128i* input, size_t input_size, std::vector<std::vector<bool>>& outputs)
{
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

    while (output_index < input_size)
    {
        {
            size_t mask_index = 0;
            __m128i b = _mm_shuffle_epi8(source, shuffle_mask[mask_index]);
            __m128i c = _mm_mullo_epi32(b, shift_mask[mask_index]);
            __m128i d = _mm_srli_epi32(c, 32 - compression);

            for (size_t key_id = 0; key_id < predicate_keys.size(); key_id++)
            {
                __m128i predicate = _mm_set1_epi32(predicate_keys[key_id]);
                __m128i e = _mm_cmpeq_epi32(d, predicate);

                for (size_t i = 0; i < 4; i++)
                {
                    outputs[key_id][output_index + i] =
#ifdef _MSC_VER
                        e.m128i_u32[i]
#else
                        e[i]
#endif
                    > 0;
                }
            }

            output_index += 4;

            // load next
            total_processed_bytes = output_index * compression / 8;
            source = _mm_loadu_si128((__m128i*)&((uint8_t*)input)[total_processed_bytes]);
        }

        {
            size_t mask_index = 1;
            __m128i b = _mm_shuffle_epi8(source, shuffle_mask[mask_index]);
            __m128i c = _mm_mullo_epi32(b, shift_mask[mask_index]);
            __m128i d = _mm_srli_epi32(c, 32 - compression);

            for (size_t key_id = 0; key_id < predicate_keys.size(); key_id++)
            {
                __m128i predicate = _mm_set1_epi32(predicate_keys[key_id]);
                __m128i e = _mm_cmpeq_epi32(d, predicate);

                for (size_t i = 0; i < 4; i++)
                {
                    outputs[key_id][output_index + i] =
#ifdef _MSC_VER
                        e.m128i_u32[i]
#else
                        e[i]
#endif
                    > 0;
                }
            }

            output_index += 4;

            // load next
            total_processed_bytes = output_index * compression / 8;
            source = _mm_loadu_si128((__m128i*)&((uint8_t*)input)[total_processed_bytes]);
        }
    }
}

#ifdef __AVX__
// based on scan_unvectorized, just 8 times in parallel with AVX!
void shared_scan_256_horizontal(std::vector<int> const& predicate_keys, __m128i* input, size_t input_size, std::vector<std::vector<bool>>& outputs)
{
    size_t predicate_key_count = predicate_keys.size();
    uint32_t* in = reinterpret_cast<uint32_t*>(input);
    auto compression = BITS_NEEDED;
    auto mem_size = compression * input_size;
    size_t array_size = ceil((double)mem_size / (8 * sizeof(uint32_t)));

    __m256i mask = _mm256_set1_epi32((1 << compression) - 1);

    for (size_t key_id = 0; key_id < predicate_key_count; key_id += 8)
    {
        __m256i current = _mm256_setzero_si256();
        size_t overflow_bits = 0;

        size_t oi = 0;

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
                __m256i decompressed_element = _mm256_and_si256(current, mask);
                __m256i match = _mm256_cmpeq_epi32(decompressed_element, predicate_key);

                for (size_t out_offset = 0; out_offset < 8 && key_id + out_offset < predicate_key_count; out_offset++)
                {
                    outputs[key_id + out_offset][oi] = match.m256i_u32[out_offset] > 0;
                }

                oi++;
                if (oi == input_size)
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

                for (size_t out_offset = 0; out_offset < 8 && key_id + out_offset < predicate_key_count; out_offset++)
                {
                    outputs[key_id + out_offset][oi] = match.m256i_u32[out_offset] > 0;
                }

                oi++;

                overflow_bits = compression - unread_bits;
            }
            else
            {
                overflow_bits = 0;
            }
        }

    nextBatch:;
    }
}

void shared_scan_256_vertical(std::vector<int> const& predicate_keys, __m128i* input, size_t input_size, std::vector<std::vector<bool>>& outputs)
{
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

    while (output_index < input_size)
    {
        __m256i b = _mm256_shuffle_epi8(source, shuffle_mask);
        __m256i c = _mm256_mullo_epi32(b, shift_mask);
        __m256i d = _mm256_srli_epi32(c, 32 - compression);

        for (size_t key_id = 0; key_id < predicate_keys.size(); key_id++)
        {
            __m256i predicate = _mm256_set1_epi32(predicate_keys[key_id]);
            __m256i e = _mm256_cmpeq_epi32(d, predicate);

            for (size_t i = 0; i < 8; i++)
            {
                outputs[key_id][output_index + i] = e.m256i_u32[i] > 0;
            }
        }

        output_index += 8;

        // load next
        total_processed_bytes = output_index * compression / 8;
        __m128i* next = (__m128i*)&((uint8_t*)input)[total_processed_bytes];
        source = _mm256_loadu2_m128i(next, next);
    }
}
#endif
