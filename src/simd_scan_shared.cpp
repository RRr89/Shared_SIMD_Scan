#include <immintrin.h>
#include <algorithm>

#include "simd_scan.hpp"

void shared_scan_128_sequential(std::vector<int>& predicate_keys, __m128i* input, size_t input_size, std::vector<std::vector<bool>>& outputs)
{
    for (size_t i = 0; i < predicate_keys.size(); i++)
    {
        scan_128(predicate_keys[i], input, input_size, outputs[i]);
    }
}

void shared_scan_128_threaded(std::vector<int>& predicate_keys, __m128i* input, size_t input_size, std::vector<std::vector<bool>>& outputs)
{
    #pragma omp parallel for
    for (int i = 0; i < predicate_keys.size(); i++)
    {
        scan_128(predicate_keys[i], input, input_size, outputs[i]);
    }
}

// based on scan_unvectorized, just 4 times in parallel with SSE!
void shared_scan_128_horizontal(std::vector<int>& predicate_keys, __m128i* input, size_t input_size, std::vector<std::vector<bool>>& outputs)
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
                    outputs[key_id + out_offset][oi] = match.m128i_u32[out_offset] > 0;
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
                    outputs[key_id + out_offset][oi] = match.m128i_u32[out_offset] > 0;
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

#ifdef __AVX__
// based on scan_unvectorized, just 8 times in parallel with AVX!
void shared_scan_256_horizontal(std::vector<int>& predicate_keys, __m128i* input, size_t input_size, std::vector<std::vector<bool>>& outputs)
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
#endif