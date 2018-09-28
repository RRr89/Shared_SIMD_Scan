#pragma once

#include <iostream>
#include <immintrin.h> 
#include <cmath>
#include <vector>
#include <bitset>
#include <memory>

#include "util.hpp"

#define BITS_NEEDED 9

/*
* Helper functions for calculating the size of the necessary buffers. These sizes include
* padding such that SSE/AVX instructions don't read/write outside of the buffers.
* All returned sizes are in bytes.
*/

constexpr size_t compressed_buffer_size(uint8_t compression, size_t input_array_size)
{
    auto mem_size = compression * input_array_size;
    auto bytes = mem_size / 8 + (mem_size % 8 != 0);
    auto padding = 256;
    return bytes + padding;
}

constexpr size_t decompression_output_buffer_size(size_t input_array_size)
{
    auto size = input_array_size * 4;
    auto padding = 32;
    return size + padding;
}

constexpr size_t scan_output_buffer_size(size_t input_array_size)
{
    auto bytes = input_array_size / 8 + (input_array_size % 8 != 0);
    auto padding = 32;
    return bytes + padding;
}

/* 
* Compression
*/

std::unique_ptr<uint64_t[]> compress_9bit_input(std::vector<uint16_t>& input);

/*
* Non-vectorized decompression
*/
void decompress_unvectorized(__m128i* input, size_t input_size, int* output);

/*
* SIMD decompression (SSE3; 128bit)
*/
void decompress_128_sweep(__m128i* input, size_t input_size, int* output);
void decompress_128_nosweep(__m128i* input, size_t input_size, int* output);
void decompress_128_9bit(__m128i* input, size_t input_size, int* output);
void decompress_128(__m128i* input, size_t input_size, int* output);
void decompress_128_unrolled(__m128i* input, size_t input_size, int* output);
void decompress_128_aligned(__m128i* input, size_t input_size, int* output);

/*
* SIMD decompression (AVX and AVX2; 256bit)
*/

#ifdef __AVX__
void decompress_256(__m128i* input, size_t input_size, int* output);
#endif

#ifdef __AVX2__
void decompress_256_avx2(__m128i* input, size_t input_size, int* output);
#endif

/*
* SIMD SCAN - Scans compressed input for a range of predicates  (predicate_low<=key<=predicate_high)
*
* Input: predicate low, predicate high, compressed input, and inpute size in terms of number of elements in
*        compressed input
*
* Return: number of tuples found in that range
*/

//int scan(int predicate_low, int predicate_high, __m128i* compressed_input, int input_size);

/*
* SIMD scan 
*/
int scan_unvectorized(int predicate_key, __m128i* input, size_t input_size, std::vector<uint8_t>& output);
int scan_128(int predicate_key, __m128i* input, size_t input_size, std::vector<uint8_t>& output);
int scan_128_unrolled(int predicate_key, __m128i* input, size_t input_size, std::vector<uint8_t>& output);

#ifdef __AVX__
int scan_256(int predicate_key, __m128i* input, size_t input_size, std::vector<uint8_t>& output);
int scan_256_unrolled(int predicate_key, __m128i* input, size_t input_size, std::vector<uint8_t>& output);
#endif

/*
* Shared SIMD scan
*/

void shared_scan_128_sequential(std::vector<int> const& predicate_keys, __m128i* input, size_t input_size, std::vector<std::vector<uint8_t>>& outputs);
void shared_scan_128_sequential_unrolled(std::vector<int> const& predicate_keys, __m128i* input, size_t input_size, std::vector<std::vector<uint8_t>>& outputs);
void shared_scan_128_threaded(std::vector<int> const& predicate_keys, __m128i* input, size_t input_size, std::vector<std::vector<uint8_t>>& outputs);
void shared_scan_128_standard(std::vector<int> const& predicate_keys, __m128i* input, size_t input_size, std::vector<std::vector<uint8_t>>& outputs);
void shared_scan_128_standard_v2(std::vector<int> const& predicate_keys, __m128i* input, size_t input_size, std::vector<std::vector<uint8_t>>& outputs);
void shared_scan_128_standard_unrolled(std::vector<int> const& predicate_keys, __m128i* input, size_t input_size, std::vector<std::vector<uint8_t>>& outputs);
void shared_scan_128_parallel(std::vector<int> const& predicate_keys, __m128i* input, size_t input_size, std::vector<std::vector<uint8_t>>& outputs);

#ifdef __AVX__
void shared_scan_256_sequential(std::vector<int> const& predicate_keys, __m128i* input, size_t input_size, std::vector<std::vector<uint8_t>>& outputs);
void shared_scan_256_standard(std::vector<int> const& predicate_keys, __m128i* input, size_t input_size, std::vector<std::vector<uint8_t>>& outputs);
void shared_scan_256_parallel(std::vector<int> const& predicate_keys, __m128i* input, size_t input_size, std::vector<std::vector<uint8_t>>& outputs);
#endif

/*
* Shared SIMD scan with one linear output vector
*/

void shared_scan_128_linear_standard(std::vector<int> const& predicate_keys, __m128i* input, size_t input_size, std::vector<uint8_t>& outputs);
void shared_scan_128_linear_standard_v2(std::vector<int> const& predicate_keys, __m128i* input, size_t input_size, std::vector<uint8_t>& outputs);
void shared_scan_128_linear_simple(std::vector<int> const& predicate_keys, __m128i* input, size_t input_size, std::vector<uint8_t>& output);

template <size_t NUM>
void shared_scan_128_linear_static(std::vector<int> const& predicate_keys, __m128i* input, size_t input_size, std::vector<uint8_t>& output)
{
    uint8_t* output_d = output.data();

    size_t compression = BITS_NEEDED;

    __m128i source = _mm_loadu_si128(input);

    size_t output_index = 0; // current write index of the output array (equals # of decompressed values)

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
    __m128i predicates[NUM*2];
    for (int i=0; i<NUM; ++i)
    {
        const int& predicate_key = predicate_keys[i];
        predicates[i*2] = _mm_setr_epi32(
            predicate_key << padding[0],
            predicate_key << padding[1],
            predicate_key << padding[2],
            predicate_key << padding[3]);
        predicates[i*2+1] = _mm_setr_epi32(
            predicate_key << padding[4],
            predicate_key << padding[5],
            predicate_key << padding[6],
            predicate_key << padding[7]);
    }
    
    size_t oidx = 0;

    while (output_index < input_size)
    {
        uint8_t out[NUM];

        {
            const size_t mask_index = 0;
            __m128i b = _mm_shuffle_epi8(source, shuffle_mask[mask_index]);
            __m128i c = _mm_and_si128(b, clean_mask[mask_index]);

            for (int i = 0; i < NUM; ++i)
            {
                __m128i e = _mm_cmpeq_epi32(c, predicates[i*2]);
                out[i] = _mm_movemask_ps(_mm_castsi128_ps(e));
            }

            // load next
            output_index += 4;
            size_t total_processed_bytes = output_index * compression / 8;
            source = _mm_loadu_si128((__m128i*)&((uint8_t*)input)[total_processed_bytes]);
        }

        {
            const size_t mask_index = 1;
            __m128i b = _mm_shuffle_epi8(source, shuffle_mask[mask_index]);
            __m128i c = _mm_and_si128(b, clean_mask[mask_index]);

            for (int i = 0; i < NUM; ++i)
            {
                __m128i e = _mm_cmpeq_epi32(c, predicates[i*2+1]);
                out[i] |= (_mm_movemask_ps(_mm_castsi128_ps(e)) << 4);
            }

            // load next
            output_index += 4;
            size_t total_processed_bytes = output_index * compression / 8;
            source = _mm_loadu_si128((__m128i*)&((uint8_t*)input)[total_processed_bytes]);
        }

        memcpy(output_d + oidx, out, NUM * sizeof(uint8_t));
        oidx += NUM * sizeof(uint8_t);
    }
}

