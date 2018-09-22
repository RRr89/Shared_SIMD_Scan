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
* padding such that SSE/AVX instruction don't read/write outside of the buffers.
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
void shared_scan_128_standard_unrolled(std::vector<int> const& predicate_keys, __m128i* input, size_t input_size, std::vector<std::vector<uint8_t>>& outputs);
void shared_scan_128_parallel(std::vector<int> const& predicate_keys, __m128i* input, size_t input_size, std::vector<std::vector<uint8_t>>& outputs);

#ifdef __AVX__
void shared_scan_256_sequential(std::vector<int> const& predicate_keys, __m128i* input, size_t input_size, std::vector<std::vector<uint8_t>>& outputs);
void shared_scan_256_standard(std::vector<int> const& predicate_keys, __m128i* input, size_t input_size, std::vector<std::vector<uint8_t>>& outputs);
void shared_scan_256_parallel(std::vector<int> const& predicate_keys, __m128i* input, size_t input_size, std::vector<std::vector<uint8_t>>& outputs);
#endif