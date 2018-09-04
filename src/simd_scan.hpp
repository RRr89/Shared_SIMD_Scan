#pragma once

#include <iostream>
#include <immintrin.h> 
#include <cmath>
#include <vector>
#include <bitset>
#include <memory>

#define BITS_NEEDED 9

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
int scan_unvectorized(int predicate_key, __m128i* input, size_t input_size, std::vector<bool>& output);
int scan_128(int predicate_key, __m128i* input, size_t input_size, std::vector<bool>& output);

int scan_128_alternative(int predicate_key, __m128i* input, size_t input_size, std::vector<bool>& output);

#ifdef __AVX__
int scan_256(int predicate_key, __m128i* input, size_t input_size, std::vector<bool>& output);
#endif

/*
* Shared SIMD scan
*/

void shared_scan_128_sequential(std::vector<int> const& predicate_keys, __m128i* input, size_t input_size, std::vector<std::vector<bool>>& outputs);
void shared_scan_128_threaded(std::vector<int> const& predicate_keys, __m128i* input, size_t input_size, std::vector<std::vector<bool>>& outputs);
void shared_scan_128_horizontal(std::vector<int> const& predicate_keys, __m128i* input, size_t input_size, std::vector<std::vector<bool>>& outputs);
void shared_scan_128_vertical(std::vector<int> const& predicate_keys, __m128i* input, size_t input_size, std::vector<std::vector<bool>>& outputs);

#ifdef __AVX__
void shared_scan_256_horizontal(std::vector<int> const& predicate_keys, __m128i* input, size_t input_size, std::vector<std::vector<bool>>& outputs);
void shared_scan_256_vertical(std::vector<int> const& predicate_keys, __m128i* input, size_t input_size, std::vector<std::vector<bool>>& outputs);
#endif