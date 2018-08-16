#pragma once

#include <iostream>
#include <immintrin.h> 
#include <cmath>
#include <vector>
#include <bitset>

#define BITS_NEEDED 9

void* compress_9bit_input(std::vector<uint16_t> &input);

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

int scan_unvectorized(int predicate_key, __m128i* input, size_t input_size, std::vector<bool>& output);