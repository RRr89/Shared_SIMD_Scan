#pragma once

#if defined(__GNUC__) && (__GNUC__ >= 6) && defined(__x86_64__)
#  define COMPILER_SUPPORTS_AVX512
#endif

#include <iostream>
#include <immintrin.h> // AVVX2
#include <xmmintrin.h> //SSE
#include <emmintrin.h> //SSE 2
#include <smmintrin.h> // SSE4.1
#ifdef COMPILER_SUPPORTS_AVX512
#  include <x86intrin.h>
typedef __m256i  avxi_t;
#else
typedef __m128i  avxi_t;
#endif
typedef avxi_t* avxiptr_t;
#include <cmath>
#include <vector>
#include <bitset>

#define BITS_NEEDED 9

avxiptr_t compress_9bit_input(std::vector<uint16_t> &input);

/*
* Non-vectorized decompression
*/
void decompress_standard(avxiptr_t input, size_t input_size, std::vector<uint16_t>& output);

/*
* SIMD decompression (SSE3; 128bit)
*/
void decompress_128_sweep(__m128i* input, size_t input_size, int* output);
void decompress_128_nosweep(__m128i* input, size_t input_size, int* output);
void decompress_128_9bit(__m128i* input, size_t input_size, int* output);
void decompress_128(__m128i* input, size_t input_size, int* output);
void decompress_128_aligned(__m128i* input, size_t input_size, int* output);
#ifdef COMPILER_SUPPORTS_AVX512
void decompress_256(__m128i* input, size_t input_size, int* output);
void decompress_256_avx2(__m128i* input, size_t input_size, int* output);
#endif

/*
* SIMD DECOMPRESS - decompresses the compressed input use 16 Byte, 4 Byte and Bit Alignment as described in the paper
*
* Input:compressed input, inpute size in terms of number of elements in compressed input, and result buffer
*
* Return: void
*/

void decompress(avxiptr_t buffer, int input_size, int* result_buffer);

/*
* SIMD SCAN - Scans compressed input for a range of predicates  (predicate_low<=key<=predicate_high)
*
* Input: predicate low, predicate high, compressed input, and inpute size in terms of number of elements in
*        compressed input
*
* Return: number of tuples found in that range
*/

int scan(int predicate_low, int predicate_high, avxiptr_t compressed_input, int input_size);

