#pragma once

#include <iostream>
#include <immintrin.h>
#include <cmath>
#include <vector>
#include <bitset>

#define BITS_NEEDED 9


__m256i* compress_9bit_input(std::vector<uint16_t> &input);

/*
* Non-vectorized decompression
*/
void decompress_standard(__m256i* input, size_t input_size, std::vector<uint16_t>& output);

/*
* SIMD decompression (SSE3; 128bit)
*/
void decompress_128(__m128i* input, size_t input_size, int* output);
void decompress_128_1group(__m128i* input, size_t input_size, int* output);
void decompress_128_9bit_1group(__m128i* input, size_t input_size, int* output);

/*
* SIMD DECOMPRESS - decompresses the compressed input use 16 Byte, 4 Byte and Bit Alignment as described in the paper
*
* Input:compressed input, inpute size in terms of number of elements in compressed input, and result buffer
*
* Return: void
*/

void decompress(__m256i* buffer, int input_size, int* result_buffer);

/*
* SIMD SCAN - Scans compressed input for a range of predicates  (predicate_low<=key<=predicate_high)
*
* Input: predicate low, predicate high, compressed input, and inpute size in terms of number of elements in
*        compressed input
*
* Return: number of tuples found in that range
*/

int scan(int predicate_low, int predicate_high, __m256i* compressed_input, int input_size);