#pragma once

#include <iostream>
#include <tmmintrin.h>
#include <cmath>
#include <vector>
#include <bitset>

#define BITS_NEEDED 9


__m256i* compress_9bit_input(std::vector<uint16_t> &input);


/*
*SIMD DECOMPRESS - decompresses the compressed input use 16 Byte, 4 Byte and Bit Alignment as described in the paper
* Input:compressed input, inpute size in terms of number of elements in compressed input, and result buffer
*
* Return: void
*/

void decompress(__m256i *buffer, int input_size, int* result_buffer);