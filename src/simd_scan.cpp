#include <iostream>
#include <immintrin.h>
#include <cmath>
#include <vector>
#include <bitset>
#include <algorithm>

#include "simd_scan.hpp"

void decompress(__m256i* buffer, int input_size, int* result_buffer)
{
	//To be implemented   
}

int scan(int predicate_low, int predicate_high, __m256i* compressed_input, int input_size)
{
	//To be implemented
	return 0;
}