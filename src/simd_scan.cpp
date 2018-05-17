#include <iostream>
#include <tmmintrin.h>
#include <immintrin.h>
#include <cmath>
#include <vector>
#include <bitset>

#include "simd_scan.hpp"

__m256i* compress_9bit_input(std::vector<uint16_t>& input)
{
	auto bits_needed = BITS_NEEDED;
	auto mem_size = bits_needed * input.size();
	int array_size = ceil((double)mem_size / 64);
	auto buffer = new long long[array_size]();

	int remaining_buffer_size = 64;
	int idx_ = 0;
	for (size_t i = 0; i < input.size(); i++)
	{
		long long tmp_buffer = 0;
		tmp_buffer = tmp_buffer | input[i];
		tmp_buffer = tmp_buffer << (i * bits_needed);
		buffer[idx_] = buffer[idx_] | tmp_buffer;
		remaining_buffer_size -= bits_needed;

		if (remaining_buffer_size == 0)
		{
			idx_++;
			remaining_buffer_size = 64;
			continue;
		}
		else if (remaining_buffer_size < bits_needed)
		{
			//logic to handle overflow_bits
			i++;
			tmp_buffer = 0;
			tmp_buffer = tmp_buffer | input[i];
			tmp_buffer = tmp_buffer << (64 - remaining_buffer_size);
			buffer[idx_] = buffer[idx_] | tmp_buffer;

			idx_++;

			// Second half
			tmp_buffer = 0;
			tmp_buffer = tmp_buffer | input[i];
			tmp_buffer = tmp_buffer >> remaining_buffer_size;
			buffer[idx_] = buffer[idx_] | tmp_buffer;
			remaining_buffer_size = 64 - (bits_needed - remaining_buffer_size);
		}
	}

	return (__m256i*) buffer;
}

void decompress_9bit_slow(__m256i* input, size_t input_size, std::vector<uint16_t>& output) {
	output.reserve(input_size);

	uint16_t current = 0;

	for (size_t i = 0; i < input_size; i++) 
	{

	}
}

void decompress(__m256i* buffer, int input_size, int* result_buffer)
{
	//To be implemented   
}

int scan(int predicate_low, int predicate_high, __m256i* compressed_input, int input_size)
{
	//To be implemented
	return 0;
}