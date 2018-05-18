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
		tmp_buffer = tmp_buffer << (i * bits_needed); // undefined behaviour?
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
			// logic to handle overflow_bits
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

void decompress_9bit_slow(__m256i* input, size_t input_size, std::vector<uint16_t>& output) 
{
	output.reserve(input_size);

	uint64_t* in = reinterpret_cast<uint64_t*>(input);
	auto bits_needed = BITS_NEEDED;
	auto mem_size = bits_needed * input_size;
	int array_size = ceil((double)mem_size / 64);

	uint64_t mask = (1 << bits_needed) - 1;

	uint64_t current = 0;
	size_t overflow_bits = 0;

	for (size_t i = 0; i < array_size; i++) 
	{
		current = in[i] >> overflow_bits;

		size_t unread_bits = 64 - overflow_bits;
		while (unread_bits >= bits_needed) 
		{
			uint16_t decompressed_element = current & mask;
			output.push_back(decompressed_element);

			if (output.size() == input_size) 
			{
				return;
			}

			current = current >> bits_needed;
			unread_bits -= bits_needed;
		}

		// handle overlapping element
		if (unread_bits != 0)
		{
			uint64_t next = in[i + 1];
			current = current | (next << unread_bits);

			uint16_t decompressed_element = current & mask;
			output.push_back(decompressed_element);

			overflow_bits = bits_needed - unread_bits;
		}
		else {
			overflow_bits = 0;
		}
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