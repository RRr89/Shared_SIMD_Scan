#include <iostream>
#include <immintrin.h>
#include <cmath>
#include <vector>
#include <bitset>
#include <algorithm>

#include "simd_scan.hpp"

__m256i* compress_9bit_input(std::vector<uint16_t>& input)
{
	auto bits_needed = BITS_NEEDED;
	auto mem_size = bits_needed * input.size();
	int array_size = ceil((double)mem_size / 64);
	auto buffer = new long long[array_size](); // TODO allocated on heap!

	int remaining_buffer_size = 64;
	int idx_ = 0;
	for (size_t i = 0; i < input.size(); i++)
	{
		long long tmp_buffer = 0;
		tmp_buffer = tmp_buffer | input[i];
		tmp_buffer = tmp_buffer << (i * bits_needed); // TODO undefined behaviour?
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

#include "util.hpp"
void decompress_128(__m128i* input, size_t input_size, int* output)
{
	size_t compression = BITS_NEEDED;
	size_t free_bits = 32 - compression; // most significant bits in result values that must be 0

	__m128i source = _mm_loadu_si128(input);
	int unread_bits = 128;

	size_t output_index = 0; // current write index of the output array (equals # of decompressed values)
	size_t total_processed_bytes = 0; // holds # of input bytes that have been processed completely

	while (output_index < input_size)
	{
		// TODO can this be optimized? is this constant?
		size_t input_offset[4] = {
			(compression * (output_index + 0) / 8) - total_processed_bytes,
			(compression * (output_index + 1) / 8) - total_processed_bytes,
			(compression * (output_index + 2) / 8) - total_processed_bytes,
			(compression * (output_index + 3) / 8) - total_processed_bytes
		};

		__m128i shuffle_mask = _mm_setr_epi8(
			input_offset[0], input_offset[0] + 1, input_offset[0] + 2, input_offset[0] + 3,
			input_offset[1], input_offset[1] + 1, input_offset[1] + 2, input_offset[1] + 3,
			input_offset[2], input_offset[2] + 1, input_offset[2] + 2, input_offset[2] + 3,
			input_offset[3], input_offset[3] + 1, input_offset[3] + 2, input_offset[3] + 3);
		__m128i b = _mm_shuffle_epi8(source, shuffle_mask);

		// shift left by variable amounts (using integer multiplication)
		// TODO can this be optimized?
		size_t padding[4] = {
			compression * (output_index + 0) % 8,
			compression * (output_index + 1) % 8,
			compression * (output_index + 2) % 8,
			compression * (output_index + 3) % 8
		};

		__m128i mult = _mm_setr_epi32(
			1 << (free_bits - padding[0]),
			1 << (free_bits - padding[1]),
			1 << (free_bits - padding[2]),
			1 << (free_bits - padding[3]));

		__m128i c = _mm_mullo_epi32(b, mult);

		// shift right by fixed amount
		__m128i d = _mm_srli_epi32(c, 32 - compression);

		// and masking (TODO really needed?)
		int mask = (1 << compression) - 1;
		__m128i and_mask = _mm_set1_epi32(mask);
		__m128i e = _mm_and_si128(d, and_mask);

		// TODO handle cases where output size in not multiple of 4
		_mm_storeu_si128((__m128i*)&output[output_index], e);

		output_index += 4;
		unread_bits -= 4 * compression;

		// load next 
		if (unread_bits < 4 * compression) 
		{
			// TODO uses unaligned loads --> possibly slow?
			total_processed_bytes = output_index * compression / 8;
			source = _mm_loadu_si128((__m128i*)&((uint8_t*)input)[total_processed_bytes]);
			unread_bits = 128;
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