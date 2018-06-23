#include "simd_scan.hpp"
#include "util.hpp"

void decompress_standard(__m256i* input, size_t input_size, std::vector<uint16_t>& output)
{
	output.resize(input_size);

	uint64_t* in = reinterpret_cast<uint64_t*>(input);
	auto bits_needed = BITS_NEEDED;
	auto mem_size = bits_needed * input_size;
	int array_size = ceil((double)mem_size / 64);

	uint64_t mask = (1 << bits_needed) - 1;

	uint64_t current = 0;
	size_t overflow_bits = 0;

	size_t oi = 0;

	for (size_t i = 0; i < array_size; i++)
	{
		current = in[i] >> overflow_bits;

		size_t unread_bits = 64 - overflow_bits;
		while (unread_bits >= bits_needed)
		{
			uint16_t decompressed_element = current & mask;
			output[oi++] = decompressed_element;

			if (oi == input_size)
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
			output[oi++] = decompressed_element;

			overflow_bits = bits_needed - unread_bits;
		}
		else {
			overflow_bits = 0;
		}
	}
}

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

		//std::cout << "input_offset: " << input_offset[0] << " " << input_offset[1] << " " << input_offset[2] << " " << input_offset[3] << std::endl;

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

		// TODO handle cases where output size in not multiple of 4
		_mm_storeu_si128((__m128i*)&output[output_index], d);

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

void decompress_128_1group(__m128i* input, size_t input_size, int* output)
{
	size_t compression = BITS_NEEDED;
	size_t free_bits = 32 - compression; // most significant bits in result values that must be 0

	__m128i source = _mm_loadu_si128(input);

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

		//std::cout << "input_offset: " << input_offset[0] << " " << input_offset[1] << " " << input_offset[2] << " " << input_offset[3] << std::endl;

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

		// TODO handle cases where output size in not multiple of 4
		_mm_storeu_si128((__m128i*)&output[output_index], d);

		output_index += 4;

		// load next immediately
		// TODO uses unaligned loads --> possibly slow?
		total_processed_bytes = output_index * compression / 8;
		source = _mm_loadu_si128((__m128i*)&((uint8_t*)input)[total_processed_bytes]);
	}
}

void decompress_128_9bit_1group(__m128i* input, size_t input_size, int* output)
{
	size_t compression = 9;
	size_t free_bits = 32 - compression; // most significant bits in result values that must be 0

	__m128i source = _mm_loadu_si128(input);

	size_t output_index = 0; // current write index of the output array (equals # of decompressed values)
	size_t total_processed_bytes = 0; // holds # of input bytes that have been processed completely

	// shuffle mask, pulled outside the loop (compression 9 has one constant shuffle mask)
	size_t input_offset[4] = { 0, 1, 2, 3 };

	__m128i shuffle_mask = _mm_setr_epi8(
		input_offset[0], input_offset[0] + 1, input_offset[0] + 2, input_offset[0] + 3,
		input_offset[1], input_offset[1] + 1, input_offset[1] + 2, input_offset[1] + 3,
		input_offset[2], input_offset[2] + 1, input_offset[2] + 2, input_offset[2] + 3,
		input_offset[3], input_offset[3] + 1, input_offset[3] + 2, input_offset[3] + 3);

	// shift masks (compression 9 has two shift masks used in alternation)
	__m128i shift_mask_1 = _mm_setr_epi32(
		1 << (free_bits - 0),
		1 << (free_bits - 1),
		1 << (free_bits - 2),
		1 << (free_bits - 3));

	__m128i shift_mask_2 = _mm_setr_epi32(
		1 << (free_bits - 4),
		1 << (free_bits - 5),
		1 << (free_bits - 6),
		1 << (free_bits - 7));

	while (output_index < input_size)
	{
		__m128i b = _mm_shuffle_epi8(source, shuffle_mask);

		__m128i c;
		if (output_index % 8 == 0)
		{
			c = _mm_mullo_epi32(b, shift_mask_1);
		}
		else 
		{
			c = _mm_mullo_epi32(b, shift_mask_2);
		}

		// shift right by fixed amount
		__m128i d = _mm_srli_epi32(c, 32 - compression);

		// TODO handle cases where output size in not multiple of 4
		_mm_storeu_si128((__m128i*)&output[output_index], d);

		output_index += 4;

		// load next
		total_processed_bytes = output_index * compression / 8;
		source = _mm_loadu_si128((__m128i*)&((uint8_t*)input)[total_processed_bytes]);
	}
}
		total_processed_bytes = output_index * compression / 8;
		source = _mm_loadu_si128((__m128i*)&((uint8_t*)input)[total_processed_bytes]);
	}
}