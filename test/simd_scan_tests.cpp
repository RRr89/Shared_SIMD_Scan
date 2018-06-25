#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "simd_scan.hpp"

TEST_CASE("Compress and decompress", "[simd-decompress]")
{
	// choose a number slightly smaller than the max possible to avoid 
	// lining up to 64 bit boundaries
	size_t input_size = (1 << BITS_NEEDED) - 3;
	std::vector<uint16_t> input_numbers(input_size);
	for (size_t i = 0; i < input_size; i++)
	{
		input_numbers[i] = (uint16_t) i;
	}

	__m256i* compressed = compress_9bit_input(input_numbers);

	SECTION("Non-vectorized decompression") 
	{
		std::vector<uint16_t> decompressed;
		decompress_standard(compressed, input_numbers.size(), decompressed);

		REQUIRE(decompressed.size() == input_numbers.size());

		for (size_t i = 0; i < input_numbers.size(); i++)
		{
			REQUIRE(input_numbers[i] == decompressed[i]);
		}
	}

	SECTION("SIMD decompression (SSE)") 
	{
		// TODO remove reserve once compression has been updated...
		int *result_buffer = new int[input_numbers.size() + 4]();
		decompress_128_sweep((__m128i*)compressed, input_numbers.size(), result_buffer);

		for (size_t i = 0; i < input_numbers.size(); i++)
		{
			REQUIRE(input_numbers[i] == result_buffer[i]);
		}

		delete result_buffer;
	}
}

TEST_CASE("SIMD Scan", "[simd-scan]")
{
	std::vector<uint16_t> input_numbers{ 1, 2, 3, 4, 5,
		6, 7, 8, 9, 10, 11, 12 };

	__m256i* compressed_data = compress_9bit_input(input_numbers);
	int qualified_tuples = scan(3, 8, compressed_data, input_numbers.size());
	REQUIRE(qualified_tuples == 6);
}