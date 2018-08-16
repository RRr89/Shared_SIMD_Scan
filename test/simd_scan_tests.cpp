#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "util.hpp"
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

    __m128i* compressed = (__m128i*)compress_9bit_input(input_numbers);

    SECTION("Unvectorized decompression") 
    {
        auto result_buffer = std::make_unique<int[]>(next_multiple(input_size, 8));
        decompress_unvectorized(compressed, input_numbers.size(), result_buffer.get());

        for (size_t i = 0; i < input_numbers.size(); i++)
        {
            REQUIRE(input_numbers[i] == result_buffer[i]);
        }
    }

    SECTION("SIMD decompression (SSE)") 
    {
        auto result_buffer = std::make_unique<int[]>(next_multiple(input_size, 8));
        decompress_128_sweep((__m128i*)compressed, input_numbers.size(), result_buffer.get());

        for (size_t i = 0; i < input_numbers.size(); i++)
        {
            REQUIRE(input_numbers[i] == result_buffer[i]);
        }
    }
}

TEST_CASE("SIMD Scan", "[simd-scan]")
{
    std::vector<uint16_t> input_numbers{ 1, 2, 3, 3, 2,
        1, 1, 2, 3, 1, 2, 3 };

    __m128i* compressed_data = (__m128i*) compress_9bit_input(input_numbers);

    SECTION("Unvectorized scan") 
    {
        std::vector<bool> output(input_numbers.size());
        int predicate_key = 3;
        int hits = scan_unvectorized(predicate_key, compressed_data, input_numbers.size(), output);

        REQUIRE(hits == 4);

        for (size_t i = 0; i < input_numbers.size(); i++)
        {
            REQUIRE(output[i] == (input_numbers[i] == predicate_key));
        }
    }

    SECTION("SIMD scan (SSE)") 
    {
        std::vector<bool> output(next_multiple(input_numbers.size(), 8));
        int predicate_key = 3;
        int hits = scan_128(predicate_key, compressed_data, input_numbers.size(), output);

        REQUIRE(hits == 4);

        for (size_t i = 0; i < input_numbers.size(); i++)
        {
            REQUIRE(output[i] == (input_numbers[i] == predicate_key));
        }
    }
}

