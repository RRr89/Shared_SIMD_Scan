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

    auto compressed = compress_9bit_input(input_numbers);
    __m128i* compressed_ptr = (__m128i*) compressed.get();

    SECTION("Unvectorized decompression") 
    {
        auto result_buffer = std::make_unique<int[]>(next_multiple(input_size, 8));
        decompress_unvectorized(compressed_ptr, input_numbers.size(), result_buffer.get());

        for (size_t i = 0; i < input_numbers.size(); i++)
        {
            REQUIRE(input_numbers[i] == result_buffer[i]);
        }
    }

    SECTION("SIMD decompression (SSE)") 
    {
        auto result_buffer = std::make_unique<int[]>(next_multiple(input_size, 8));
        decompress_128_sweep(compressed_ptr, input_numbers.size(), result_buffer.get());

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

    auto compressed = compress_9bit_input(input_numbers);
    __m128i* compressed_ptr = (__m128i*) compressed.get();

    SECTION("Unvectorized scan") 
    {
        std::vector<bool> output(input_numbers.size());
        int predicate_key = 3;
        int hits = scan_unvectorized(predicate_key, compressed_ptr, input_numbers.size(), output);

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
        int hits = scan_128(predicate_key, compressed_ptr, input_numbers.size(), output);

        REQUIRE(hits == 4);

        for (size_t i = 0; i < input_numbers.size(); i++)
        {
            REQUIRE(output[i] == (input_numbers[i] == predicate_key));
        }
    }

    SECTION("SIMD scan v2 (SSE)")
    {
        std::vector<uint8_t> output(next_multiple(input_numbers.size(), 8) / 8);
        int predicate_key = 3;
        int hits = scan_128_v2(predicate_key, compressed_ptr, input_numbers.size(), output);

        REQUIRE(hits == 4);

        for (size_t i = 0; i < input_numbers.size(); i++)
        {
            REQUIRE(get_bit(output, i) == (input_numbers[i] == predicate_key));
        }
    }
}

TEST_CASE("Shared SIMD Scan", "[shared-simd-scan]")
{
    std::vector<uint16_t> input_numbers{ 1, 2, 3, 3, 2,
        1, 1, 2, 3, 1, 2, 3 };

    auto compressed = compress_9bit_input(input_numbers);
    __m128i* compressed_ptr = (__m128i*) compressed.get();

    std::vector<int> predicate_keys{ 1, 2, 3 };

    std::vector<std::vector<bool>> outputs(predicate_keys.size());
    for (size_t i = 0; i < predicate_keys.size(); i++) 
    {
        outputs.emplace(outputs.begin() + i, std::vector<bool>(next_multiple(input_numbers.size(), 8)));
    }

    shared_scan_128_sequential(predicate_keys, compressed_ptr, input_numbers.size(), outputs);

    for (size_t key_id = 0; key_id < predicate_keys.size(); key_id++)
    {
        for (size_t i = 0; i < input_numbers.size(); i++)
        {
            REQUIRE(outputs[key_id][i] == (input_numbers[i] == predicate_keys[key_id]));
        }
    }
}

