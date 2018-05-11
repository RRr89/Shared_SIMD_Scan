#define CATCH_CONFIG_MAIN // This tells Catch to provide a main() - only do this in one cpp file
#include "../catch/catch.hpp"
#include "../src/SIMD_Scan.cpp"

TEST_CASE("TEST DECOMPRESS", "[simd-decompress]")
{
    
    std::vector<uint16_t> input_numbers{1, 2, 3, 4, 5,
                                        6, 7, 8, 9, 10,
                                        11, 12};
    
    __m256i* compressed_data = compress_9bit_input(input_numbers);
    int *result_buffer = new int[input_numbers.size()]();
    decompress(compressed_data,input_numbers.size(), result_buffer);

    for (size_t i = 0; i < input_numbers.size(); i++)
    {
        REQUIRE(input_numbers[i] == result_buffer[i]);
    }
}

TEST_CASE("TEST SCAN", "[simd-scan]")
{


    std::vector<uint16_t> input_numbers{1, 2, 3, 4, 5,
                                        6, 7, 8, 9, 10,
                                        11, 12};

    __m256i* compressed_data = compress_9bit_input(input_numbers);
    int qualified_tuples = scan(3,8,compressed_data,input_numbers.size());
    REQUIRE(qualified_tuples == 6);
}