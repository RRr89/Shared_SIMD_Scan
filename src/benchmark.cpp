#include "benchmark.hpp"
#include "simd_scan.hpp"
#include "util.hpp"
#include "profiling.hpp"

#include <chrono>
#include <vector>
#include <iomanip>
#include <sstream>
#include <memory>
#include <functional>
#include <omp.h>

void print_numbers(std::string benchmark_name, const size_t elapsed_time_us[5])
{
    std::cout << benchmark_name
        << ": [" << (elapsed_time_us[0]/1000000)
        << ", "  << (elapsed_time_us[1]/1000000)
        << ", "  << (elapsed_time_us[2]/1000000)
        << ", "  << (elapsed_time_us[3]/1000000)
        << ", "  << (elapsed_time_us[4]/1000000)
        << "] ms" << std::endl;
}

bool check_decompression_result(std::vector<uint16_t> input, int* output, size_t size)
{
    for (size_t i = 0; i < size; i++)
    {
        if (input[i] != output[i])
        {
            std::cout << "first mismatch at index " << i << std::endl;
            return false;
        }
    }
    return true;
}

void do_decompression_benchmark(
    std::string name, 
    std::vector<uint16_t> input,
    size_t input_size, 
    __m128i* compressed_data,
    std::function<void(__m128i*, size_t, int*)> decompression_function) 
{
    size_t elapsed_time_us[5];
    std::unique_ptr<int[]> output_buffer = std::make_unique<int[]>(next_multiple(input_size, 8));

    for (int i = 0; i < 5; ++i)
    {
        _clock();
        decompression_function(compressed_data, input_size, output_buffer.get());
        elapsed_time_us[i] = _clock().count();
    }
    print_numbers(name, elapsed_time_us);
    check_decompression_result(input, output_buffer.get(), input_size);
}

void bench_decompression()
{
    size_t compression = 9;
    size_t buffer_target_size = data_size;
    size_t input_size = buffer_target_size * 8 / compression;

    std::vector<uint16_t> input(input_size);
    for (size_t i = 0; i < input_size; i++)
    {
        input[i] = (uint16_t)(i & ((1 << compression) - 1));
    }

    std::unique_ptr<uint64_t[]> compressed = compress_9bit_input(input);
    __m128i* compressed_ptr = (__m128i*) compressed.get();

    std::cout.imbue(std::locale(""));
    std::cout << "## decompression benchmarks ##" << std::endl;
    std::cout << "compressed input: " << input_size << " (" << buffer_target_size << " bytes)" << std::endl;

    do_decompression_benchmark("unvectorized", input, input_size, compressed_ptr, decompress_unvectorized);
    //do_decompression_benchmark("sse 128 (sweep)", input, input_size, compressed_ptr, decompress_128_sweep);
    //do_decompression_benchmark("sse 128 (load after 4)", input, input_size, compressed_ptr, decompress_128_nosweep);
    //do_decompression_benchmark("sse 128 (9 bit optimized masks)", input, input_size, compressed_ptr, decompress_128_9bit);
    //do_decompression_benchmark("sse 128 (optimized masks)", input, input_size, compressed_ptr, decompress_128);
    do_decompression_benchmark("sse 128 (optimized masks + unrolled loop)", input, input_size, compressed_ptr, decompress_128_unrolled);
    //do_decompression_benchmark("sse 128 (optimized masks + aligned loads)", input, input_size, compressed_ptr, decompress_128_aligned);

#ifdef __AVX__
    do_decompression_benchmark("avx 256", input, input_size, compressed_ptr, decompress_256);
#ifdef __AVX2__
    do_decompression_benchmark("avx 256 (avx2 shift)", input, input_size, compressed_ptr, decompress_256_avx2);
#endif
#else
    std::cout << "avx 256 is not supported" << std::endl;
#endif

    std::cout << "finished benchmark" << std::endl;
}

bool check_scan_result(std::vector<uint16_t> input, size_t size, std::vector<uint8_t> const& output, int predicate_key)
{
    for (size_t i = 0; i < size; i++)
    {
        if (get_bit(output, i) != (input[i] == predicate_key))
        {
            std::cout << "first mismatch at index " << i << std::endl;
            return false;
        }
    }
    return true;
}

void do_scan_benchmark(
    std::string name,
    std::vector<uint16_t> input,
    size_t input_size,
    __m128i* compressed_data,
    std::function<int(int, __m128i*, size_t, std::vector<uint8_t>&)> scan_function)
{
    int predicate_key = 3;
    size_t elapsed_time_us[5];
    std::vector<uint8_t> output_buffer(next_multiple(input_size, 8) / 8);

    for (int i = 0; i < 5; ++i)
    {
        _clock();
        scan_function(predicate_key, compressed_data, input_size, output_buffer);
        elapsed_time_us[i] = _clock().count();
    }
    print_numbers(name, elapsed_time_us);
    check_scan_result(input, input_size, output_buffer, predicate_key);
}

void bench_scan() 
{
    size_t compression = 9;
    size_t buffer_target_size = data_size;
    size_t input_size = buffer_target_size * 8 / compression;

    std::vector<uint16_t> input(input_size);
    for (size_t i = 0; i < input_size; i++)
    {
        input[i] = (uint16_t)(i % 5);
    }

    std::unique_ptr<uint64_t[]> compressed = compress_9bit_input(input);
    __m128i* compressed_ptr = (__m128i*) compressed.get();

    std::cout.imbue(std::locale(""));
    std::cout << "## scan benchmarks ##" << std::endl;
    std::cout << "compressed input: " << input_size << " (" << buffer_target_size << " bytes)" << std::endl;

    do_scan_benchmark("unvectorized", input, input_size, compressed_ptr, scan_unvectorized);
    do_scan_benchmark("sse 128", input, input_size, compressed_ptr, scan_128);

#ifdef __AVX__
    do_scan_benchmark("avx 256", input, input_size, compressed_ptr, scan_256);
#else
    std::cout << "avx 256 is not supported" << std::endl;
#endif

    std::cout << "finished benchmark" << std::endl;
}

void do_shared_scan_benchmark(
    std::string name,
    std::vector<uint16_t> input,
    size_t input_size,
    __m128i* compressed_data,
    std::function<void(std::vector<int> const&, __m128i*, size_t, std::vector<std::vector<uint8_t>>&)> shared_scan_function,
    int predicate_key_count)
{
    std::vector<int> predicate_keys(predicate_key_count);
    for (size_t i = 0; i < predicate_key_count; i++) 
    {
        predicate_keys[i] = i;
    }

    size_t elapsed_time_us[5];

    std::vector<std::vector<uint8_t>> output_buffers(predicate_key_count);
    for (size_t i = 0; i < predicate_key_count; i++) 
    {
        output_buffers.emplace(output_buffers.begin() + i, std::vector<uint8_t>(next_multiple(input_size, 8) / 8));
    }

    for (int i = 0; i < 5; ++i)
    {
        _clock();
        shared_scan_function(predicate_keys, compressed_data, input_size, output_buffers);
        elapsed_time_us[i] = _clock().count();
    }

    print_numbers(name, elapsed_time_us);

    for (size_t i = 0; i < predicate_key_count; i++)
    {
        check_scan_result(input, input_size, output_buffers[i], predicate_keys[i]);
    }
}

void bench_shared_scan()
{
    int predicate_key_count = 8;

    size_t compression = 9;
    size_t buffer_target_size = data_size >> 3;
    size_t input_size = buffer_target_size * 8 / compression;

    std::vector<uint16_t> input(input_size);
    for (size_t i = 0; i < input_size; i++)
    {
        input[i] = (uint16_t)(i % predicate_key_count);
    }

    std::unique_ptr<uint64_t[]> compressed = compress_9bit_input(input);
    __m128i* compressed_ptr = (__m128i*) compressed.get();

    std::cout.imbue(std::locale(""));
    std::cout << "## shared scan benchmarks ##" << std::endl;
    std::cout << "compressed input: " << input_size << " (" << buffer_target_size << " bytes)" << std::endl;
    std::cout << "predicate key count: " << predicate_key_count << std::endl;

    do_shared_scan_benchmark("sse 128, sequential", input, input_size, compressed_ptr, shared_scan_128_sequential, predicate_key_count);
    
    int num_threads = omp_get_max_threads();
    do_shared_scan_benchmark("sse 128, threaded (" + std::to_string(num_threads) + " threads)", input, input_size, compressed_ptr, shared_scan_128_threaded, predicate_key_count);

    do_shared_scan_benchmark("sse 128, horizontal", input, input_size, compressed_ptr, shared_scan_128_horizontal, predicate_key_count);
    do_shared_scan_benchmark("sse 128, vertical", input, input_size, compressed_ptr, shared_scan_128_vertical, predicate_key_count);

#ifdef __AVX__
    do_shared_scan_benchmark("avx 256, horizontal", input, input_size, compressed_ptr, shared_scan_256_horizontal, predicate_key_count);
    do_shared_scan_benchmark("avx 256, vertical", input, input_size, compressed_ptr, shared_scan_256_vertical, predicate_key_count);
#endif

    std::cout << "finished benchmark" << std::endl;
}