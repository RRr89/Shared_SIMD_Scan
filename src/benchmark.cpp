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

void print_numbers(std::string benchmark_name, std::vector<size_t> const& elapsed_time_us)
{
    size_t benchmark_repetitions = elapsed_time_us.size();
    std::cout << "* " << benchmark_name << ": ";

    size_t sum = 0;
    for (int i = 0; i < benchmark_repetitions; i++)
    {
        sum += elapsed_time_us[i];
    }

    size_t avg = sum / benchmark_repetitions;

    std::cout << (avg / 1000000) << " ms; [";

    for (int i = 0; i < benchmark_repetitions; i++)
    {
        if (i != 0) std::cout << ", ";
        std::cout << (elapsed_time_us[i] / 1000000);
    }

    std::cout << "] ms" << std::endl;
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
    size_t benchmark_repetitions,
    std::vector<uint16_t> input,
    size_t input_size, 
    __m128i* compressed_data,
    std::function<void(__m128i*, size_t, int*)> decompression_function) 
{
    std::vector<size_t> elapsed_time_us(benchmark_repetitions);
    size_t output_buffer_size = decompression_output_buffer_size(input_size) / sizeof(int);
    std::unique_ptr<int[]> output_buffer = std::make_unique<int[]>(output_buffer_size);

    for (int i = 0; i < benchmark_repetitions; ++i)
    {
        _clock();
        decompression_function(compressed_data, input_size, output_buffer.get());
        elapsed_time_us[i] = _clock().count();
    }
    print_numbers(name, elapsed_time_us);
    check_decompression_result(input, output_buffer.get(), input_size);
}

void bench_decompression(size_t data_size, size_t repetitions)
{
    size_t compression = 9;
    size_t input_size = data_size * 8 / compression;

    std::vector<uint16_t> input(input_size);
    for (size_t i = 0; i < input_size; i++)
    {
        input[i] = (uint16_t)(i & ((1 << compression) - 1));
    }

    std::unique_ptr<uint64_t[]> compressed = compress_9bit_input(input);
    __m128i* compressed_ptr = (__m128i*) compressed.get();

    std::cout << "## decompression benchmarks ##" << std::endl;
    std::cout << "compressed input: " << input_size << " (" << data_size << " bytes)" << std::endl;

    do_decompression_benchmark("unvectorized", repetitions, input, input_size, compressed_ptr, decompress_unvectorized);
    do_decompression_benchmark("sse 128 (sweep)", repetitions, input, input_size, compressed_ptr, decompress_128_sweep);
    do_decompression_benchmark("sse 128 (load after 4)", repetitions, input, input_size, compressed_ptr, decompress_128_nosweep);
    do_decompression_benchmark("sse 128 (9 bit optimized masks)", repetitions, input, input_size, compressed_ptr, decompress_128_9bit);
    do_decompression_benchmark("sse 128 (optimized masks)", repetitions, input, input_size, compressed_ptr, decompress_128);
    do_decompression_benchmark("sse 128 (optimized masks + unrolled loop)", repetitions, input, input_size, compressed_ptr, decompress_128_unrolled);
    do_decompression_benchmark("sse 128 (optimized masks + aligned loads)", repetitions, input, input_size, compressed_ptr, decompress_128_aligned);

#ifdef __AVX__
    do_decompression_benchmark("avx 256", repetitions, input, input_size, compressed_ptr, decompress_256);
#ifdef __AVX2__
    do_decompression_benchmark("avx 256 (avx2 shift)", repetitions, input, input_size, compressed_ptr, decompress_256_avx2);
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
    size_t benchmark_repetitions,
    std::vector<uint16_t> input,
    size_t input_size,
    __m128i* compressed_data,
    std::function<int(int, __m128i*, size_t, std::vector<uint8_t>&)> scan_function)
{
    int predicate_key = 3;
    std::vector<size_t> elapsed_time_us(benchmark_repetitions);
    auto output_buffer_size = scan_output_buffer_size(input_size);
    std::vector<uint8_t> output_buffer(output_buffer_size);

    for (int i = 0; i < benchmark_repetitions; ++i)
    {
        _clock();
        scan_function(predicate_key, compressed_data, input_size, output_buffer);
        elapsed_time_us[i] = _clock().count();
    }
    print_numbers(name, elapsed_time_us);
    check_scan_result(input, input_size, output_buffer, predicate_key);
}

void bench_scan(size_t data_size, size_t repetitions) 
{
    size_t compression = 9;
    size_t input_size = data_size * 8 / compression;

    std::vector<uint16_t> input(input_size);
    for (size_t i = 0; i < input_size; i++)
    {
        input[i] = (uint16_t)(i % 5);
    }

    std::unique_ptr<uint64_t[]> compressed = compress_9bit_input(input);
    __m128i* compressed_ptr = (__m128i*) compressed.get();

    std::cout << "## scan benchmarks ##" << std::endl;
    std::cout << "compressed input: " << input_size << " (" << data_size << " bytes)" << std::endl;

    do_scan_benchmark("unvectorized", repetitions, input, input_size, compressed_ptr, scan_unvectorized);
    do_scan_benchmark("sse 128", repetitions, input, input_size, compressed_ptr, scan_128);
    do_scan_benchmark("sse 128 (unrolled)", repetitions, input, input_size, compressed_ptr, scan_128_unrolled);

#ifdef __AVX__
    do_scan_benchmark("avx 256", repetitions, input, input_size, compressed_ptr, scan_256);
    do_scan_benchmark("avx 256 (unrolled)", repetitions, input, input_size, compressed_ptr, scan_256_unrolled);
#else
    std::cout << "avx 256 is not supported" << std::endl;
#endif

    std::cout << "finished benchmark" << std::endl;
}

void do_shared_scan_benchmark(
    std::string name,
    size_t benchmark_repetitions,
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

    std::vector<size_t> elapsed_time_us(benchmark_repetitions);

    size_t output_buffer_size = scan_output_buffer_size(input_size);
    std::vector<std::vector<uint8_t>> output_buffers(predicate_key_count, std::vector<uint8_t>(output_buffer_size));

    for (int i = 0; i < benchmark_repetitions; ++i)
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

void bench_shared_scan(size_t data_size, size_t repetitions, int predicate_key_count, bool relative_data_size)
{
    size_t compression = 9;

    if (relative_data_size)
    {
        data_size / predicate_key_count;
    }
    
    size_t input_size = data_size * 8 / compression;

    std::vector<uint16_t> input(input_size);
    for (size_t i = 0; i < input_size; i++)
    {
        input[i] = (uint16_t)(i % predicate_key_count % (1 << compression));
    }

    std::unique_ptr<uint64_t[]> compressed = compress_9bit_input(input);
    __m128i* compressed_ptr = (__m128i*) compressed.get();

    std::cout << "## shared scan benchmarks ##" << std::endl;
    std::cout << "compressed input: " << input_size << " (" << data_size << " bytes)" << std::endl;
    std::cout << "predicate key count: " << predicate_key_count << std::endl;

    do_shared_scan_benchmark("sse 128, sequential", repetitions, input, input_size, compressed_ptr, shared_scan_128_sequential, predicate_key_count);
    do_shared_scan_benchmark("sse 128, sequential (unrolled)", repetitions, input, input_size, compressed_ptr, shared_scan_128_sequential_unrolled, predicate_key_count);
    
    int num_threads = omp_get_max_threads();
    do_shared_scan_benchmark("sse 128, threaded (" + std::to_string(num_threads) + " threads)", repetitions, input, input_size, compressed_ptr, shared_scan_128_threaded, predicate_key_count);

    do_shared_scan_benchmark("sse 128, standard", repetitions, input, input_size, compressed_ptr, shared_scan_128_standard, predicate_key_count);
    do_shared_scan_benchmark("sse 128, standard (unrolled)", repetitions, input, input_size, compressed_ptr, shared_scan_128_standard_unrolled, predicate_key_count);
    do_shared_scan_benchmark("sse 128, parallel", repetitions, input, input_size, compressed_ptr, shared_scan_128_parallel, predicate_key_count);

#ifdef __AVX__
    do_shared_scan_benchmark("avx 256, sequential", repetitions, input, input_size, compressed_ptr, shared_scan_256_sequential, predicate_key_count);
    do_shared_scan_benchmark("avx 256, standard", repetitions, input, input_size, compressed_ptr, shared_scan_256_standard, predicate_key_count);
    do_shared_scan_benchmark("avx 256, parallel", repetitions, input, input_size, compressed_ptr, shared_scan_256_parallel, predicate_key_count);
#endif

    std::cout << "finished benchmark" << std::endl;
}