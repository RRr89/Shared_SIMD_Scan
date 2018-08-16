#include "benchmark.hpp"
#include "simd_scan.hpp"
#include "util.hpp"

#include <chrono>
#include <vector>
#include <iomanip>
#include <sstream>
#include <memory>
#include <functional>

static const size_t data_size = 500 * 1 << 20;
static const char* const data_size_str = "500 MB";

std::chrono::nanoseconds _clock()
{
    static std::chrono::high_resolution_clock::time_point last;
    auto now = std::chrono::high_resolution_clock::now();
    auto elapsed = now - last;
    last = now;
    return elapsed;
}

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

bool check_result(std::vector<uint16_t> input, int* output, size_t size)
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
    check_result(input, output_buffer.get(), input_size);
    print_numbers(name, elapsed_time_us);
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

    __m128i* compressed = (__m128i*) compress_9bit_input(input);

    std::cout.imbue(std::locale(""));
    std::cout << "## decompression benchmarks ##" << std::endl;
    std::cout << "compressed input: " << input_size << " (" << buffer_target_size << " bytes)" << std::endl;

    // ------------

    do_decompression_benchmark("unvectorized", input, input_size, compressed, decompress_standard);
    do_decompression_benchmark("sse 128 (sweep)", input, input_size, compressed, decompress_128_sweep);
    do_decompression_benchmark("sse 128 (load after 4)", input, input_size, compressed, decompress_128_nosweep);
    do_decompression_benchmark("sse 128 (9 bit optimized masks)", input, input_size, compressed, decompress_128_9bit);
    do_decompression_benchmark("sse 128 (optimized masks)", input, input_size, compressed, decompress_128);
    do_decompression_benchmark("sse 128 (optimized masks + unrolled loop)", input, input_size, compressed, decompress_128_unrolled);
    do_decompression_benchmark("sse 128 (optimized masks + aligned loads)", input, input_size, compressed, decompress_128_aligned);

#ifdef __AVX__
    do_decompression_benchmark("avx 256", input, input_size, compressed, decompress_256);

#ifdef __AVX2__
    do_decompression_benchmark("avx 256 (avx2 shift)", input, input_size, compressed, decompress_256_avx2);
#endif
#else
    std::cout << "avx 256 is not supported" << std::endl;
#endif

    std::cout << "finished benchmark" << std::endl;
}

template<typename T>
void bench_memory()
{
    size_t size = data_size / sizeof(T);
    auto a = std::vector<T>(size);
    for (T& x : a) x = rand() & 0xFF;

    auto b = std::vector<T>(size);

    _clock();

    for (size_t i = 0; i < size; i++)
    {
        b[i] = a[i];
    }

    std::cout.imbue(std::locale(""));
    std::cout << "copy memory (" << sizeof(T) << " byte(s) at a time, " << data_size_str << "): " << (_clock().count()/1000000) << " ms" << std::endl;
}

template void bench_memory<uint8_t>();
template void bench_memory<uint16_t>();
template void bench_memory<uint32_t>();
template void bench_memory<uint64_t>();