
#include "benchmark.hpp"
#include "simd_scan.hpp"

#include <chrono>
#include <vector>
#include <iomanip>
#include <sstream>

static const size_t data_size = 500 * 1<<20; // 500MB
static const char* const data_size_str = "500 MB";

std::chrono::nanoseconds _clock() 
{
	static std::chrono::high_resolution_clock::time_point last;
	auto now = std::chrono::high_resolution_clock::now();
	auto elapsed = now - last;
	last = now;
	return elapsed;
}

void print_numbers(const char* benchmark, const size_t elapsed_time_us[5])
{
    std::cout << benchmark << ": [" << (elapsed_time_us[0]/1000000)
                           << ", "  << (elapsed_time_us[1]/1000000)
                           << ", "  << (elapsed_time_us[1]/1000000)
                           << ", "  << (elapsed_time_us[1]/1000000)
                           << ", "  << (elapsed_time_us[1]/1000000)
                           << "] ms" << std::endl;
}

void bench_decompression() 
{
	size_t compression = 9;
	size_t buffer_target_size = data_size;
	size_t input_size = buffer_target_size * 8 / compression;
        size_t elapsed_time_us[5];
	
	std::vector<uint16_t> input(input_size);
	for (size_t i = 0; i < input_size; i++)
	{
		input[i] = (uint16_t)(i & ((1 << compression) - 1));
	}

	avxiptr_t compressed = compress_9bit_input(input);

	std::cout.imbue(std::locale(""));

	std::cout << "compressed input: " << input_size << " (" << buffer_target_size << " bytes)" << std::endl;

	// ------------

        for (int i=0; i<5; ++i)
        {
            _clock();
	
            std::vector<uint16_t> decompressed;
            decompress_standard(compressed, input_size, decompressed);
            elapsed_time_us[i] = _clock().count();
        }
        print_numbers("unvectorized", elapsed_time_us);

	// ------------

        for (int i=0; i<5; ++i)
        {
            _clock();

            int* decompressed2 = new int[input_size]();
            decompress_128_sweep((__m128i*)compressed, input_size, decompressed2);
            elapsed_time_us[i] = _clock().count();
        }
        print_numbers("sse 128 (sweep)", elapsed_time_us);

	// ------------	

        for (int i=0; i<5; ++i)
        {
            _clock();

            int* decompressed3 = new int[input_size]();
            decompress_128_nosweep((__m128i*)compressed, input_size, decompressed3);
            elapsed_time_us[i] = _clock().count();
        }
        print_numbers("sse 128 (load after 4)", elapsed_time_us);

	// ------------	
	
        for (int i=0; i<5; ++i)
        {
            _clock();

            int* decompressed4 = new int[input_size]();
            decompress_128_9bit((__m128i*)compressed, input_size, decompressed4);
            elapsed_time_us[i] = _clock().count();
        }
        print_numbers("sse 128 (9bit optimized masks)", elapsed_time_us);

	// ------------	

        for (int i=0; i<5; ++i)
        {
            _clock();

            int* decompressed7 = new int[input_size]();
            decompress_128((__m128i*)compressed, input_size, decompressed7);
            elapsed_time_us[i] = _clock().count();
        }
        print_numbers("sse 128 (optimized masks)", elapsed_time_us);

	// ------------

        for (int i=0; i<5; ++i)
        {
            _clock();

            int* decompressed5 = new int[input_size]();
            decompress_128_aligned((__m128i*)compressed, input_size, decompressed5);
            elapsed_time_us[i] = _clock().count();
        }
        print_numbers("sse 128 (optimized masks + aligned loads)", elapsed_time_us);

	// ------------
#ifdef COMPILER_SUPPORTS_AVX512

	_clock();

	int* decompressed6 = new int[input_size]();
	decompress_256((__m128i*)compressed, input_size, decompressed6);

	std::cout << "avx 256: " << _clock().count() << " ns" << std::endl;

	// ------------

	_clock();

	int* decompressed8 = new int[input_size]();
	decompress_256_avx2((__m128i*)compressed, input_size, decompressed8);

	std::cout << "avx 256 (avx2 shift): " << _clock().count() << " ns" << std::endl;

	// ------------

	// checking results...
	for (size_t i = 0; i < input_size; i++) 
	{
		if (!(input[i] == decompressed[i] && input[i] == decompressed2[i] 
			&& input[i] == decompressed3[i] && input[i] == decompressed4[i]
			&& input[i] == decompressed5[i] && input[i] == decompressed6[i]
			&& input[i] == decompressed7[i] && input[i] == decompressed8[i]))
		{
			std::cout << "mismatch at index " << i << std::endl;
		}
	}
	std::cout << "finished checking results" << std::endl;
#else
        std::cout << "avx 256 is not supported" << std::endl;
#endif
}

void bench_memory() 
{
	const int size = data_size;

	auto a = std::vector<uint8_t>(size);
	for (uint8_t& x : a) x = rand() & 0xFF;

	auto b = std::vector<uint8_t>(size);

	_clock();
	
	for (size_t i = 0; i < size; i++)
	{
		b[i] = a[i];
	}

	std::cout.imbue(std::locale(""));
	std::cout << "copy memory (" << data_size_str << "): " << (_clock().count()/1000000) << " ms" << std::endl;
}

