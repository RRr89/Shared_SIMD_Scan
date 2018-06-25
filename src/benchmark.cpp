#include <chrono>
#include <vector>
#include <iomanip>
#include <sstream>

#include "benchmark.hpp"
#include "simd_scan.hpp"

std::chrono::nanoseconds _clock() 
{
	static std::chrono::high_resolution_clock::time_point last;
	auto now = std::chrono::high_resolution_clock::now();
	auto elapsed = now - last;
	last = now;
	return elapsed;
}

void bench_decompression() 
{
	size_t compression = 9;
	size_t buffer_target_size = 50 * 1 << 20; // 50 megabytes
	size_t input_size = buffer_target_size * 8 / compression;
	
	std::vector<uint16_t> input(input_size);
	for (size_t i = 0; i < input_size; i++)
	{
		input[i] = (uint16_t)(i & ((1 << compression) - 1));
	}

	__m256i* compressed = compress_9bit_input(input);

	std::cout.imbue(std::locale(""));

	std::cout << "compressed input: " << input_size << " (" << buffer_target_size << " bytes)" << std::endl;

	// ------------

	_clock();
	
	std::vector<uint16_t> decompressed;
	decompress_standard(compressed, input_size, decompressed);
	
	std::cout << "unvectorized: " << _clock().count() << " ns" << std::endl;

	// ------------

	_clock();

	int* decompressed2 = new int[input_size]();
	decompress_128_sweep((__m128i*)compressed, input_size, decompressed2);
	
	std::cout << "sse 128 (sweep): " << _clock().count() << " ns" << std::endl;

	// ------------	

	_clock();

	int* decompressed3 = new int[input_size]();
	decompress_128_nosweep((__m128i*)compressed, input_size, decompressed3);

	std::cout << "sse 128 (load after 4): " << _clock().count() << " ns" << std::endl;

	// ------------	
	
	_clock();

	int* decompressed4 = new int[input_size]();
	decompress_128_9bit((__m128i*)compressed, input_size, decompressed4);

	std::cout << "sse 128 (9bit optimized masks): " << _clock().count() << " ns" << std::endl;

	// ------------	

	_clock();

	int* decompressed7 = new int[input_size]();
	decompress_128((__m128i*)compressed, input_size, decompressed7);

	std::cout << "sse 128 (optimized masks): " << _clock().count() << " ns" << std::endl;

	// ------------

	_clock();

	int* decompressed5 = new int[input_size]();
	decompress_128_aligned((__m128i*)compressed, input_size, decompressed5);

	std::cout << "sse 128 (optimized masks + aligned loads): " << _clock().count() << " ns" << std::endl;

	// ------------

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
}

void bench_memory() 
{
	const int size = 50 * 1 << 20;    //  50 MB

	auto a = std::vector<uint8_t>(size);
	for (uint8_t& x : a) x = rand() & 0xFF;

	auto b = std::vector<uint8_t>(size);

	_clock();
	
	for (size_t i = 0; i < size; i++)
	{
		b[i] = a[i];
	}

	std::cout.imbue(std::locale(""));
	std::cout << "copy memory (50mb): " << _clock().count() << " ns" << std::endl;
}