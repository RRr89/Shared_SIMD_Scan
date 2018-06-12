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

void measure_decompression() 
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
	decompress_9bit_slow(compressed, input_size, decompressed);
	
	std::cout << "slow: " << _clock().count() << " ns" << std::endl;

	// ------------

	_clock();

	int* decompressed2 = new int[input_size]();
	decompress_128((__m128i*)compressed, input_size, decompressed2);
	
	std::cout << "sse 128: " << _clock().count() << " ns" << std::endl;

	// ------------
}