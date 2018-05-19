#include <iostream>
#include <chrono>
#include <thread>
#include <vector>

#include "simd_scan.hpp"
#include "util.hpp"

int main(int argc, char* argv)
{
	size_t compressed_int_max = (1 << BITS_NEEDED) - 3;
	std::vector<uint16_t> column(compressed_int_max);
	for (size_t i = 0; i < compressed_int_max; i++)
	{
		column[i] = (uint16_t) i;
	}

	__m256i* compressed = compress_9bit_input(column);
	size_t compressed_bytes = ceil((double)column.size() * BITS_NEEDED / 8);

	dump_memory(compressed, compressed_bytes);

	std::vector<uint16_t> decompressed;
	decompress_9bit_slow(compressed, column.size(), decompressed);

    return 0;
}
