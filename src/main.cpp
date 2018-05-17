#include <iostream>
#include <chrono>
#include <thread>
#include <vector>

#include "simd_scan.hpp"
#include "util.hpp"

int main(int argc, char* argv)
{
	std::vector<uint16_t> column(1024);
	for (size_t i = 0; i < 1024; i++)
	{
		column[i] = (uint16_t) i;
	}

	__m256i* compressed = compress_9bit_input(column);
	size_t compressed_bytes = ceil((double)column.size() * BITS_NEEDED / 8);

	dump_memory(compressed, compressed_bytes);

	std::this_thread::sleep_for(std::chrono::seconds(5));

    return 0;
}
