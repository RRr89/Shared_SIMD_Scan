#include <iostream>
#include <chrono>
#include <thread>
#include <vector>

#include "simd_scan.hpp"
#include "util.hpp"
#include "benchmark.hpp"

int main(int argc, char* argv)
{
	measure_decompression();

	std::cin.get();
	return 0;
}
