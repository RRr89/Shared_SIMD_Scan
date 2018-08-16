#include <iostream>
#include <chrono>
#include <thread>
#include <vector>

#include "simd_scan.hpp"
#include "util.hpp"
#include "benchmark.hpp"

int main(int argc, char** argv)
{
    bench_memory();
    std::cout << "------" << std::endl;
    bench_decompression();

    std::cin.get();
    return 0;
}
