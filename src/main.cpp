#include <iostream>
#include <chrono>
#include <thread>
#include <vector>

#include "simd_scan.hpp"
#include "util.hpp"
#include "benchmark.hpp"
#include "profiling.hpp"

int main(int argc, char** argv)
{
    /*bench_memory<uint8_t>();
    bench_memory<uint16_t>();
    bench_memory<uint32_t>();
    bench_memory<uint64_t>();
    bench_memcpy();
    std::cout << "------" << std::endl;*/
    
    /*bench_decompression();
    std::cout << "------" << std::endl;
    
    bench_scan();
    std::cout << "------" << std::endl;*/

    for (int i = 0; i < 5; i++)
        bench_shared_scan(1 << i, false);

    std::cin.get();
    return 0;
}
