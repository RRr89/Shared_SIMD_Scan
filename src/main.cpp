#include <iostream>
#include <chrono>
#include <thread>
#include <vector>
#include <cstring>

#include "simd_scan.hpp"
#include "util.hpp"
#include "benchmark.hpp"
#include "profiling.hpp"

int arg_main(int argc, char** argv)
{
    char* bench_name = argv[1];
    if (strcmp(bench_name, "memory") == 0)
    {
        bench_memory<uint8_t>();
        bench_memory<uint16_t>();
        bench_memory<uint32_t>();
        bench_memory<uint64_t>();
        bench_memcpy();    
    }
    else if (strcmp(bench_name, "decompression") == 0)
    {
        bench_decompression();
    }
    else if (strcmp(bench_name, "scan") == 0)
    {
        bench_scan();
    }
    else if (strcmp(bench_name, "sharedscan") == 0)
    {
        size_t predicate_count = 8;
        if (argc >= 3)
        {
            predicate_count = atoi(argv[2]);
        }

        bench_shared_scan(predicate_count);
    }
    else
    {
        std::cout << "Format: ./shared_simd_scan [bench_name] [bench_args...]" << std::endl;
        std::cout << "bench_name = memory | decompression | scan | sharedscan [predicate_count] " << std::endl;
    }
    return 0;
}

int main(int argc, char** argv)
{
    if (argc >= 2)
    {
        return arg_main(argc, argv);
    }

    bench_memory<uint8_t>();
    bench_memory<uint16_t>();
    bench_memory<uint32_t>();
    bench_memory<uint64_t>();
    bench_memcpy();    
    std::cout << "------" << std::endl;
    
    bench_decompression();
    std::cout << "------" << std::endl;
    
    bench_scan();
    std::cout << "------" << std::endl;

    bench_shared_scan();

    std::cin.get();
    return 0;
}
