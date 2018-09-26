#include <iostream>
#include <chrono>
#include <thread>
#include <vector>
#include <cstring>

#include "simd_scan.hpp"
#include "util.hpp"
#include "benchmark.hpp"
#include "profiling.hpp"

void print_cmd_help()
{
    std::cout << "Format: ./shared_simd_scan data_size repetitions bench_name [bench_args...]" << std::endl;
    std::cout << "data_size = _ (for default) | number (in megabytes)" << std::endl;
    std::cout << "repetitions = _ (for default) | number (for number of repetitions" << std::endl;
    std::cout << "bench_name = memory | decompression | scan | sharedscan [predicate_count] " << std::endl;
}

int arg_main(int argc, char** argv)
{
    if (argc < 4)
    {
        print_cmd_help();
        return 1;
    }

    size_t data_size = default_data_size;
    if (strcmp(argv[1], "_") != 0)
    {
        data_size = atoi(argv[1]) * 1 << 20;
    }

    size_t repetitions = default_benchmark_repetitions;
    if (strcmp(argv[2], "_") != 0)
    {
        repetitions = atoi(argv[2]);
    }

    char* bench_name = argv[3];
    if (strcmp(bench_name, "memory") == 0)
    {
        bench_memory<uint8_t>(data_size);
        bench_memory<uint16_t>(data_size);
        bench_memory<uint32_t>(data_size);
        bench_memory<uint64_t>(data_size);
        bench_memcpy();    
    }
    else if (strcmp(bench_name, "decompression") == 0)
    {
        bench_decompression(data_size, repetitions);
    }
    else if (strcmp(bench_name, "scan") == 0)
    {
        bench_scan(data_size, repetitions);
    }
    else if (strcmp(bench_name, "sharedscan") == 0)
    {
        size_t predicate_count = 8;
        if (argc > 4)
        {
            predicate_count = atoi(argv[4]);
        }

        bench_shared_scan(data_size, repetitions, predicate_count);
    }
    else
    {
        print_cmd_help();
        return 1;
    }
    return 0;
}

int main(int argc, char** argv)
{
    if (argc > 1)
    {
        return arg_main(argc, argv);
    }

    print_cmd_help();
    std::cout << "Running all benchmarks with default settings..." << std::endl << "------" << std::endl;

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

    bench_shared_scan(default_data_size >> 3);

    std::cin.get();
    return 0;
}

