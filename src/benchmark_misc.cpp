#include <chrono>
#include <vector>
#include <iostream>
#include <intrin.h>

#include "benchmark.hpp"
#include "profiling.hpp"

template<typename T>
void bench_memory()
{
    size_t size = data_size / sizeof(T);
    auto a = std::vector<T>(size);
    for (T& x : a) x = rand() & 0xFF;

    auto b = std::vector<T>(size);

    _clock();

    for (size_t i = 0; i < size; i++)
    {
        b[i] = a[i];
    }

    std::cout.imbue(std::locale(""));
    std::cout << "copy memory (" << sizeof(T) << " byte(s) at a time, " << data_size_str << "): " << (_clock().count() / 1000000) << " ms" << std::endl;
}

template void bench_memory<uint8_t>();
template void bench_memory<uint16_t>();
template void bench_memory<uint32_t>();
template void bench_memory<uint64_t>();

void test_timer()
{
    std::vector<std::chrono::nanoseconds> diffs(1000);
    auto last = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < 1000; i++)
    {
        auto now = std::chrono::high_resolution_clock::now();
        auto diff = now - last;
        last = now;
        diffs[i] = diff;
    }

    for (auto& n : diffs) 
    {
        std::cout << n.count() << std::endl;
    }
}

void test_tsc()
{
    std::vector<uint64_t> diffs(1000);
    uint64_t last;
    uint32_t high, low;

    for (size_t i = 0; i < 1000; i++)
    {
        //__asm__ __volatile__("rdtsc" : "=a" (low), "=d" (high));
        uint64_t cycles = __rdtsc();

        diffs[i] = cycles - last;
        last = cycles;
    }

    for (auto& d : diffs) 
    {
        std::cout << d << std::endl;
    }
}