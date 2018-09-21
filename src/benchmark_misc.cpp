#include <iostream>
#include <chrono>
#include <vector>
#include <cstring>

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

    auto time = _clock();

    std::cout << "copy memory (" << sizeof(T) << " byte(s) at a time, " << data_size_str << "): " << (time.count() / 1000000) << " ms";
    std::cout << ", throughput: " << (2 * (double)(data_size >> 20) / 1000 / ((double)time.count() / 1000000000)) << " GB/s" << std::endl;
}

template void bench_memory<uint8_t>();
template void bench_memory<uint16_t>();
template void bench_memory<uint32_t>();
template void bench_memory<uint64_t>();

void bench_memcpy()
{
    size_t size = data_size;
    auto a = std::vector<uint8_t>(size);
    for (uint8_t& x : a) x = rand() & 0xFF;

    auto b = std::vector<uint8_t>(size);

    _clock();

    memcpy(b.data(), a.data(), size);

    auto time = _clock();

    std::cout << "copy memory (memcpy, " << data_size_str << "): " << (time.count() / 1000000) << " ms";
    std::cout << ", throughput: " << (2 * (double)(data_size >> 20) / 1000 / ((double)time.count() / 1000000000)) << " GB/s" << std::endl;
}

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

