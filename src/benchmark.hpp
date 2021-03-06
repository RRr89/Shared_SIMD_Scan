#pragma once
#include <string>

const size_t default_data_size = 500 * 1 << 20;
const size_t default_benchmark_repetitions = 5;

void bench_decompression(size_t data_size = default_data_size, size_t benchmark_repetitions = default_benchmark_repetitions);
void bench_scan(size_t data_size = default_data_size, size_t benchmark_repetitions = default_benchmark_repetitions);
void bench_shared_scan(size_t data_size = default_data_size, size_t benchmark_repetitions = default_benchmark_repetitions, 
                       int predicate_key_count = 8, bool relative_data_size = false);

// misc
template<typename T> void bench_memory(size_t data_size = default_data_size);
void bench_memcpy(size_t data_size = default_data_size);
void test_timer();
