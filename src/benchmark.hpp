#pragma once
#include <string>

static const size_t data_size = 500 * 1 << 20;
static const std::string data_size_str = "500 MB";

void bench_decompression();
void bench_scan();
void bench_shared_scan();

// misc
template<typename T> void bench_memory();
void test_timer();
