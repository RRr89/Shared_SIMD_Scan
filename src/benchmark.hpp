#pragma once
#include <string>

static const size_t data_size = 1.5*(1 << 30);
static const std::string data_size_str = "1.5 GB";

void bench_decompression();
void bench_scan();
void bench_shared_scan();

// misc
template<typename T> void bench_memory();
void test_timer();
