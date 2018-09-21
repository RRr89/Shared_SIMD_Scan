#pragma once
#include <iostream>
#include <vector>

void dump_byte(uint8_t byte, std::ostream& os = std::cout);

void dump_memory(const void* mem, size_t size, std::ostream& os = std::cout);

int next_multiple(int number, int multiple);

bool get_bit(std::vector<uint8_t> const& vector, size_t absolute_index);

#if defined(_MSC_VER)
    #include <intrin.h>
    #define POPCNT(i) __popcnt(i)
#elif defined(__GNUC__)
    #define POPCNT(i) __builtin_popcount(i)
#else 
    #define POPCNT(i) (0)
#endif