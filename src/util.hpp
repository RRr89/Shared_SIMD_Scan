#pragma once
#include <iostream>

void dump_memory(const void* mem, size_t size, std::ostream &os = std::cout);

int next_multiple(int number, int multiple);