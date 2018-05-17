#include <iostream>
#include <stdint.h>

#include "util.hpp"

static const char byte2str[16][5] = {
	"0000", "0001", "0010", "0011",
	"0100", "0101", "0110", "0111",
	"1000", "1001", "1010", "1011",
	"1100", "1101", "1110", "1111",
};

void dump_memory(const void* mem, size_t size, std::ostream& os) {
	const uint8_t* p = reinterpret_cast<const uint8_t*>(mem);
	for (size_t i = 0; i < size; i++) 
	{
		os << byte2str[p[i] >> 4] << " " << byte2str[p[i] & 0x0F] << " ";
		if ((i+1) % 8 == 0)
		{
			os << "\n";
		}
	}
}