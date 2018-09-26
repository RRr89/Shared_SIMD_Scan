#include <iostream>
#include <stdint.h>
#include <iomanip>
#include <math.h>

#include "util.hpp"

static const char nibble2str[16][5] = {
    "0000", "0001", "0010", "0011",
    "0100", "0101", "0110", "0111",
    "1000", "1001", "1010", "1011",
    "1100", "1101", "1110", "1111",
};

void dump_byte(uint8_t byte, std::ostream& os)
{
    os << nibble2str[((byte >> 4) & 0x0F)];
    os << nibble2str[byte & 0x0F];
    os << std::endl;
}

// dumps 64bit blocks, MSB on the left
void dump_memory(const void* mem, size_t bytes, std::ostream& os) 
{
    const uint64_t* p = reinterpret_cast<const uint64_t*>(mem);
    size_t lwords = ceil((double)bytes / 8);

    for (size_t i = 0; i < lwords; i++)
    {
        os << std::setw(4) << i << ": ";

        uint64_t word = p[i];

        // 4 bit steps
        for (size_t j = 0; j < 16; j++) 
        {
            auto nibble = (word >> (60 - 4 * j)) & 0x0F;
            os << nibble2str[nibble] << " ";

            // additional space after each byte
            if ((j + 1) % 2 == 0)
            {
                os << " ";
            }
        }

        os << std::endl;
    }
}

int next_multiple(int number, int multiple)
{
    return ((number + multiple - 1) / multiple) * multiple;
}

bool get_bit(std::vector<uint8_t> const& vector, size_t absolute_index)
{
    size_t vector_index = absolute_index / 8;
    uint8_t element = vector[vector_index];
    size_t bit_index = absolute_index % 8;
    bool bit = (element & (1 << bit_index)) > 0;
    return bit;
}

bool get_bit(std::vector<uint32_t> const& vector, size_t absolute_index)
{
    size_t vector_index = absolute_index / 8;
    uint8_t element = vector[vector_index];
    size_t bit_index = absolute_index % 8;
    bool bit = (element & (1 << bit_index)) > 0;
    return bit;
}

