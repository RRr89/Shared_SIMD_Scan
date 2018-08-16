#include <iostream>
#include <immintrin.h>
#include <cmath>
#include <vector>
#include <bitset>
#include <algorithm>

#include "simd_scan.hpp"

int scan_unvectorized(int predicate_key, __m128i* input, size_t input_size, std::vector<bool>& output)
{
    uint64_t* in = reinterpret_cast<uint64_t*>(input);
    auto bits_needed = BITS_NEEDED;
    auto mem_size = bits_needed * input_size;
    int array_size = ceil((double)mem_size / 64);

    uint64_t mask = (1 << bits_needed) - 1;

    uint64_t current = 0;
    size_t overflow_bits = 0;

    size_t oi = 0;

    int hits = 0;

    for (size_t i = 0; i < array_size; i++)
    {
        current = in[i] >> overflow_bits;

        size_t unread_bits = 64 - overflow_bits;
        while (unread_bits >= bits_needed)
        {
            uint16_t decompressed_element = current & mask;
            bool match = decompressed_element == predicate_key;
            output[oi++] = match;
            if (match) hits++;

            if (oi == input_size)
            {
                return hits;
            }

            current = current >> bits_needed;
            unread_bits -= bits_needed;
        }

        // handle overlapping element
        if (unread_bits != 0)
        {
            uint64_t next = in[i + 1];
            current = current | (next << unread_bits);

            uint16_t decompressed_element = current & mask;
            bool match = decompressed_element == predicate_key;
            output[oi++] = match;
            if (match) hits++;

            overflow_bits = bits_needed - unread_bits;
        }
        else {
            overflow_bits = 0;
        }
    }

    return hits;
}