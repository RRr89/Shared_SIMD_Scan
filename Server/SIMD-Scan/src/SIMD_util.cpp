#ifndef SIMD_UTIL_H
#define SIMD_UTIL_H

#include <iostream>
#include <tmmintrin.h>
#include <immintrin.h>
#include <cmath>
#include <vector>
#include <bitset>
#include <boost/format.hpp>

//#define DEBUGCODE
#if defined(DEBUGCODE)
#define DebugCode(code_fragment) \
    {                            \
        code_fragment            \
    }
#else
#define DebugCode(code_fragment)
#endif

using namespace std;

enum PRINTOPTIONS
{
    hex,
    integer,
    bits,
    all
};

enum INDENT_UNALIGNED : bool
{
    YES,
    NO,
};

void dump_bits(void *ptr,
               std::size_t size, INDENT_UNALIGNED align,
               std::ostream &os = std::cout)
{
    typedef unsigned char byte;
    typedef unsigned long uint;

DebugCode(
        std::cout << "dump_bits: calling reinterpret_cast on ptr" << '\n';
        );
    // Allow direct arithmetic on the pointer
    uint iptr = reinterpret_cast<uint>(ptr);

    // If the object is not aligned
    if (iptr % 16 != 0)
    {
        // Print the first address
        os << boost::format("\nBit Values:%|20t|");

        // Indent to the offset
        if (align == INDENT_UNALIGNED::YES)
        {
            // Indent to the offset
            for (std::size_t i = 0; i < iptr % 16; ++i)
            {
                os << boost::format("%|15t|");
            }
        }
    }
    // Dump the memory
    for (std::size_t i = 0; i < size; ++i, ++iptr)
    {
        // New line and address every 16 bytes, spaces every 4 bytes
        if (iptr % 16 == 0)
        {
            // |...t| defines space at beginning of line
            os << boost::format("\nBit Values:%|20t|");
        }

        // Write the address contents
        //|...t| defines space between bytes
        os << boost::format("%1%%|15t|") % static_cast<bitset<8>>(*reinterpret_cast<byte *>(iptr));
    }
    os << std::endl;
}

void dump_int(void *ptr,
              std::size_t size, INDENT_UNALIGNED align,
              std::ostream &os = std::cout)
{
    typedef unsigned char byte;
    typedef unsigned long uint;

    // Allow direct arithmetic on the pointer
    DebugCode(
        std::cout << "dump_int: calling reinterpret_cast on ptr" << '\n';
        );
    uint iptr = reinterpret_cast<uint>(ptr);
    // If the object is not aligned
    if (iptr % 16 != 0)
    {
        // Print the first address
        os << boost::format("\nDec. Values:%|20t|") << std::endl;

        if (align == INDENT_UNALIGNED::YES)
        {
            // Indent to the offset
            for (std::size_t i = 0; i < iptr % 16; ++i)
            {
                os << boost::format("%|15t|");
            }
        }
    }
    // Dump the memory
    for (std::size_t i = 0; i < size; ++i, ++iptr)
    {
        // New line and address every 16 bytes, spaces every 4 bytes
        if (iptr % 16 == 0)
        {
            os << boost::format("\nDec. Values:%|20t|");
        }
        // Write the address contents
        os << boost::format("%d%|10t|") % static_cast<uint>(*reinterpret_cast<byte *>(iptr));
    }
    os << std::endl;
}

void dump_hex(void *ptr,
              std::size_t size, INDENT_UNALIGNED align,
              std::ostream &os = std::cout)
{
    typedef unsigned char byte;
    typedef unsigned long uint;

DebugCode(
        std::cout << "dump_hex: calling reinterpret_cast on ptr" << '\n';
        );
    // Allow direct arithmetic on the pointer
    uint iptr = reinterpret_cast<uint>(ptr);
    // If the object is not aligned
    if (iptr % 16 != 0)
    {
        // Print the first address
        os << boost::format("\n0x%016lX:%|20t|") % (iptr & ~15);

        if (align == INDENT_UNALIGNED::YES)
        {
            // Indent to the offset
            for (std::size_t i = 0; i < iptr % 16; ++i)
            {
                os << boost::format("%|15t|");
            }
        }
    }

    // Dump the memory
    for (std::size_t i = 0; i < size; ++i, ++iptr)
    {
        // New line and address every 16 bytes, spaces every 4 bytes
        if (iptr % 16 == 0)
        {
            os << boost::format("\n0x%016lX:%|20t|") % iptr;
        }

        // Write the address contents
        os << boost::format("%02hhX%|10t|") % static_cast<uint>(*reinterpret_cast<byte *>(iptr));
    }

    os << std::endl;
}

void dump_memory(void *ptr,
                 std::size_t size, PRINTOPTIONS opt, INDENT_UNALIGNED align,
                 std::ostream &os = std::cout)
{
    // typedef unsigned char byte;
    // typedef unsigned long uint;

    DebugCode(
        std::cout << "dump_memory: size is " << size << '\n';
        );
    os << "-----------------------------------------------------------------------\n";
    os << boost::format("%d bytes%|10t|") % size;
    DebugCode(
        std::cout << "dump_memory: boost::format % size done" << '\n';
        );

    switch (opt)
    {
    case PRINTOPTIONS::hex:
        dump_hex(ptr, size, align);
        break;
    case PRINTOPTIONS::integer:
        dump_int(ptr, size, align);
        break;
    case PRINTOPTIONS::bits:
        dump_bits(ptr, size, align);
        break;
    case PRINTOPTIONS::all:
        DebugCode(
        std::cout << "dump_memory: In case PRINTOPTIONS::all" << '\n';
        );
        dump_hex(ptr, size, align);
        dump_int(ptr, size, align);
        dump_bits(ptr, size, align);
        break;
    }
    os << "\n-----------------------------------------------------------------------"
       << std::endl;
}

#endif