#pragma once

#include <tmmintrin.h>
#include <vector>

class Table
{
private:
  static std::vector<uint16_t> columnA;
  static std::vector<uint16_t> columnB;
  static __m256i* compressedA;
  
  static void read(const std::string& filename);
public:
  static void load(const std::string& filename)
  {
      read(filename);
      compressedA = compress_9bit_input(columnA);
  }
  
  static __m256i* getColumnA( )
  {
      return compressedA;
  }
  
  static uint16_t get_A_at(unsigned int idx)
  {
      return columnA[idx];
  }
  
  static uint16_t get_B_at(unsigned int idx)
  {
      return columnB[idx];
  }
};
