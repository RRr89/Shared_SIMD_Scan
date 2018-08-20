#include "simd_scan.hpp"

void shared_scan_128_sequential(std::vector<int>& predicate_keys, __m128i* input, size_t input_size, std::vector<std::vector<bool>>& outputs)
{
    for (size_t i = 0; i < predicate_keys.size(); i++)
    {
        scan_128(predicate_keys[i], input, input_size, outputs[i]);
    }
}

void shared_scan_128_threaded(std::vector<int>& predicate_keys, __m128i* input, size_t input_size, std::vector<std::vector<bool>>& outputs)
{
    #pragma omp parallel for
    for (int i = 0; i < predicate_keys.size(); i++)
    {
        scan_128(predicate_keys[i], input, input_size, outputs[i]);
    }
}