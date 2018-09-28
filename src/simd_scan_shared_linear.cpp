#include <immintrin.h>
#include <algorithm>
#include <cstring> // memcpy

#include "simd_scan.hpp"
#include "simd_scan_commons.hpp"
#include "util.hpp"

void shared_scan_128_linear_standard(std::vector<int> const& predicate_keys, __m128i* input, size_t input_size, std::vector<uint8_t>& outputs)
{
    size_t predicate_key_count = predicate_keys.size();
    size_t compression = BITS_NEEDED;

    __m128i source = _mm_loadu_si128(input);

    size_t output_index = 0; // current write index of the output array

    __m128i shuffle_mask[2];
    generate_shuffle_mask_128(compression, shuffle_mask);
 
    __m128i shift_mask[2];
    generate_shift_masks_128(compression, shift_mask);

    while (8 * output_index < input_size)
    {
        __m128i d1, d2;

        {
            size_t mask_index = 0;
            __m128i b = _mm_shuffle_epi8(source, shuffle_mask[mask_index]);
            __m128i c = _mm_mullo_epi32(b, shift_mask[mask_index]);
            d1 = _mm_srli_epi32(c, 32 - compression);

            size_t total_processed_bytes = (8 * output_index + 4) * compression / 8;
            source = _mm_loadu_si128((__m128i*)&((uint8_t*)input)[total_processed_bytes]);
        }

        {
            size_t mask_index = 1;
            __m128i b = _mm_shuffle_epi8(source, shuffle_mask[mask_index]);
            __m128i c = _mm_mullo_epi32(b, shift_mask[mask_index]);
            d2 = _mm_srli_epi32(c, 32 - compression);

            size_t total_processed_bytes = (8 * output_index + 8) * compression / 8;
            source = _mm_loadu_si128((__m128i*)&((uint8_t*)input)[total_processed_bytes]);
        }

        for (size_t key_id = 0; key_id < predicate_key_count; key_id++)
        {
            __m128i predicate = _mm_set1_epi32(predicate_keys[key_id]);
            __m128i e1 = _mm_cmpeq_epi32(d1, predicate);
            __m128i e2 = _mm_cmpeq_epi32(d2, predicate);

            uint8_t matches1 = _mm_movemask_ps(_mm_castsi128_ps(e1));
            uint8_t matches2 = _mm_movemask_ps(_mm_castsi128_ps(e2));

            outputs[(output_index * predicate_key_count) + key_id] = matches1 | (matches2 << 4);
        }

        output_index += 1;
    }
}

void shared_scan_128_linear_simple(std::vector<int> const& predicate_keys, __m128i* input, size_t input_size, std::vector<uint8_t>& output)
{
    switch (predicate_keys.size())
    {
        case 1: shared_scan_128_linear_static<1>(predicate_keys, input, input_size, output); break;
        case 2: shared_scan_128_linear_static<2>(predicate_keys, input, input_size, output); break;
        case 4: shared_scan_128_linear_static<4>(predicate_keys, input, input_size, output); break;
        case 8: shared_scan_128_linear_static<8>(predicate_keys, input, input_size, output); break;
        case 16: shared_scan_128_linear_static<16>(predicate_keys, input, input_size, output); break;
        case 32: shared_scan_128_linear_static<32>(predicate_keys, input, input_size, output); break;
        case 64: shared_scan_128_linear_static<64>(predicate_keys, input, input_size, output); break;
        case 128: shared_scan_128_linear_static<128>(predicate_keys, input, input_size, output); break;
        case 256: shared_scan_128_linear_static<256>(predicate_keys, input, input_size, output); break;
        case 512: shared_scan_128_linear_static<512>(predicate_keys, input, input_size, output); break;
        case 1024: shared_scan_128_linear_static<1024>(predicate_keys, input, input_size, output); break;
        default:
            std::cerr << "not supported for " << predicate_keys.size() << " predicate keys!" << std::endl;
    }
}