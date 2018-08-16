#include "simd_scan.hpp"
#include "util.hpp"

void decompress_standard(__m128i* input, size_t input_size, int* output)
{
    uint64_t* in = reinterpret_cast<uint64_t*>(input);
    auto bits_needed = BITS_NEEDED;
    auto mem_size = bits_needed * input_size;
    int array_size = ceil((double)mem_size / 64);

    uint64_t mask = (1 << bits_needed) - 1;

    uint64_t current = 0;
    size_t overflow_bits = 0;

    size_t oi = 0;

    for (size_t i = 0; i < array_size; i++)
    {
        current = in[i] >> overflow_bits;

        size_t unread_bits = 64 - overflow_bits;
        while (unread_bits >= bits_needed)
        {
            uint16_t decompressed_element = current & mask;
            output[oi++] = decompressed_element;

            if (oi == input_size)
            {
                return;
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
            output[oi++] = decompressed_element;

            overflow_bits = bits_needed - unread_bits;
        }
        else {
            overflow_bits = 0;
        }
    }
}

void decompress_128_sweep(__m128i* input, size_t input_size, int* output)
{
    size_t compression = BITS_NEEDED;
    size_t free_bits = 32 - compression; // most significant bits in result values that must be 0

    __m128i source = _mm_loadu_si128(input);
    int unread_bits = 128;

    size_t output_index = 0; // current write index of the output array (equals # of decompressed values)
    size_t total_processed_bytes = 0; // holds # of input bytes that have been processed completely

    while (output_index < input_size)
    {
        size_t input_offset[4] = {
            (compression * (output_index + 0) / 8) - total_processed_bytes,
            (compression * (output_index + 1) / 8) - total_processed_bytes,
            (compression * (output_index + 2) / 8) - total_processed_bytes,
            (compression * (output_index + 3) / 8) - total_processed_bytes
        };

        __m128i shuffle_mask = _mm_setr_epi8(
            input_offset[0], input_offset[0] + 1, input_offset[0] + 2, input_offset[0] + 3,
            input_offset[1], input_offset[1] + 1, input_offset[1] + 2, input_offset[1] + 3,
            input_offset[2], input_offset[2] + 1, input_offset[2] + 2, input_offset[2] + 3,
            input_offset[3], input_offset[3] + 1, input_offset[3] + 2, input_offset[3] + 3);

        __m128i b = _mm_shuffle_epi8(source, shuffle_mask);

        // shift left by variable amounts (using integer multiplication)
        size_t padding[4] = {
            compression * (output_index + 0) % 8,
            compression * (output_index + 1) % 8,
            compression * (output_index + 2) % 8,
            compression * (output_index + 3) % 8
        };

        __m128i mult = _mm_setr_epi32(
            1 << (free_bits - padding[0]),
            1 << (free_bits - padding[1]),
            1 << (free_bits - padding[2]),
            1 << (free_bits - padding[3]));

        __m128i c = _mm_mullo_epi32(b, mult);

        // shift right by fixed amount
        __m128i d = _mm_srli_epi32(c, 32 - compression);

        _mm_storeu_si128((__m128i*)&output[output_index], d);

        output_index += 4;
        unread_bits -= 4 * compression;

        // load next 
        if (unread_bits < 4 * compression)
        {
            // TODO uses unaligned loads --> possibly slow?
            total_processed_bytes = output_index * compression / 8;
            source = _mm_loadu_si128((__m128i*)&((uint8_t*)input)[total_processed_bytes]);
            unread_bits = 128;
        }
    }
}

void decompress_128_nosweep(__m128i* input, size_t input_size, int* output)
{
    size_t compression = BITS_NEEDED;
    size_t free_bits = 32 - compression; // most significant bits in result values that must be 0

    __m128i source = _mm_loadu_si128(input);

    size_t output_index = 0; // current write index of the output array (equals # of decompressed values)
    size_t total_processed_bytes = 0; // holds # of input bytes that have been processed completely

    while (output_index < input_size)
    {
        size_t input_offset[4] = {
            (compression * (output_index + 0) / 8) - total_processed_bytes,
            (compression * (output_index + 1) / 8) - total_processed_bytes,
            (compression * (output_index + 2) / 8) - total_processed_bytes,
            (compression * (output_index + 3) / 8) - total_processed_bytes
        };

        __m128i shuffle_mask = _mm_setr_epi8(
            input_offset[0], input_offset[0] + 1, input_offset[0] + 2, input_offset[0] + 3,
            input_offset[1], input_offset[1] + 1, input_offset[1] + 2, input_offset[1] + 3,
            input_offset[2], input_offset[2] + 1, input_offset[2] + 2, input_offset[2] + 3,
            input_offset[3], input_offset[3] + 1, input_offset[3] + 2, input_offset[3] + 3);

        __m128i b = _mm_shuffle_epi8(source, shuffle_mask);

        // shift left by variable amounts (using integer multiplication)
        size_t padding[4] = {
            compression * (output_index + 0) % 8,
            compression * (output_index + 1) % 8,
            compression * (output_index + 2) % 8,
            compression * (output_index + 3) % 8
        };

        __m128i mult = _mm_setr_epi32(
            1 << (free_bits - padding[0]),
            1 << (free_bits - padding[1]),
            1 << (free_bits - padding[2]),
            1 << (free_bits - padding[3]));

        __m128i c = _mm_mullo_epi32(b, mult);

        // shift right by fixed amount
        __m128i d = _mm_srli_epi32(c, 32 - compression);

        _mm_storeu_si128((__m128i*)&output[output_index], d);

        output_index += 4;

        // load next immediately
        // TODO uses unaligned loads --> possibly slow?
        total_processed_bytes = output_index * compression / 8;
        source = _mm_loadu_si128((__m128i*)&((uint8_t*)input)[total_processed_bytes]);
    }
}

void decompress_128_9bit(__m128i* input, size_t input_size, int* output)
{
    size_t compression = 9;
    size_t free_bits = 32 - compression; // most significant bits in result values that must be 0

    __m128i source = _mm_loadu_si128(input);

    size_t output_index = 0; // current write index of the output array (equals # of decompressed values)
    size_t total_processed_bytes = 0; // holds # of input bytes that have been processed completely

    // shuffle mask, pulled outside the loop (compression 9 has one constant shuffle mask)
    size_t input_offset[4] = { 0, 1, 2, 3 };

    __m128i shuffle_mask = _mm_setr_epi8(
        input_offset[0], input_offset[0] + 1, input_offset[0] + 2, input_offset[0] + 3,
        input_offset[1], input_offset[1] + 1, input_offset[1] + 2, input_offset[1] + 3,
        input_offset[2], input_offset[2] + 1, input_offset[2] + 2, input_offset[2] + 3,
        input_offset[3], input_offset[3] + 1, input_offset[3] + 2, input_offset[3] + 3);

    // shift masks (compression 9 has two shift masks used in alternation)
    __m128i shift_mask_1 = _mm_setr_epi32(
        1 << (free_bits - 0),
        1 << (free_bits - 1),
        1 << (free_bits - 2),
        1 << (free_bits - 3));

    __m128i shift_mask_2 = _mm_setr_epi32(
        1 << (free_bits - 4),
        1 << (free_bits - 5),
        1 << (free_bits - 6),
        1 << (free_bits - 7));

    while (output_index < input_size)
    {
        __m128i b = _mm_shuffle_epi8(source, shuffle_mask);

        __m128i c;
        if (output_index % 8 == 0)
        {
            c = _mm_mullo_epi32(b, shift_mask_1);
        }
        else 
        {
            c = _mm_mullo_epi32(b, shift_mask_2);
        }

        // shift right by fixed amount
        __m128i d = _mm_srli_epi32(c, 32 - compression);

        // TODO handle cases where output size in not multiple of 4
        _mm_storeu_si128((__m128i*)&output[output_index], d);

        output_index += 4;

        // load next
        total_processed_bytes = output_index * compression / 8;
        source = _mm_loadu_si128((__m128i*)&((uint8_t*)input)[total_processed_bytes]);
    }
}

void decompress_128(__m128i* input, size_t input_size, int* output)
{
    size_t compression = BITS_NEEDED;
    size_t free_bits = 32 - compression; // most significant bits in result values that must be 0

    __m128i source = _mm_loadu_si128(input);

    size_t output_index = 0; // current write index of the output array (equals # of decompressed values)
    size_t total_processed_bytes = 0; // holds # of input bytes that have been processed completely

    // shuffle masks
    size_t input_offset[8];
    for (size_t i = 0; i < 8; i++)
    {
        input_offset[i] = (compression * i) / 8;
    }
    size_t correction = input_offset[4];
    for (size_t i = 4; i < 8; i++) 
    {
        input_offset[i] -= correction;
    }

    __m128i shuffle_mask[2]; 
    shuffle_mask[0] = _mm_setr_epi8(
        input_offset[0], input_offset[0] + 1, input_offset[0] + 2, input_offset[0] + 3,
        input_offset[1], input_offset[1] + 1, input_offset[1] + 2, input_offset[1] + 3,
        input_offset[2], input_offset[2] + 1, input_offset[2] + 2, input_offset[2] + 3,
        input_offset[3], input_offset[3] + 1, input_offset[3] + 2, input_offset[3] + 3);
    shuffle_mask[1] = _mm_setr_epi8(
        input_offset[4], input_offset[4] + 1, input_offset[4] + 2, input_offset[4] + 3,
        input_offset[5], input_offset[5] + 1, input_offset[5] + 2, input_offset[5] + 3,
        input_offset[6], input_offset[6] + 1, input_offset[6] + 2, input_offset[6] + 3,
        input_offset[7], input_offset[7] + 1, input_offset[7] + 2, input_offset[7] + 3);
    
    // shift masks
    size_t padding[8];
    for (size_t i = 0; i < 8; i++)
    {
        padding[i] = (compression * i) % 8;
    }

    __m128i shift_mask[2];
    shift_mask[0] = _mm_setr_epi32(
        1 << (free_bits - padding[0]),
        1 << (free_bits - padding[1]),
        1 << (free_bits - padding[2]),
        1 << (free_bits - padding[3]));
    shift_mask[1] = _mm_setr_epi32(
        1 << (free_bits - padding[4]),
        1 << (free_bits - padding[5]),
        1 << (free_bits - padding[6]),
        1 << (free_bits - padding[7]));

    while (output_index < input_size)
    {
        size_t mask_index = output_index % 8 != 0;

        __m128i b = _mm_shuffle_epi8(source, shuffle_mask[mask_index]);

        __m128i c = _mm_mullo_epi32(b, shift_mask[mask_index]);

        // shift right by fixed amount
        __m128i d = _mm_srli_epi32(c, 32 - compression);

        _mm_storeu_si128((__m128i*)&output[output_index], d);

        output_index += 4;

        // load next
        total_processed_bytes = output_index * compression / 8;
        source = _mm_loadu_si128((__m128i*)&((uint8_t*)input)[total_processed_bytes]);
    }
}

void decompress_128_unrolled(__m128i* input, size_t input_size, int* output)
{
    size_t compression = BITS_NEEDED;
    size_t free_bits = 32 - compression; // most significant bits in result values that must be 0

    __m128i source = _mm_loadu_si128(input);

    size_t output_index = 0; // current write index of the output array (equals # of decompressed values)
    size_t total_processed_bytes = 0; // holds # of input bytes that have been processed completely

    // shuffle masks
    size_t input_offset[8];
    for (size_t i = 0; i < 8; i++)
    {
        input_offset[i] = (compression * i) / 8;
    }
    size_t correction = input_offset[4];
    for (size_t i = 4; i < 8; i++)
    {
        input_offset[i] -= correction;
    }

    __m128i shuffle_mask[2];
    shuffle_mask[0] = _mm_setr_epi8(
        input_offset[0], input_offset[0] + 1, input_offset[0] + 2, input_offset[0] + 3,
        input_offset[1], input_offset[1] + 1, input_offset[1] + 2, input_offset[1] + 3,
        input_offset[2], input_offset[2] + 1, input_offset[2] + 2, input_offset[2] + 3,
        input_offset[3], input_offset[3] + 1, input_offset[3] + 2, input_offset[3] + 3);
    shuffle_mask[1] = _mm_setr_epi8(
        input_offset[4], input_offset[4] + 1, input_offset[4] + 2, input_offset[4] + 3,
        input_offset[5], input_offset[5] + 1, input_offset[5] + 2, input_offset[5] + 3,
        input_offset[6], input_offset[6] + 1, input_offset[6] + 2, input_offset[6] + 3,
        input_offset[7], input_offset[7] + 1, input_offset[7] + 2, input_offset[7] + 3);

    // shift masks
    size_t padding[8];
    for (size_t i = 0; i < 8; i++)
    {
        padding[i] = (compression * i) % 8;
    }

    __m128i shift_mask[2];
    shift_mask[0] = _mm_setr_epi32(
        1 << (free_bits - padding[0]),
        1 << (free_bits - padding[1]),
        1 << (free_bits - padding[2]),
        1 << (free_bits - padding[3]));
    shift_mask[1] = _mm_setr_epi32(
        1 << (free_bits - padding[4]),
        1 << (free_bits - padding[5]),
        1 << (free_bits - padding[6]),
        1 << (free_bits - padding[7]));

    while (output_index < input_size)
    {
        {
            size_t mask_index = 0;
            __m128i b = _mm_shuffle_epi8(source, shuffle_mask[mask_index]);
            __m128i c = _mm_mullo_epi32(b, shift_mask[mask_index]);
            __m128i d = _mm_srli_epi32(c, 32 - compression);

            _mm_storeu_si128((__m128i*)&output[output_index], d);

            output_index += 4;

            // load next
            total_processed_bytes = output_index * compression / 8;
            source = _mm_loadu_si128((__m128i*)&((uint8_t*)input)[total_processed_bytes]);
        }

        {
            size_t mask_index = 1;
            __m128i b = _mm_shuffle_epi8(source, shuffle_mask[mask_index]);
            __m128i c = _mm_mullo_epi32(b, shift_mask[mask_index]);
            __m128i d = _mm_srli_epi32(c, 32 - compression);

            _mm_storeu_si128((__m128i*)&output[output_index], d);

            output_index += 4;

            // load next
            total_processed_bytes = output_index * compression / 8;
            source = _mm_loadu_si128((__m128i*)&((uint8_t*)input)[total_processed_bytes]);
        }
    }
}

inline __m128i _mm_alignr_epi8_nonconst(__m128i a, __m128i b, int count)
{
    switch (count)
    {
    case 0: return b;
    case 1: return _mm_alignr_epi8(a, b, 1);
    case 2: return _mm_alignr_epi8(a, b, 2);
    case 3: return _mm_alignr_epi8(a, b, 3);
    case 4: return _mm_alignr_epi8(a, b, 4);
    case 5: return _mm_alignr_epi8(a, b, 5);
    case 6: return _mm_alignr_epi8(a, b, 6);
    case 7: return _mm_alignr_epi8(a, b, 7);
    case 8: return _mm_alignr_epi8(a, b, 8);
    case 9: return _mm_alignr_epi8(a, b, 9);
    case 10: return _mm_alignr_epi8(a, b, 10);
    case 11: return _mm_alignr_epi8(a, b, 11);
    case 12: return _mm_alignr_epi8(a, b, 12);
    case 13: return _mm_alignr_epi8(a, b, 13);
    case 14: return _mm_alignr_epi8(a, b, 14);
    case 15: return _mm_alignr_epi8(a, b, 15);
    case 16: return a;
    }
}

void decompress_128_aligned(__m128i* input, size_t input_size, int* output)
{
    size_t compression = BITS_NEEDED;
    size_t free_bits = 32 - compression; // most significant bits in result values that must be 0

    size_t mi = 0;
    __m128i source = _mm_load_si128(&input[mi]);
    __m128i current = source;
    __m128i next = _mm_load_si128(&input[mi+1]);

    size_t output_index = 0; // current write index of the output array (equals # of decompressed values)
    size_t total_processed_bytes = 0; // holds # of input bytes that have been processed completely

    // shuffle masks
    size_t input_offset[8];
    for (size_t i = 0; i < 8; i++)
    {
        input_offset[i] = (compression * i) / 8;
    }
    size_t correction = input_offset[4];
    for (size_t i = 4; i < 8; i++)
    {
        input_offset[i] -= correction;
    }

    __m128i shuffle_mask[2];
    shuffle_mask[0] = _mm_setr_epi8(
        input_offset[0], input_offset[0] + 1, input_offset[0] + 2, input_offset[0] + 3,
        input_offset[1], input_offset[1] + 1, input_offset[1] + 2, input_offset[1] + 3,
        input_offset[2], input_offset[2] + 1, input_offset[2] + 2, input_offset[2] + 3,
        input_offset[3], input_offset[3] + 1, input_offset[3] + 2, input_offset[3] + 3);
    shuffle_mask[1] = _mm_setr_epi8(
        input_offset[4], input_offset[4] + 1, input_offset[4] + 2, input_offset[4] + 3,
        input_offset[5], input_offset[5] + 1, input_offset[5] + 2, input_offset[5] + 3,
        input_offset[6], input_offset[6] + 1, input_offset[6] + 2, input_offset[6] + 3,
        input_offset[7], input_offset[7] + 1, input_offset[7] + 2, input_offset[7] + 3);

    // shift masks
    size_t padding[8];
    for (size_t i = 0; i < 8; i++)
    {
        padding[i] = (compression * i) % 8;
    }

    __m128i shift_mask[2];
    shift_mask[0] = _mm_setr_epi32(
        1 << (free_bits - padding[0]),
        1 << (free_bits - padding[1]),
        1 << (free_bits - padding[2]),
        1 << (free_bits - padding[3]));
    shift_mask[1] = _mm_setr_epi32(
        1 << (free_bits - padding[4]),
        1 << (free_bits - padding[5]),
        1 << (free_bits - padding[6]),
        1 << (free_bits - padding[7]));

    while (output_index < input_size)
    {
        size_t mask_index = output_index % 8 != 0;

        __m128i b = _mm_shuffle_epi8(source, shuffle_mask[mask_index]);

        __m128i c = _mm_mullo_epi32(b, shift_mask[mask_index]);

        // shift right by fixed amount
        __m128i d = _mm_srli_epi32(c, 32 - compression);

        _mm_storeu_si128((__m128i*)&output[output_index], d);

        output_index += 4;

        // load next
        total_processed_bytes = output_index * compression / 8;
        if (total_processed_bytes / 16 != mi) 
        {
            mi++;
            current = next;
            next = _mm_load_si128(&input[mi + 1]);
        }
        source = _mm_alignr_epi8_nonconst(next, current, total_processed_bytes % 16);
    }
}

#ifdef __AVX__
void decompress_256(__m128i* input, size_t input_size, int* output)
{
    size_t compression = BITS_NEEDED;
    size_t free_bits = 32 - compression; // most significant bits in result values that must be 0

    //avxi_t source = _mm256_loadu_si256(input);
    __m256i source = _mm256_loadu2_m128i(input, input);

    size_t output_index = 0; // current write index of the output array (equals # of decompressed values)
    size_t total_processed_bytes = 0; // holds # of input bytes that have been processed completely

    // shuffle mask
    size_t input_offset[8];
    for (size_t i = 0; i < 8; i++)
    {
        input_offset[i] = (compression * i) / 8;
    }

    __m256i shuffle_mask = _mm256_setr_epi8(
        input_offset[0], input_offset[0] + 1, input_offset[0] + 2, input_offset[0] + 3,
        input_offset[1], input_offset[1] + 1, input_offset[1] + 2, input_offset[1] + 3,
        input_offset[2], input_offset[2] + 1, input_offset[2] + 2, input_offset[2] + 3,
        input_offset[3], input_offset[3] + 1, input_offset[3] + 2, input_offset[3] + 3,

        input_offset[4], input_offset[4] + 1, input_offset[4] + 2, input_offset[4] + 3,
        input_offset[5], input_offset[5] + 1, input_offset[5] + 2, input_offset[5] + 3,
        input_offset[6], input_offset[6] + 1, input_offset[6] + 2, input_offset[6] + 3,
        input_offset[7], input_offset[7] + 1, input_offset[7] + 2, input_offset[7] + 3);
    
    // shift mask
    size_t padding[8];
    for (size_t i = 0; i < 8; i++)
    {
        padding[i] = (compression * i) % 8;
    }

    __m256i shift_mask = _mm256_setr_epi32(
        1 << (free_bits - padding[0]),
        1 << (free_bits - padding[1]),
        1 << (free_bits - padding[2]),
        1 << (free_bits - padding[3]),
        1 << (free_bits - padding[4]),
        1 << (free_bits - padding[5]),
        1 << (free_bits - padding[6]),
        1 << (free_bits - padding[7]));

    while (output_index < input_size)
    {
        __m256i b = _mm256_shuffle_epi8(source, shuffle_mask);

        __m256i c = _mm256_mullo_epi32(b, shift_mask);

        // shift right by fixed amount
        __m256i d = _mm256_srli_epi32(c, 32 - compression);

        _mm256_storeu_si256((__m256i*)&output[output_index], d);

        output_index += 8;

        // load next
        total_processed_bytes = output_index * compression / 8;
        //source = _mm_loadu_si128((__m128i*)&((uint8_t*)input)[total_processed_bytes]);
        __m128i* next = (__m128i*)&((uint8_t*)input)[total_processed_bytes];
        source = _mm256_loadu2_m128i(next, next);
    }
}
#endif

#ifdef __AVX2__
void decompress_256_avx2(__m128i* input, size_t input_size, int* output)
{
    size_t compression = BITS_NEEDED;
    size_t free_bits = 32 - compression; // most significant bits in result values that must be 0

    __m256i source = _mm256_loadu2_m128i(input, input);

    size_t output_index = 0; // current write index of the output array (equals # of decompressed values)
    size_t total_processed_bytes = 0; // holds # of input bytes that have been processed completely

    // shuffle mask
    size_t input_offset[8];
    for (size_t i = 0; i < 8; i++)
    {
        input_offset[i] = (compression * i) / 8;
    }

    __m256i shuffle_mask = _mm256_setr_epi8(
        input_offset[0], input_offset[0] + 1, input_offset[0] + 2, input_offset[0] + 3,
        input_offset[1], input_offset[1] + 1, input_offset[1] + 2, input_offset[1] + 3,
        input_offset[2], input_offset[2] + 1, input_offset[2] + 2, input_offset[2] + 3,
        input_offset[3], input_offset[3] + 1, input_offset[3] + 2, input_offset[3] + 3,

        input_offset[4], input_offset[4] + 1, input_offset[4] + 2, input_offset[4] + 3,
        input_offset[5], input_offset[5] + 1, input_offset[5] + 2, input_offset[5] + 3,
        input_offset[6], input_offset[6] + 1, input_offset[6] + 2, input_offset[6] + 3,
        input_offset[7], input_offset[7] + 1, input_offset[7] + 2, input_offset[7] + 3);

    // shift mask
    size_t padding[8];
    for (size_t i = 0; i < 8; i++)
    {
        padding[i] = (compression * i) % 8;
    }

    __m256i shift_mask = _mm256_setr_epi32(
        padding[0], padding[1], padding[2], padding[3], 
        padding[4], padding[5], padding[6], padding[7]);

    // and masking (is needed here since we only do one shift!)	
    uint32_t mask = (1 << compression) - 1;
    __m256i and_mask = _mm256_set1_epi32(mask);

    while (output_index < input_size)
    {
        __m256i b = _mm256_shuffle_epi8(source, shuffle_mask);

        // shift right by variable amount (according to static shift mask)
        __m256i c = _mm256_srlv_epi32(b, shift_mask);

        __m256i d = _mm256_and_si256(c, and_mask);

        _mm256_storeu_si256((__m256i*)&output[output_index], d);

        output_index += 8;

        // load next
        total_processed_bytes = output_index * compression / 8;
        //source = _mm_loadu_si128((__m128i*)&((uint8_t*)input)[total_processed_bytes]);
        __m128i* next = (__m128i*)&((uint8_t*)input)[total_processed_bytes];
        source = _mm256_loadu2_m128i(next, next);
    }
}
#endif

