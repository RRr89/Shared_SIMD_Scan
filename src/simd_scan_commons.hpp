#pragma once

#define _mm256_loadu2_m128i(hi, lo) (_mm256_set_m128i(_mm_loadu_si128(hi), _mm_loadu_si128(lo)))

inline void generate_shuffle_mask_128(int compression, __m128i shuffle_mask[2])
{
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
}

inline void generate_shift_masks_128(int compression, __m128i shift_mask[2])
{
    size_t free_bits = 32 - compression;

    size_t padding[8];
    for (size_t i = 0; i < 8; i++)
    {
        padding[i] = (compression * i) % 8;
    }

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
}

inline void generate_clean_masks_128(int compression, __m128i clean_mask[2])
{
    size_t padding[8];
    for (size_t i = 0; i < 8; i++)
    {
        padding[i] = (compression * i) % 8;
    }

    clean_mask[0] = _mm_setr_epi32(
        ((1 << compression) - 1) << padding[0],
        ((1 << compression) - 1) << padding[1],
        ((1 << compression) - 1) << padding[2],
        ((1 << compression) - 1) << padding[3]);
    clean_mask[1] = _mm_setr_epi32(
        ((1 << compression) - 1) << padding[4],
        ((1 << compression) - 1) << padding[5],
        ((1 << compression) - 1) << padding[6],
        ((1 << compression) - 1) << padding[7]);
}

inline void generate_predicate_masks_128(int compression, int predicate_key, __m128i predicate[2])
{
    size_t padding[8];
    for (size_t i = 0; i < 8; i++)
    {
        padding[i] = (compression * i) % 8;
    }

    predicate[0] = _mm_setr_epi32(
        predicate_key << padding[0],
        predicate_key << padding[1],
        predicate_key << padding[2],
        predicate_key << padding[3]);
    predicate[1] = _mm_setr_epi32(
        predicate_key << padding[4],
        predicate_key << padding[5],
        predicate_key << padding[6],
        predicate_key << padding[7]);
}

#ifdef __AVX__
inline __m256i generate_shuffle_mask_256(int compression)
{
    size_t input_offset[8];
    for (size_t i = 0; i < 8; i++)
    {
        input_offset[i] = (compression * i) / 8;
    }

    return _mm256_setr_epi8(
        input_offset[0], input_offset[0] + 1, input_offset[0] + 2, input_offset[0] + 3,
        input_offset[1], input_offset[1] + 1, input_offset[1] + 2, input_offset[1] + 3,
        input_offset[2], input_offset[2] + 1, input_offset[2] + 2, input_offset[2] + 3,
        input_offset[3], input_offset[3] + 1, input_offset[3] + 2, input_offset[3] + 3,

        input_offset[4], input_offset[4] + 1, input_offset[4] + 2, input_offset[4] + 3,
        input_offset[5], input_offset[5] + 1, input_offset[5] + 2, input_offset[5] + 3,
        input_offset[6], input_offset[6] + 1, input_offset[6] + 2, input_offset[6] + 3,
        input_offset[7], input_offset[7] + 1, input_offset[7] + 2, input_offset[7] + 3);
}

inline __m256i generate_shift_mask_256(int compression)
{
    size_t free_bits = 32 - compression;

    size_t padding[8];
    for (size_t i = 0; i < 8; i++)
    {
        padding[i] = (compression * i) % 8;
    }

    return _mm256_setr_epi32(
        1 << (free_bits - padding[0]),
        1 << (free_bits - padding[1]),
        1 << (free_bits - padding[2]),
        1 << (free_bits - padding[3]),
        1 << (free_bits - padding[4]),
        1 << (free_bits - padding[5]),
        1 << (free_bits - padding[6]),
        1 << (free_bits - padding[7]));
}

inline __m256i generate_clean_mask_256(int compression)
{
    size_t padding[8];
    for (size_t i = 0; i < 8; i++)
    {
        padding[i] = (compression * i) % 8;
    }

    return _mm256_setr_epi32(
        ((1 << compression) - 1) << padding[0],
        ((1 << compression) - 1) << padding[1],
        ((1 << compression) - 1) << padding[2],
        ((1 << compression) - 1) << padding[3],
        ((1 << compression) - 1) << padding[4],
        ((1 << compression) - 1) << padding[5],
        ((1 << compression) - 1) << padding[6],
        ((1 << compression) - 1) << padding[7]);
}

inline __m256i generate_predicate_mask_256(int compression, int predicate_key)
{
    size_t padding[8];
    for (size_t i = 0; i < 8; i++)
    {
        padding[i] = (compression * i) % 8;
    }

    return _mm256_setr_epi32(
        predicate_key << padding[0],
        predicate_key << padding[1],
        predicate_key << padding[2],
        predicate_key << padding[3],
        predicate_key << padding[4],
        predicate_key << padding[5],
        predicate_key << padding[6],
        predicate_key << padding[7]);
}
#endif