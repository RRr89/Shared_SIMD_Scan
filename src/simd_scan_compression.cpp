#include "simd_scan.hpp"
#include "util.hpp"

// original compression function (with some bug fixes...) for reference
void* compress_9bit_input_original(std::vector<uint16_t>& input)
{
    auto bits_needed = BITS_NEEDED;
    auto mem_size = bits_needed * input.size();
    int array_size = ceil((double)mem_size / 64);
    auto buffer = new long long[array_size](); // TODO allocated on heap!

    int remaining_buffer_size = 64;
    int idx_ = 0;
    for (size_t i = 0; i < input.size(); i++)
    {
        long long tmp_buffer = 0;
        tmp_buffer = tmp_buffer | input[i];
        tmp_buffer = tmp_buffer << (i * bits_needed); // TODO undefined behaviour?
        buffer[idx_] = buffer[idx_] | tmp_buffer;
        remaining_buffer_size -= bits_needed;

        if (remaining_buffer_size == 0)
        {
            idx_++;
            remaining_buffer_size = 64;
            continue;
        }
        else if (remaining_buffer_size < bits_needed)
        {
            // logic to handle overflow_bits
            i++; // TODO index out of range here
            if (i == input.size()) break; // quick fix for that...

            tmp_buffer = 0;
            tmp_buffer = tmp_buffer | input[i];
            tmp_buffer = tmp_buffer << (64 - remaining_buffer_size);
            buffer[idx_] = buffer[idx_] | tmp_buffer;

            idx_++;

            // Second half
            tmp_buffer = 0;
            tmp_buffer = tmp_buffer | input[i];
            tmp_buffer = tmp_buffer >> remaining_buffer_size;
            buffer[idx_] = buffer[idx_] | tmp_buffer;
            remaining_buffer_size = 64 - (bits_needed - remaining_buffer_size);
        }
    }

    return (void*) buffer;
}

std::unique_ptr<uint64_t[]> compress_9bit_input(std::vector<uint16_t>& input)
{
    const int element_size = 8 * sizeof(uint64_t);

    auto bits_needed = BITS_NEEDED;
    auto mem_size = bits_needed * input.size();
    int array_size = next_multiple(ceil((double)mem_size / element_size), 4);

    auto buffer = std::make_unique<uint64_t[]>(array_size);

    // these are for making the ptr align to 16/32 byte boundaries,
    // turns out it makes no difference in performance though.
    //auto buffer = std::unique_ptr<uint64_t[]>((uint64_t*) new __m128i[array_size / 2]);
    //auto buffer = std::unique_ptr<uint64_t[]>((uint64_t*) new __m256i[array_size / 4]);
    //auto buffer = std::unique_ptr<uint64_t[]>((uint64_t*) _mm_malloc(array_size * sizeof(uint64_t), 256));

    int remaining_buffer_size = element_size;
    int idx_ = 0;
    for (size_t i = 0; i < input.size(); i++)
    {
        uint64_t tmp_buffer = 0;
        tmp_buffer = tmp_buffer | input[i];
        tmp_buffer = tmp_buffer << (i * bits_needed); // TODO undefined behaviour?
        buffer[idx_] = buffer[idx_] | tmp_buffer;
        remaining_buffer_size -= bits_needed;

        if (remaining_buffer_size == 0)
        {
            idx_++;
            remaining_buffer_size = element_size;
            continue;
        }
        else if (remaining_buffer_size < bits_needed)
        {
            i++;
            if (i == input.size()) break;

            tmp_buffer = 0;
            tmp_buffer = tmp_buffer | input[i];
            tmp_buffer = tmp_buffer << (element_size - remaining_buffer_size);
            buffer[idx_] = buffer[idx_] | tmp_buffer;

            idx_++;

            // Second half
            tmp_buffer = 0;
            tmp_buffer = tmp_buffer | input[i];
            tmp_buffer = tmp_buffer >> remaining_buffer_size;
            buffer[idx_] = buffer[idx_] | tmp_buffer;
            remaining_buffer_size = element_size - (bits_needed - remaining_buffer_size);
        }
    }

    return buffer;
}
