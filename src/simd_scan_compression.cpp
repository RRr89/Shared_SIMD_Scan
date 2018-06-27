#include "simd_scan.hpp"

avxiptr_t compress_9bit_input(std::vector<uint16_t>& input)
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

	return (avxiptr_t) buffer;
}

