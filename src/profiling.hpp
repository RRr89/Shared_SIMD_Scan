#pragma once

#include <chrono>
#include <unordered_map>
#include <string>

std::chrono::nanoseconds _clock();

class ProfileSample 
{
private:
    std::string id;
    uint64_t last;
    size_t running_count;
    uint64_t running_sum;

public:
    ProfileSample();

    ProfileSample(std::string id);

    ~ProfileSample();

    void start();

    void end();

    uint64_t avg() const;
};

std::chrono::nanoseconds get_sample(std::string id);

#ifndef ENABLE_PROFILING
#define ENABLE_PROFILING 0
#endif

#if ENABLE_PROFILING

#define PROFILE_SAMPLE(id) ProfileSample _sample_##id(#id)
#define PROFILE_BLOCK_START(id) _sample_##id.start()
#define PROFILE_BLOCK_END(id) _sample_##id.end()

#else 

#define PROFILE_SAMPLE(id)
#define PROFILE_BLOCK_START(id)
#define PROFILE_BLOCK_END(id)

#endif