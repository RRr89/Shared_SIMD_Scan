#include <iostream>

#include "profiling.hpp"


std::chrono::nanoseconds _clock() 
{
    static std::chrono::high_resolution_clock::time_point last;
    auto now = std::chrono::high_resolution_clock::now();
    auto elapsed = now - last;
    last = now;
    return elapsed;
}

std::unordered_map<std::string, ProfileSample> __samples(16);

ProfileSample::ProfileSample()
    : ProfileSample("?")
{ }

ProfileSample::ProfileSample(std::string id)
    : id(id), last(), running_count(0), running_sum()
{ }

ProfileSample::~ProfileSample()
{
    std::cout << "[profiler] " << id << ": " << avg().count() << " (avg over " << running_count << " samples)" << std::endl;
    __samples[id] = *this;
}

void ProfileSample::start()
{
    last = std::chrono::high_resolution_clock::now();
}

void ProfileSample::end()
{
    auto now = std::chrono::high_resolution_clock::now();
    auto diff = now - last;
    running_sum += diff;
    running_count++;
}

std::chrono::nanoseconds ProfileSample::avg() const
{
    return running_sum / running_count;
}

std::chrono::nanoseconds get_sample(std::string id)
{
    return __samples[id].avg();
}
