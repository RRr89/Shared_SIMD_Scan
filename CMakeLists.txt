cmake_minimum_required(VERSION 3.0)
project(shared_simd_scan)

file(GLOB_RECURSE SRC_FILES src/*.h src/*.cpp)

add_executable(shared_simd_scan ${SRC_FILES})
target_include_directories(shared_simd_scan PRIVATE ${PROJECT_SOURCE_DIR}/src)