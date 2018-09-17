cmake_minimum_required(VERSION 3.0)
project(shared_simd_scan)

SET(CMAKE_C_COMPILER /data/hdbcc/.hdbccdata/tools/gcc-6.2.1+r239768-2.4.sap20170207-linuxx86_64/bin/gcc)
SET(CMAKE_CXX_COMPILER /data/hdbcc/.hdbccdata/tools/gcc-6.2.1+r239768-2.4.sap20170207-linuxx86_64/bin/g++)

option(ENABLE_AVX "Enable AVX-aware code" ON)
option(ENABLE_AVX2 "Enable AVX2-aware code (AVX must be enabled)" ON)
option(ENABLE_PROFILING "Enable code sections for profiling" OFF)

# doesn't work on Windows!
include(checkVectorizationSupport.cmake)

# main executable
file(GLOB_RECURSE SRC_FILES 
	${PROJECT_SOURCE_DIR}/src/*.cpp)

add_executable(${PROJECT_NAME} ${SRC_FILES})
target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/src)

if (ENABLE_PROFILING)
	target_compile_definitions(${PROJECT_NAME} PRIVATE ENABLE_PROFILING=1)
else()
	target_compile_definitions(${PROJECT_NAME} PRIVATE ENABLE_PROFILING=0)
endif ()

if (${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
	# GCC Flags

	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++17 -msse3 -msse4 -msse4.1 -fopenmp")

	if (AVX_FOUND)
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx")

		if(ENABLE_AVX2)
			set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2")
		endif()
	endif()


elseif (${CMAKE_CXX_COMPILER_ID} STREQUAL "MSVC")
	# MSVC Flags

	target_compile_options(${PROJECT_NAME} PRIVATE /openmp)

	# needed to use some VS profiler features
	set_property(TARGET ${PROJECT_NAME} APPEND PROPERTY LINK_FLAGS /Profile)

	if (ENABLE_AVX)
		target_compile_options(${PROJECT_NAME} PRIVATE /arch:AVX)

		if(ENABLE_AVX2)
			target_compile_options(${PROJECT_NAME} PRIVATE /arch:AVX2)
		endif()
	endif()
endif()

# main library
add_library(${PROJECT_NAME}_lib ${SRC_FILES})
target_include_directories(${PROJECT_NAME}_lib PUBLIC ${PROJECT_SOURCE_DIR}/src)

# catch library
set(CATCH_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/lib/catch)
add_library(Catch INTERFACE)
target_include_directories(Catch INTERFACE ${CATCH_INCLUDE_DIR})

# unit tests
file(GLOB_RECURSE TEST_FILES 
	${PROJECT_SOURCE_DIR}/test/*.cpp)

add_executable(unit_tests ${TEST_FILES})
target_link_libraries(unit_tests Catch ${PROJECT_NAME}_lib)
target_include_directories(unit_tests PUBLIC ${PROJECT_SOURCE_DIR}/test)
