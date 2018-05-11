set (CMAKE_CXX_STANDARD 11)

cmake_minimum_required(VERSION 3.0)
project(adms_labs)



set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx -Wall -g -std=c++11")
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  # using Clang
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  # using GCC
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
  # using Intel C++
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
  # using Visual Studio C++
endif()

add_subdirectory(Client)
add_subdirectory(Server)
add_subdirectory(Table)
add_subdirectory(utils)
