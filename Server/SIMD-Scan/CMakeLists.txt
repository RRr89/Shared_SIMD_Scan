
cmake_minimum_required (VERSION 3.3)
set(CMAKE_C_COMPILER "gcc")
set(CMAKE_CXX_COMPILER "g++")
project(simdscan CXX)

include(CTest)
include ( "${PROJECT_SOURCE_DIR}/cmake/ParseAndAddCatchTests.cmake")

set (CMAKE_CXX_STANDARD 14)
set (CMAKE_CXX_STANDARD_REQUIRED ON)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx -Wall -Wextra")

if(NOT CMAKE_BUILD_TYPE) 
    set(CMAKE_BUILD_TYPE Release)
endif(NOT CMAKE_BUILD_TYPE)

# Prepare "Catch" library for other executables
set(CATCH_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/catch)
add_library(Catch INTERFACE)
target_include_directories(Catch INTERFACE ${CATCH_INCLUDE_DIR})

#Only for memory dump - you can comment this out if not needed
find_package(Boost 1.67.0)
if(Boost_FOUND)
  include_directories(${Boost_INCLUDE_DIRS})
endif()

file (GLOB PROJECT_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
#add_library (${PROJECT_NAME}_lib STATIC ${PROJECT_SOURCES} ${PROJECT_HEADERS})
add_library (${PROJECT_NAME}_lib STATIC ${PROJECT_SOURCES})

file (GLOB PROJECT_TESTS "${CMAKE_CURRENT_SOURCE_DIR}/test/*.cpp")
add_executable(${PROJECT_NAME}_test ${PROJECT_TESTS})
target_link_libraries(${PROJECT_NAME}_test Catch ${PROJECT_NAME}_lib)
enable_testing(true)  # Enables unit-testing
ParseAndAddCatchTests (${PROJECT_NAME}_test)