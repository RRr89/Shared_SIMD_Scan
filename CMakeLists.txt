cmake_minimum_required(VERSION 3.0)
project(shared_simd_scan)

# main executable
file(GLOB_RECURSE SRC_FILES 
	${PROJECT_SOURCE_DIR}/src/*.cpp)

add_executable(${PROJECT_NAME} ${SRC_FILES})
target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/src)

if (MSVC)
    set_property(TARGET ${PROJECT_NAME} APPEND PROPERTY LINK_FLAGS /Profile)
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