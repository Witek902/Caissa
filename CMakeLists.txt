cmake_minimum_required(VERSION 2.8.11)
project (caissa)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O2 -std=c99")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O0 -g -std=c99")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -std=c++2a")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g -std=c++2a")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wpedantic -Wextra")

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    message("Building in Debug configuration")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCONFIGURATION_DEBUG")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DCONFIGURATION_DEBUG")
elseif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    message("Building in Rlease configuration")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCONFIGURATION_RELEASE")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DCONFIGURATION_RELEASE")
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    message("Building in Final configuration")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCONFIGURATION_FINAL")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DCONFIGURATION_FINAL")
else()
    message("Unknown configuration")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfma")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mbmi2")

add_definitions(-DARCHITECTURE_X64)
add_definitions(-DUSE_SSE)
add_definitions(-DUSE_SSE2)
add_definitions(-DUSE_SSE4)
add_definitions(-DUSE_POPCNT)
add_definitions(-DUSE_AVX)
add_definitions(-DUSE_AVX2)
add_definitions(-DUSE_BMI2)

# Warnings excpetions
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-multichar")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-switch")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-attributes")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-missing-field-initializers")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-gnu-anonymous-struct -Wno-nested-anon-types")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-four-char-constants")

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_subdirectory("src")
