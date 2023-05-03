cmake_minimum_required(VERSION 3.12)
project (caissa)


if (NOT DEFINED CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

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
    message(FATAL_ERROR "Unknown build configuration")
endif()


# detect target architecture if not defined
if (NOT DEFINED TARGET_ARCH)
    if (CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
        set(TARGET_ARCH "x64-avx2-bmi2")
    elseif (CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
        set(TARGET_ARCH "aarch64-neon")
    else()
        message(FATAL_ERROR "Unknown target architecture")
    endif()
endif()

# set target architecture compiler flags and definitions
if (TARGET_ARCH STREQUAL "x64-legacy")
    add_definitions(-DARCHITECTURE_X64)
elseif (TARGET_ARCH STREQUAL "x64-sse4-popcnt")
    add_definitions(-DARCHITECTURE_X64)
    add_definitions(-DUSE_SSE -DUSE_SSE2 -DUSE_SSE4 -DUSE_POPCNT)
    if (NOT MSVC)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4.2")
    endif()
elseif (TARGET_ARCH STREQUAL "x64-avx2-bmi2")
    add_definitions(-DARCHITECTURE_X64)
    add_definitions(-DUSE_SSE -DUSE_SSE2 -DUSE_SSE4 -DUSE_POPCNT)
    add_definitions(-DUSE_AVX -DUSE_AVX2 -DUSE_BMI2)
    if (MSVC)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX2")
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx -mfma -mavx2 -mbmi2")
    endif()
elseif (TARGET_ARCH STREQUAL "x64-avx512")
    add_definitions(-DARCHITECTURE_X64)
    add_definitions(-DUSE_SSE -DUSE_SSE2 -DUSE_SSE4 -DUSE_POPCNT)
    add_definitions(-DUSE_AVX -DUSE_AVX2 -DUSE_BMI2)
    add_definitions(-DUSE_AVX512)
    if (MSVC)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX512")
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfma -mavx2 -mbmi2 -mavx512bw")
    endif()
elseif (TARGET_ARCH STREQUAL "aarch64")
    add_definitions(-DARCHITECTURE_AARCH64)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8-a")
elseif (TARGET_ARCH STREQUAL "aarch64-neon")
    add_definitions(-DARCHITECTURE_AARCH64)
    add_definitions(-DUSE_ARM_NEON)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8-a+simd")
else()
    message(FATAL_ERROR "Unknown target architecture: ${TARGET_ARCH}")
endif()

message("Target architecture: ${TARGET_ARCH}")


# set C/C++ standards
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_C_STANDARD 17)


# optimization flags
if (MSVC)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /O2 /Oy /GS-")
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} /DEBUG /Od")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2 /Oy /GS-")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /DEBUG /Od")
else()
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O2")
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O0 -g")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g")
endif()


# enable link time code generation
if (MSVC)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /GL")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /GL")
	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /LTCG")
	set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} /LTCG")
	set(CMAKE_STATIC_LINKER_FLAGS "${CMAKE_STATIC_LINKER_FLAGS} /LTCG")
else()
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -flto")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -flto")
endif()


# set max warning level
if (MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4 /WX")
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /W3")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror")
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -Werror")
endif()


# disable C++ RTTI
if (MSVC)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /GR-")
else()
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")
endif()


# enable multiprocess compilation
if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /MP")
endif()


# warning exceptions
if (MSVC)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-multichar")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-switch")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-attributes")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-missing-field-initializers")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-gnu-anonymous-struct -Wno-nested-anon-types")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-four-char-constants")
endif()


set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)


add_subdirectory("src")
