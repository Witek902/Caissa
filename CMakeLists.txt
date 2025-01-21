cmake_minimum_required(VERSION 3.15)

project (caissa)


# enable IPO (interprocedural optimization) if supported
include(CheckIPOSupported)
check_ipo_supported(RESULT supported OUTPUT error)


# set supported configurations
set(CMAKE_CONFIGURATION_TYPES "Debug;Release;Final" CACHE STRING "" FORCE)

IF(NOT CMAKE_BUILD_TYPE)
   SET(CMAKE_BUILD_TYPE Final
       CACHE STRING "Choose one of build types: Debug Release Final."
       FORCE)
ENDIF(NOT CMAKE_BUILD_TYPE)
message("* Current build type: ${CMAKE_BUILD_TYPE}")

# set definitions for each configuration
set_property(DIRECTORY APPEND PROPERTY COMPILE_DEFINITIONS
    $<$<CONFIG:Debug>:CONFIGURATION_DEBUG>
    $<$<CONFIG:Release>:CONFIGURATION_RELEASE>
    $<$<CONFIG:Final>:CONFIGURATION_FINAL>
)

# detect target architecture if not defined
if (NOT DEFINED TARGET_ARCH)
    if (CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
        set(TARGET_ARCH "x64-bmi2")
    elseif (CMAKE_SYSTEM_PROCESSOR STREQUAL "AMD64")
        set(TARGET_ARCH "x64-bmi2")
    elseif (CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
        set(TARGET_ARCH "aarch64-neon")
    else()
        message(FATAL_ERROR "Unknown target architecture")
    endif()
endif()

# set target architecture compiler flags and definitions
if (TARGET_ARCH STREQUAL "x64-legacy")
    add_definitions(-DARCHITECTURE_X64)
    add_definitions(-DUSE_SSE -DUSE_SSE2)
elseif (TARGET_ARCH STREQUAL "x64-sse4-popcnt")
    add_definitions(-DARCHITECTURE_X64)
    add_definitions(-DUSE_SSE -DUSE_SSE2 -DUSE_SSE4 -DUSE_POPCNT)
    if (NOT MSVC)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4.2")
    endif()
elseif (TARGET_ARCH STREQUAL "x64-avx2")
    add_definitions(-DARCHITECTURE_X64)
    add_definitions(-DUSE_SSE -DUSE_SSE2 -DUSE_SSE4 -DUSE_POPCNT)
    add_definitions(-DUSE_AVX -DUSE_AVX2)
    if (MSVC)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX2")
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx -mfma -mavx2")
    endif()
elseif (TARGET_ARCH STREQUAL "x64-bmi2")
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
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfma -mavx2 -mbmi2 -mavx512bw -mavx512f")
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

message("* Current architecture: ${TARGET_ARCH}")


# set C/C++ standards
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_C_STANDARD 11)


# optimization flags
if (MSVC)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /O2 /Oy /GS- /Zi")
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} /DDEBUG /Od /Zi")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2 /Oy /GS- /Zi")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /DDEBUG /Od /Zi")
else()
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -g")
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O0 -g")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -g")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g")
endif()


if (MSVC)
	set(CMAKE_EXE_LINKER_FLAGS "/DEBUG:FULL")
	set(CMAKE_EXE_LINKER_FLAGS "/OPT:ICF")
	set(CMAKE_EXE_LINKER_FLAGS "/LTCG")
	set(CMAKE_SHARED_LINKER_FLAGS "/DEBUG:FULL")
endif()


set(CMAKE_EXE_LINKER_FLAGS_FINAL "${CMAKE_EXE_LINKER_FLAGS}")
set(CMAKE_SHARED_LINKER_FLAGS_FINAL "${CMAKE_SHARED_LINKER_FLAGS}")
set(CMAKE_STATIC_LINKER_FLAGS_FINAL "${CMAKE_STATIC_LINKER_FLAGS}")

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
if(MSVC AND NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /MP")
endif()

# warning exceptions
if (MSVC)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-function")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-multichar")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-switch")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-attributes")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-missing-field-initializers")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-gnu-anonymous-struct")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-nested-anon-types")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-four-char-constants")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-const-variable")
    # workaround for "writing to an object of type ‘class Position’ with no trivial copy-assignment"
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-class-memaccess")
endif()

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# copy neural net file to build directory
set (DEFAULT_NEURAL_NET_FILE_NAME eval-55.pnn)
file(COPY ${CMAKE_SOURCE_DIR}/data/neuralNets/${DEFAULT_NEURAL_NET_FILE_NAME}
    DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

add_subdirectory("src")
