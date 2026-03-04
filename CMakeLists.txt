cmake_minimum_required(VERSION 3.15)

project(caissa LANGUAGES C CXX)


option(USE_NUMA "Enable NUMA support via libnuma" ON)

if (USE_NUMA AND UNIX AND NOT APPLE)
    # first try to find libnuma via pkg-config
    find_package(PkgConfig QUIET)
    if (PkgConfig_FOUND)
        pkg_check_modules(NUMA QUIET numa)
    endif()
  
    # fall back to manual search
    if (NOT NUMA_FOUND)
        find_path(NUMA_INCLUDE_DIR NAMES numa.h)
        find_library(NUMA_LIBRARY NAMES numa)
    
        if (NUMA_INCLUDE_DIR AND NUMA_LIBRARY)
            set(NUMA_FOUND TRUE)
            set(NUMA_INCLUDE_DIRS "${NUMA_INCLUDE_DIR}")
            set(NUMA_LIBRARIES "${NUMA_LIBRARY}")
        endif()
    endif()
  
    if (NUMA_FOUND)
		if (NOT TARGET NUMA::numa)
			add_library(NUMA::numa INTERFACE IMPORTED)
			target_include_directories(NUMA::numa INTERFACE ${NUMA_INCLUDE_DIRS})
			target_link_libraries(NUMA::numa INTERFACE ${NUMA_LIBRARIES})
		endif()
        message(STATUS "libnuma found: ${NUMA_LIBRARIES} (includes: ${NUMA_INCLUDE_DIRS})")
        add_definitions(-DUSE_LIBNUMA)
    else()
        message(STATUS "libnuma not found - NUMA support disabled")
    endif()
endif()


# enable IPO (interprocedural optimization) if supported
include(CheckIPOSupported)
check_ipo_supported(RESULT IPO_SUPPORTED OUTPUT IPO_ERROR)
if (IPO_SUPPORTED)
    message(STATUS "IPO supported by this toolchain")
else()
    message(STATUS "IPO not supported: ${IPO_ERROR}")
endif()


# set supported configurations
set(CMAKE_CONFIGURATION_TYPES "Debug;Release;Final" CACHE STRING "" FORCE)

IF(NOT CMAKE_BUILD_TYPE)
   SET(CMAKE_BUILD_TYPE Final
       CACHE STRING "Choose one of build types: Debug Release Final."
       FORCE)
ENDIF(NOT CMAKE_BUILD_TYPE)
message(STATUS "Current build type: ${CMAKE_BUILD_TYPE}")

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
    elseif (CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
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

message(STATUS "Current architecture: ${TARGET_ARCH}")


# set C/C++ standards
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
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

# linker flags
if (MSVC)
	add_link_options(/DEBUG:FULL /OPT:ICF /LTCG)
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
    add_compile_options(/W4 /WX /GR- /MP)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
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
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")
endif()

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# copy neural net file to build directory
set (DEFAULT_NEURAL_NET_FILE_NAME eval-71.pnn)
file(COPY ${CMAKE_SOURCE_DIR}/data/neuralNets/${DEFAULT_NEURAL_NET_FILE_NAME}
    DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

add_subdirectory("src")
