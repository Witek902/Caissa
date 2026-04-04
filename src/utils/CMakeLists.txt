# Optional CUDA support
find_package(CUDAToolkit QUIET)

if (CUDAToolkit_FOUND)
    enable_language(CUDA)
    message(STATUS "CUDA found: enabling CUDA trainer")
else()
    message(STATUS "CUDA not found: CUDA trainer disabled")
endif()

file(
    GLOB_RECURSE CAISSA_UTILS_SOURCES
    *.cpp *.hpp *.c *.h *.cu
)

# Exclude CUDA sources when CUDA is not available
if (NOT CUDAToolkit_FOUND)
    list(FILTER CAISSA_UTILS_SOURCES EXCLUDE REGEX ".*\\.cu$")
    list(FILTER CAISSA_UTILS_SOURCES EXCLUDE REGEX ".*/cudaTrainer/.*")
    list(FILTER CAISSA_UTILS_SOURCES EXCLUDE REGEX ".*CudaNetworkTrainer\\.cpp$")
endif()

foreach(_source IN ITEMS ${CAISSA_UTILS_SOURCES})
    get_filename_component(_source_path "${_source}" PATH)
    string(REPLACE "${CMAKE_CURRENT_SOURCE_DIR}" "" _group_path "${_source_path}")
    string(REPLACE "/" "\\" _group_path "${_group_path}")
    source_group("${_group_path}" FILES "${_source}")
endforeach()

add_executable(utils ${CAISSA_UTILS_SOURCES})

set_property(TARGET utils PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)

add_dependencies(utils backend)

target_link_libraries(utils PRIVATE backend)

if (NOT MSVC)
	find_package(Threads REQUIRED)
	target_link_libraries(utils PRIVATE Threads::Threads)

	if (NUMA_FOUND)
		target_link_libraries(utils PRIVATE NUMA::numa)
	endif()
endif()

if (CUDAToolkit_FOUND)
    target_compile_definitions(utils PRIVATE USE_CUDA)
    target_link_libraries(utils PRIVATE CUDA::cudart)
    set_property(TARGET utils PROPERTY CUDA_ARCHITECTURES native)
    set_property(TARGET utils PROPERTY CUDA_SEPARABLE_COMPILATION ON)
    set_target_properties(utils PROPERTIES
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
    )
endif()
