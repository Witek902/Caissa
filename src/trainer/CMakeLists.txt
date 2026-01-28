enable_language(CUDA)

file(
    GLOB_RECURSE CAISSA_TRAINER_SOURCES
    *.cpp *.hpp *.c *.h *.cu
)

# organize source files in IDE
foreach(_source IN ITEMS ${CAISSA_TRAINER_SOURCES})
    get_filename_component(_source_path "${_source}" PATH)
    string(REPLACE "${CMAKE_CURRENT_SOURCE_DIR}" "" _group_path "${_source_path}")
    string(REPLACE "/" "\\" _group_path "${_group_path}")
    source_group("${_group_path}" FILES "${_source}")
endforeach()

add_executable(trainer ${CAISSA_TRAINER_SOURCES})

set_property(TARGET trainer PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
set_property(TARGET trainer PROPERTY CUDA_ARCHITECTURES native)
set_property(TARGET trainer PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# Link backend
add_dependencies(trainer backend)

# CUDA toolkit
find_package(CUDAToolkit REQUIRED)

target_link_libraries(trainer PRIVATE backend CUDA::cudart)

# Standards
set_target_properties(trainer PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CUDA_STANDARD 17
  CUDA_STANDARD_REQUIRED ON
)

if (NOT MSVC)
	set_target_properties(trainer PROPERTIES LINK_FLAGS "-pthread")
endif()
