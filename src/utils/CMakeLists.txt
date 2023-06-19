file(
    GLOB_RECURSE CAISSA_UTILS_SOURCES
    *.cpp *.hpp *.c *.h
)

foreach(_source IN ITEMS ${CAISSA_UTILS_SOURCES})
    get_filename_component(_source_path "${_source}" PATH)
    string(REPLACE "${CMAKE_CURRENT_SOURCE_DIR}" "" _group_path "${_source_path}")
    string(REPLACE "/" "\\" _group_path "${_group_path}")
    source_group("${_group_path}" FILES "${_source}")
endforeach()

add_executable(utils ${CAISSA_UTILS_SOURCES})

set_property(TARGET utils PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)

add_dependencies(utils backend)

target_link_libraries(utils backend)

if (NOT MSVC)
	set_target_properties(utils PROPERTIES LINK_FLAGS "-pthread")
endif()
