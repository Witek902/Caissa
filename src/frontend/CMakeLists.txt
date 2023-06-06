file(GLOB CAISSA_SOURCES *.cpp)

add_executable(caissa ${CAISSA_SOURCES})

set_property(TARGET caissa PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)


add_dependencies(caissa backend)

target_link_libraries(caissa backend)

if (NOT MSVC)
	set_target_properties(caissa PROPERTIES LINK_FLAGS "-pthread")
endif()
