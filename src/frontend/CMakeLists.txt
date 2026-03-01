file(GLOB CAISSA_SOURCES *.cpp)

add_executable(caissa ${CAISSA_SOURCES})

set_property(TARGET caissa PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)

add_dependencies(caissa backend)

target_link_libraries(caissa PRIVATE backend)

if (NOT MSVC)
	find_package(Threads REQUIRED)
	target_link_libraries(caissa PRIVATE Threads::Threads)

	if (NUMA_FOUND)
		target_link_libraries(caissa PRIVATE NUMA::numa)
	endif()
endif()
