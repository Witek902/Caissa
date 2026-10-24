file(GLOB CAISSA_SOURCES *.cpp)

add_executable(caissa ${CAISSA_SOURCES})

set_property(TARGET caissa PROPERTY INTERPROCEDURAL_OPTIMIZATION_FINAL TRUE)

if (MSVC)
	target_compile_options(caissa PRIVATE /EHs-c-)
else()
	target_compile_options(caissa PRIVATE -fno-exceptions)
endif()

add_dependencies(caissa backend)

target_link_libraries(caissa PRIVATE backend)

if (NOT MSVC)
	find_package(Threads REQUIRED)
	target_link_libraries(caissa PRIVATE Threads::Threads)

	if (NUMA_FOUND)
		target_link_libraries(caissa PRIVATE NUMA::numa)
	endif()
endif()
