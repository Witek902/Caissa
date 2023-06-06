file(GLOB CHESS_UTILS_SOURCES *.cpp minitrace/*.c)

add_executable(utils ${CHESS_UTILS_SOURCES})

set_property(TARGET utils PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)

add_dependencies(utils backend)

target_link_libraries(utils backend)

if (NOT MSVC)
	set_target_properties(utils PROPERTIES LINK_FLAGS "-pthread")
endif()
