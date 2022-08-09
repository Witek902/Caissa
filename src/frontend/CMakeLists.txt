file(GLOB CAISSA_SOURCES *.cpp)

add_executable(caissa ${CAISSA_SOURCES})

add_dependencies(caissa backend)

target_link_libraries(caissa backend)

set_target_properties(caissa PROPERTIES LINK_FLAGS "-pthread")
