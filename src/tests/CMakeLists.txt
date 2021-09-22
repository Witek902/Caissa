file(GLOB CHESS_TESTS_SOURCES *.cpp)

add_executable(tests ${CHESS_TESTS_SOURCES})

add_dependencies(tests backend)

target_link_directories(tests PRIVATE ${OUTPUT_DIRECTORY})
target_link_libraries(tests backend)

set_target_properties(tests PROPERTIES LINK_FLAGS "-pthread")
