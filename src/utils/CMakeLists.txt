file(GLOB CHESS_UTILS_SOURCES *.cpp)

add_executable(utils ${CHESS_UTILS_SOURCES})

add_dependencies(utils backend)

target_link_directories(utils PRIVATE ${OUTPUT_DIRECTORY})
target_link_libraries(utils backend)

set_target_properties(utils PROPERTIES LINK_FLAGS "-pthread")
