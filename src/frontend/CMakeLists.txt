file(GLOB CHESS_SOURCES *.cpp)

add_executable(chess ${CHESS_SOURCES})

add_dependencies(chess backend)

target_link_libraries(chess backend)

set_target_properties(chess PROPERTIES LINK_FLAGS "-pthread")
