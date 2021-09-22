file(GLOB CHESS_BACKEND_SOURCES *.cpp nnue-probe/*.cpp)

add_library(backend ${CHESS_BACKEND_SOURCES})
