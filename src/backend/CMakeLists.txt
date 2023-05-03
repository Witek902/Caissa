file(GLOB CHESS_BACKEND_SOURCES
     *.cpp
     syzygy/tbprobe.cpp
)

add_library(backend ${CHESS_BACKEND_SOURCES})
