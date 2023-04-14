file(GLOB CHESS_BACKEND_SOURCES *.cpp syzygy/tbprobe.c)

add_library(backend ${CHESS_BACKEND_SOURCES})
