file(GLOB CHESS_BACKEND_SOURCES
     *.cpp
     syzygy/tbprobe.cpp
)

file(GLOB CHESS_BACKEND_HEADERS
     *.hpp
     syzygy/tbprobe.h
)

add_library(backend ${CHESS_BACKEND_SOURCES} ${CHESS_BACKEND_HEADERS})
