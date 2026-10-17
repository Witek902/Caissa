file(GLOB CHESS_BACKEND_SOURCES
     *.cpp
     syzygy/tbprobe.cpp
)

file(GLOB CHESS_BACKEND_HEADERS
     *.hpp
     syzygy/tbprobe.h
)

add_library(backend ${CHESS_BACKEND_SOURCES} ${CHESS_BACKEND_HEADERS})

target_compile_definitions(backend PRIVATE CAISSA_DEFAULT_EVALFILE="${DEFAULT_NEURAL_NET_FILE_NAME}")

set_property(TARGET backend PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
