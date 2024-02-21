file(GLOB CAISSA_BACKEND_SOURCES
     *.cpp
     syzygy/tbprobe.cpp
)

# exclude Unity.cpp from the list of sources
list(REMOVE_ITEM CAISSA_BACKEND_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/Unity.cpp)

file(GLOB CHESS_BACKEND_HEADERS
     *.hpp
     syzygy/tbprobe.h
)

add_library(backend ${CAISSA_BACKEND_SOURCES} ${CHESS_BACKEND_HEADERS})

set_property(TARGET backend PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
