file(GLOB CHESS_BACKEND_SOURCES
     *.cpp
     syzygy/tbprobe.cpp
)

file(GLOB CHESS_BACKEND_HEADERS
     *.hpp
     syzygy/tbprobe.h
)

# Vendored Gaviota tablebase probing code (third-party C): built without warnings
set(GAVIOTA_SOURCES
     gaviota/gtb-probe.c
     gaviota/gtb-dec.c
     gaviota/gtb-att.c
     gaviota/sysport/sysport.c
     gaviota/compression/wrap.c
     gaviota/compression/huffman/hzip.c
     gaviota/compression/lzma/LzmaEnc.c
     gaviota/compression/lzma/LzmaDec.c
     gaviota/compression/lzma/Alloc.c
     gaviota/compression/lzma/LzFind.c
     gaviota/compression/lzma/Lzma86Enc.c
     gaviota/compression/lzma/Lzma86Dec.c
     gaviota/compression/lzma/Bra86.c
     gaviota/compression/zlib/zcompress.c
     gaviota/compression/zlib/uncompr.c
     gaviota/compression/zlib/inflate.c
     gaviota/compression/zlib/deflate.c
     gaviota/compression/zlib/adler32.c
     gaviota/compression/zlib/crc32.c
     gaviota/compression/zlib/infback.c
     gaviota/compression/zlib/inffast.c
     gaviota/compression/zlib/inftrees.c
     gaviota/compression/zlib/trees.c
     gaviota/compression/zlib/zutil.c
     gaviota/compression/liblzf/lzf_c.c
     gaviota/compression/liblzf/lzf_d.c
)

set(GAVIOTA_INCLUDE_DIRS
     ${CMAKE_CURRENT_SOURCE_DIR}/gaviota/sysport
     ${CMAKE_CURRENT_SOURCE_DIR}/gaviota/compression
     ${CMAKE_CURRENT_SOURCE_DIR}/gaviota/compression/liblzf
     ${CMAKE_CURRENT_SOURCE_DIR}/gaviota/compression/zlib
     ${CMAKE_CURRENT_SOURCE_DIR}/gaviota/compression/lzma
     ${CMAKE_CURRENT_SOURCE_DIR}/gaviota/compression/huffman
)

set(GAVIOTA_DEFINES Z_PREFIX)
if (MINGW)
    list(APPEND GAVIOTA_DEFINES MINGW)
endif()

# no LTO for these files either: link-time optimization would re-run the warnings with the link flags (-Werror)
if (MSVC)
    set(GAVIOTA_COMPILE_OPTIONS /W0)
else()
    set(GAVIOTA_COMPILE_OPTIONS -w -fno-lto)
endif()

set_source_files_properties(${GAVIOTA_SOURCES} PROPERTIES
    INCLUDE_DIRECTORIES "${GAVIOTA_INCLUDE_DIRS}"
    COMPILE_DEFINITIONS "${GAVIOTA_DEFINES}"
    COMPILE_OPTIONS "${GAVIOTA_COMPILE_OPTIONS}"
)

add_library(backend ${CHESS_BACKEND_SOURCES} ${CHESS_BACKEND_HEADERS} ${GAVIOTA_SOURCES})

target_compile_definitions(backend PUBLIC USE_GAVIOTA_TABLEBASES)

set_property(TARGET backend PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
