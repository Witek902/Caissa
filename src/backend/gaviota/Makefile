INCLUDE = -Isysport/ \
	-Icompression/ \
	-Icompression/liblzf/ \
	-Icompression/zlib/ \
	-Icompression/lzma/ \
	-Icompression/huffman/

DEFAULT_CC = /usr/bin/gcc
DEFAULT_DEFINE = -DZ_PREFIX
#DEFAULT_ARCHFLAGS = -m64
DEFAULT_CFLAGS = -Wall -Wextra -fPIC $(INCLUDE) $(DEFAULT_DEFINE) \
	$(DEFAULT_ARCHFLAGS)
PREFIX = /usr/local
OPTFLAGS = -fast -msse -DNDEBUG
DEBUGFLAGS = -O0 -g -DDEBUG
PGO1FLAGS = $(OPTFLAGS) -fprofile-generate
PGO2FLAGS = $(OPTFLAGS) -fprofile-use

SRCFILES := gtb-probe.c gtb-dec.c gtb-att.c sysport/sysport.c \
	compression/wrap.c compression/huffman/hzip.c \
	compression/lzma/LzmaEnc.c compression/lzma/LzmaDec.c \
	compression/lzma/Alloc.c compression/lzma/LzFind.c \
	compression/lzma/Lzma86Enc.c compression/lzma/Lzma86Dec.c \
	compression/lzma/Bra86.c compression/zlib/zcompress.c \
	compression/zlib/uncompr.c compression/zlib/inflate.c \
	compression/zlib/deflate.c compression/zlib/adler32.c \
	compression/zlib/crc32.c compression/zlib/infback.c \
	compression/zlib/inffast.c compression/zlib/inftrees.c \
	compression/zlib/trees.c compression/zlib/zutil.c \
	compression/liblzf/lzf_c.c compression/liblzf/lzf_d.c
OBJFILES := $(patsubst %.c,%.o,$(SRCFILES))
PROFFILES := $(SRCFILES:.c=.gcno) $(SRCFILES:.c=.gcda)
LIBNAME := libgtb.a
SONAME :=libgtb.so
SOVERSION := 1.0.1
SOMAJORVERSION := 1


.PHONY: all clean
.DEFAULT_GOAL := all

all:
	$(MAKE) $(LIBNAME) \
		CC='$(DEFAULT_CC)' \
		ARCHFLAGS='$(DEFAULT_ARCHFLAGS)' \
		DEFINE='$(DEFAULT_DEFINE)' \
		CFLAGS='$(OPPTFLAGS) $(DEFAULT_CFLAGS)'
	$(MAKE) $(SONAME) \
		CC='$(DEFAULT_CC)' \
		ARCHFLAGS='$(DEFAULT_ARCHFLAGS)' \
		DEFINE='$(DEFAULT_DEFINE)' \
		CFLAGS='$(OPPTFLAGS) $(DEFAULT_CFLAGS)'

$(LIBNAME): $(OBJFILES)
	$(AR) rcs $@ $(OBJFILES)

$(SONAME): $(OBJFILES)   
	$(CC) -shared $(OBJFILES) -Wl,-soname=$(SONAME).$(SOMAJORVERSION) -o $(SONAME).$(SOVERSION)

opt:
	$(MAKE) $(LIBNAME) \
		CC='$(DEFAULT_CC)' \
		ARCHFLAGS='$(DEFAULT_ARCHFLAGS)' \
		DEFINE='$(DEFAULT_DEFINE)' \
		CFLAGS='$(OPTFLAGS) $(DEFAULT_CFLAGS)' \
		LDFLAGS='$(LDFLAGS)'

debug:
	$(MAKE) $(LIBNAME) \
		CC='$(DEFAULT_CC)' \
		ARCHFLAGS='$(DEFAULT_ARCHFLAGS)' \
		DEFINE='$(DEFAULT_DEFINE)' \
		CFLAGS='$(DEBUGFLAGS) $(DEFAULT_CFLAGS)' \
		LDFLAGS='$(LDFLAGS)'

pgo-start:
	$(MAKE) $(LIBNAME) \
		CC='$(DEFAULT_CC)' \
		ARCHFLAGS='$(DEFAULT_ARCHFLAGS)' \
		DEFINE='$(DEFAULT_DEFINE)' \
		CFLAGS='$(PGO1FLAGS) $(DEFAULT_CFLAGS)' \
		LDFLAGS='$(LDFLAGS) -fprofile-generate'

pgo-finish:
	$(MAKE) $(LIBNAME) \
		CC='$(DEFAULT_CC)' \
		ARCHFLAGS='$(DEFAULT_ARCHFLAGS)' \
		DEFINE='$(DEFAULT_DEFINE)' \
		CFLAGS='$(PGO2FLAGS) $(DEFAULT_CFLAGS)' \
		LDFLAGS='$(LDFLAGS) -fprofile-generate'

clean:
	$(RM) -f $(OBJFILES) $(LIBNAME) $(SONAME)

pgo-clean:
	$(RM) -f $(PROFFILES)

install:
	install -m 755 -o root -g root $(LIBNAME) $(SONAME).$(SOVERSION) $(PREFIX)/lib
	ln -sf $(SONAME).$(SOMAJORVERSION) $(PREFIX)/lib/$(SONAME)
	install -m 644 -o root -g root gtb-probe.h $(PREFIX)/include
	ldconfig

.depend:
	$(CC) -MM $(DEFAULT_CFLAGS) $(SRCFILES) > $@

include .depend

