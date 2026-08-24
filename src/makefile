.DELETE_ON_ERROR:
.DEFAULT_GOAL := ob

EXE                = caissa
VERSION            := $(strip $(file < ../version.txt))
EXE_NAME           = $(EXE)-$(VERSION)
DEFAULT_EVALFILE   = ../data/neuralNets/eval-84-144B.pnn
EVALFILE           = $(DEFAULT_EVALFILE)
NET_VERSION        = $(basename $(notdir $(DEFAULT_EVALFILE)))
NET_DOWNLOAD_URL   = https://github.com/Witek902/Caissa-Nets/releases/download/$(NET_VERSION)/$(notdir $(DEFAULT_EVALFILE))
NET_TMP            = $(DEFAULT_EVALFILE).tmp
# IPv4 + HTTP/1.1: a broken IPv6 route stalls the connect, and old HTTP/2 stacks
# hit "stream not closed cleanly (PROTOCOL_ERROR)" on the GitHub CDN
NET_CURL           = curl -fsS -L -4 --http1.1 --retry 3 --retry-delay 2 --connect-timeout 15 -o $(NET_TMP) $(NET_DOWNLOAD_URL)
NET_CURL_ANY       = curl -fsS -L --retry 3 --retry-delay 2 --connect-timeout 15 -o $(NET_TMP) $(NET_DOWNLOAD_URL)
NET_WGET           = wget -q -O $(NET_TMP) $(NET_DOWNLOAD_URL)
CC                 = g++
SRC                = $(wildcard backend/*.cpp) $(wildcard frontend/*.cpp) backend/syzygy/tbprobe.cpp
OBJ                = $(SRC:.cpp=.o)

ifeq ($(OS), Windows_NT)
	EXT = .exe
else
	EXT = 
endif

WFLAGS = -Wall -Wno-unused-function -Wno-switch -Wno-attributes -Wno-missing-field-initializers -Wno-multichar
FLAGS = $(WFLAGS) -g1 -flto -flto-partition=one -std=c++20 -O3 -funroll-loops -fno-rtti -fno-exceptions -fomit-frame-pointer -fno-math-errno -fno-trapping-math -fno-stack-protector
LIBS =

ifeq ($(EXT), .exe)
	FLAGS += -static -static-libgcc -static-libstdc++
	LIBS += -pthread
else
	LIBS += -lpthread
endif

ifneq ($(findstring g++, $(CC)),)
	PGO_DIR = "pgo"
	PGO_GENERATE = -fprofile-generate=$(PGO_DIR)
	PGO_USE      = -fprofile-use=$(PGO_DIR)
else ifneq ($(findstring clang++, $(CC)),)
	PGO_MERGE    = llvm-profdata merge -output=caissa.profdata *.profraw
	PGO_GENERATE = -fprofile-instr-generate
	PGO_USE      = -fprofile-instr-use=caissa.profdata
endif

# Try to detect windows environment by seeing
# whether the shell filters out " or not.
ifeq ($(shell echo "test"), "test")
	BENCH = $(EXE)$(EXT) bench 16 > nul 2>&1
	CLEAN = rmdir /s /q $(PGO_DIR)
	CLEAN_OBJS = del /s /q $(subst /,\,$(OBJ))
	MKDIR_NET_DIR = if not exist $(subst /,\,$(dir $(DEFAULT_EVALFILE))) mkdir $(subst /,\,$(dir $(DEFAULT_EVALFILE)))
	MOVE_NET = move /y $(subst /,\,$(NET_TMP)) $(subst /,\,$(DEFAULT_EVALFILE)) > nul
else
	BENCH = ./$(EXE) bench 16 > /dev/null 2>&1
	CLEAN = $(RM) -rf $(PGO_DIR)
	CLEAN_OBJS = $(RM) -f $(OBJ)
	MKDIR_NET_DIR = mkdir -p $(dir $(DEFAULT_EVALFILE))
	MOVE_NET = mv -f $(NET_TMP) $(DEFAULT_EVALFILE)
endif

COMMONFLAGS   = $(FLAGS) $(LIBS) -DCONFIGURATION_FINAL -DNDEBUG -DCAISSA_VERSION=\"$(VERSION)\" -DCAISSA_EVALFILE=\"$(EVALFILE)\"

SSE2FLAGS     = $(COMMONFLAGS) -DUSE_SSE -DUSE_SSE2
SSE4FLAGS     = $(SSE2FLAGS) -DUSE_SSE4 -DUSE_POPCNT
AVX2FLAGS     = $(SSE4FLAGS) -DUSE_AVX2
AVX2VNNIFLAGS = $(AVX2FLAGS) -mavxvnni -DUSE_VNNI
BMI2FLAGS     = $(AVX2FLAGS) -DUSE_BMI2
AVX512FLAGS   = $(AVX2FLAGS) -mavx512f -mavx512bw -mavx512dq -DUSE_AVX512

# Native build: ask the compiler which ISA extensions -march=native enables on this host,
# so the feature defines match the CPU instead of being pinned to a fixed baseline.
ifeq ($(shell echo "test"), "test")
	NATIVE_MACROS := $(shell echo. | $(CC) -march=native -dM -E -x c++ - 2>nul)
else
	NATIVE_MACROS := $(shell $(CC) -march=native -dM -E -x c++ /dev/null 2>/dev/null)
endif

has-native = $(if $(findstring $(1),$(NATIVE_MACROS)),$(2))

ifeq ($(strip $(NATIVE_MACROS)),)
	# probe failed - fall back to the portable AVX2 baseline
	NATIVEFLAGS = $(AVX2FLAGS) -march=native -mtune=native
else
	NATIVEFLAGS = $(COMMONFLAGS) -march=native -mtune=native -DUSE_SSE -DUSE_SSE2 \
	              $(call has-native,__SSE4_1__,-DUSE_SSE4) \
	              $(call has-native,__POPCNT__,-DUSE_POPCNT) \
	              $(call has-native,__AVX__,-DUSE_AVX) \
	              $(call has-native,__AVX2__,-DUSE_AVX2) \
	              $(call has-native,__AVXVNNI__,-DUSE_VNNI) \
	              $(if $(findstring __AVX512DQ__,$(NATIVE_MACROS)),$(call has-native,__AVX512BW__,-DUSE_AVX512))
endif

%.o: %.cpp
	$(CC) $(NATIVEFLAGS) -c $< -o $@
%.o: %.c
	$(CC) $(NATIVEFLAGS) -c $< -o $@

# Evaluate.cpp embeds the net via INCBIN, so it must not be compiled before the download completes
backend/Evaluate.o: $(EVALFILE)

# for OpenBench
ob: $(EVALFILE) $(OBJ)
	$(CC) $(OBJ) $(NATIVEFLAGS) -o $(EXE)$(EXT)
	$(CLEAN_OBJS)

# download the default net file if not present (needed for embedding via INCBIN)
# pass EVALFILE=<path> to build against a local net instead, with no network access
# written via a temporary file, so an interrupted transfer never looks like a complete net
$(DEFAULT_EVALFILE):
	@echo Downloading neural network $(notdir $@) from $(NET_DOWNLOAD_URL)...
	@$(MKDIR_NET_DIR)
	@$(NET_CURL) || $(NET_CURL_ANY) || $(NET_WGET)
	@$(MOVE_NET)
	@echo Download complete: $@

bmi2: $(EVALFILE)
	$(CC) $(SRC) -march=core-avx2 $(BMI2FLAGS) -o $(EXE_NAME)-x64-bmi2$(EXT)
avx2: $(EVALFILE)
	$(CC) $(SRC) -march=core-avx2 $(AVX2FLAGS) -o $(EXE_NAME)-x64-avx2$(EXT)
avx2-vnni: $(EVALFILE)
	$(CC) $(SRC) -march=core-avx2 $(AVX2VNNIFLAGS) -o $(EXE_NAME)-x64-avx2-vnni$(EXT)
sse4: $(EVALFILE)
	$(CC) $(SRC) -march=core2 $(SSE4FLAGS) -o $(EXE_NAME)-x64-sse4-popcnt$(EXT)
sse2: $(EVALFILE)
	$(CC) $(SRC) -march=core2 $(SSE2FLAGS) -o $(EXE_NAME)-x64-sse2$(EXT)
avx512: $(EVALFILE)
	$(CC) $(SRC) -march=skylake-avx512 $(AVX512FLAGS) -o $(EXE_NAME)-x64-avx512$(EXT)
legacy: $(EVALFILE)
	$(CC) $(SRC) -march=core2 $(SSE2FLAGS) -o $(EXE_NAME)-x64-legacy$(EXT)

# Profile-guided optimization builds
bmi2_pgo: $(EVALFILE)
	$(CC) $(SRC) -march=core-avx2 $(BMI2FLAGS) -o $(EXE)$(EXT) $(PGO_GENERATE)
	$(BENCH)
	$(PGO_MERGE)
	$(CC) $(SRC) -march=core-avx2 $(BMI2FLAGS) -o $(EXE)$(EXT) $(PGO_USE)
	$(CLEAN)
avx2_pgo: $(EVALFILE)
	$(CC) $(SRC) -march=core-avx2 $(AVX2FLAGS) -o $(EXE)$(EXT) $(PGO_GENERATE)
	$(BENCH)
	$(PGO_MERGE)
	$(CC) $(SRC) -march=core-avx2 $(AVX2FLAGS) -o $(EXE)$(EXT) $(PGO_USE)
	$(CLEAN)
avx512_pgo: $(EVALFILE)
	$(CC) $(SRC) -march=skylake-avx512 $(AVX512FLAGS) -o $(EXE)$(EXT) $(PGO_GENERATE)
	$(BENCH)
	$(PGO_MERGE)
	$(CC) $(SRC) -march=core-avx2 $(AVX512FLAGS) -o $(EXE)$(EXT) $(PGO_USE)
	$(CLEAN)

release:
	make -j sse2 sse4 avx2 bmi2 avx512
