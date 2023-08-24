CC  = g++
SRC = backend/*.cpp backend/syzygy/tbprobe.cpp frontend/*.cpp
EXE = Caissa

ifeq ($(OS), Windows_NT)
	EXT = .exe
else
	EXT = 
endif


WFLAGS = -Wall -Wno-unused-function -Wno-switch -Wno-attributes -Wno-missing-field-initializers -Wno-multichar
RFLAGS = $(WFLAGS) -flto -std=c++20 -O3 -funroll-loops
LIBS =

ifeq ($(EXT), .exe)
	RFLAGS += -static -static-libgcc -static-libstdc++
	LIBS += -pthread
else
	LIBS += -lpthread
endif

ifneq ($(findstring g++, $(CC)),)
	PGO_GENERATE = -fprofile-generate
	PGO_USE      = -fprofile-use
else ifneq ($(findstring clang++, $(CC)),)
	PGO_MERGE    = llvm-profdata merge -output=caissa.profdata *.profraw
	PGO_GENERATE = -fprofile-instr-generate
	PGO_USE      = -fprofile-instr-use=caissa.profdata
endif

COMMONFLAGS   = -DCONFIGURATION_FINAL

NATIVEFLAGS   = $(COMMONFLAGS) -mno-avx512f -march=native
SSE2FLAGS     = $(COMMONFLAGS) -march=core2 -DUSE_SSE -DUSE_SSE2
AVX2FLAGS     = $(COMMONFLAGS) -march=core-avx2 -DUSE_SSE -DUSE_SSE2 -DUSE_SSE4 -DUSE_POPCNT -DUSE_AVX2 -DUSE_BMI2
AVX512FLAGS   = $(COMMONFLAGS) -march=core-avx2 -mavx512f -mavx512bw -mavx512dq -DUSE_SSE2 -DUSE_SSE4 -DUSE_POPCNT -DUSE_AVX2 -DUSE_BMI2 -DUSE_AVX512


native:
	$(CC)   $(SRC) $(RFLAGS) $(LIBS) $(NATIVEFLAGS) -o $(EXE)$(EXT)
sse2:
	$(CC)   $(SRC) $(RFLAGS) $(LIBS) $(SSE2FLAGS) -o $(EXE)-sse2$(EXT)
avx2:
	$(CC)   $(SRC) $(RFLAGS) $(LIBS) $(AVX2FLAGS) -o $(EXE)-avx2$(EXT)
avx512:
	$(CC)   $(SRC) $(RFLAGS) $(LIBS) $(AVX512FLAGS) -o $(EXE)-avx512$(EXT)
pgo:
	$(CC)   $(SRC) $(RFLAGS) $(LIBS) $(AVX2FLAGS) -o $(EXE)$(EXT) $(PGO_GENERATE)
	./$(EXE)$(EXT) "bench 15" "quit"
	$(PGO_MERGE)
	$(CC)   $(SRC) $(RFLAGS) $(LIBS) $(AVX2FLAGS) -o $(EXE)$(EXT) $(PGO_USE)

release:
	make sse2
	make avx2
	make avx512
