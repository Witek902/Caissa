EXE                = caissa
VERSION            = 1.13.4
EXE_NAME           = $(EXE)-$(VERSION)
DEFAULT_EVALFILE   = ../data/neuralNets/eval-21.pnn
EVALFILE           = $(DEFAULT_EVALFILE)
CC                 = g++
SRC                = backend/*.cpp backend/syzygy/tbprobe.cpp frontend/*.cpp

ifeq ($(OS), Windows_NT)
	EXT = .exe
else
	EXT = 
endif

WFLAGS = -Wall -Wno-unused-function -Wno-switch -Wno-attributes -Wno-missing-field-initializers -Wno-multichar
FLAGS = $(WFLAGS) -flto -std=c++20 -O3 -funroll-loops
LIBS =

ifeq ($(EXT), .exe)
	FLAGS += -static -static-libgcc -static-libstdc++
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

COMMONFLAGS   = $(FLAGS) $(LIBS) -DCONFIGURATION_FINAL  -DNDEBUG -DCAISSA_VERSION=\"$(VERSION)\" -DCAISSA_EVALFILE=\"$(EVALFILE)\"

SSE2FLAGS     = $(COMMONFLAGS) -DUSE_SSE -DUSE_SSE2
SSE4FLAGS     = $(SSE2FLAGS) -DUSE_SSE4 -DUSE_POPCNT
AVX2FLAGS     = $(SSE4FLAGS) -DUSE_AVX2 -DUSE_BMI2
AVX512FLAGS   = $(AVX2FLAGS) -mavx512f -mavx512bw -mavx512dq -DUSE_AVX512

avx2:
	$(CC)   $(SRC) -march=core-avx2 $(AVX2FLAGS) -o $(EXE_NAME)-x64-avx2-bmi2$(EXT)
sse4:
	$(CC)   $(SRC) -march=core2 $(SSE4FLAGS) -o $(EXE_NAME)-x64-sse4-popcnt$(EXT)
sse2:
	$(CC)   $(SRC) -march=core2 $(SSE2FLAGS) -o $(EXE_NAME)-x64-sse2$(EXT)
avx512:
	$(CC)   $(SRC) -march=core-avx2 $(AVX512FLAGS) -o $(EXE_NAME)-x64-avx512$(EXT)
pgo:
	$(CC)   $(SRC) $(AVX2FLAGS) -o $(EXE_NAME)-pgo$(EXT) $(PGO_GENERATE)
	./$(EXE_NAME)-pgo$(EXT) "bench 15" "quit"
	$(PGO_MERGE)
	$(CC)   $(SRC) $(AVX2FLAGS) -o $(EXE_NAME)-pgo$(EXT) $(PGO_USE)

release:
	make avx2
	make sse4
	make avx512
	make sse2
