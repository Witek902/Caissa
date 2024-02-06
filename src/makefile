EXE                = caissa
VERSION            = 1.17.5
EXE_NAME           = $(EXE)-$(VERSION)
DEFAULT_EVALFILE   = ../data/neuralNets/eval-33.pnn
EVALFILE           = $(DEFAULT_EVALFILE)
CC                 = clang++
SRC                = backend/*.cpp backend/syzygy/tbprobe.cpp frontend/*.cpp

WFLAGS = -Wall -Wno-unused-function -Wno-switch -Wno-attributes -Wno-missing-field-initializers -Wno-multichar
FLAGS = $(WFLAGS) -flto -std=c++20 -O3 -funroll-loops
LIBS =

# Detech compiler
COMPILER_VERSION := $(shell $(CC) --version)

ifneq '' '$(findstring g++,$(COMPILER_VERSION))'
	# using g++
	PGO_DATA     = "pgo"
	PGO_GENERATE = -fprofile-generate=$(PGO_DATA)
	PGO_USE      = -fprofile-use=$(PGO_DATA)
else ifneq '' '$(findstring clang,$(COMPILER_VERSION))'
	# using clang
	PGO_DATA     = *.profraw *.profdata
	PGO_MERGE    = llvm-profdata merge -output=caissa.profdata *.profraw
	PGO_GENERATE = -fprofile-instr-generate
	PGO_USE      = -fprofile-instr-use=caissa.profdata
endif

ifeq ($(OS), Windows_NT)
	# Windows
	EXT = .exe
	BENCH = $(EXE)$(EXT) bench 15 > nul 2>&1
	CLEAN = rmdir /s /q $(PGO_DATA)
	LIBS += -pthread
ifneq '' '$(findstring g++,$(COMPILER_VERSION))'
	# using g++
	FLAGS += -s -static-libgcc -static-libstdc++
else ifneq '' '$(findstring clang,$(COMPILER_VERSION))'
	# using clang
	FLAGS += -D_CRT_SECURE_NO_WARNINGS -fuse-ld=lld
endif
else
	# Linux
	EXT =
	BENCH = ./$(EXE) bench 15 > /dev/null 2>&1
	CLEAN = $(RM) -rf $(PGO_DATA)
	LIBS += -lpthread
endif

COMMONFLAGS   = $(FLAGS) $(LIBS) -DCONFIGURATION_FINAL -DNDEBUG -DCAISSA_VERSION=\"$(VERSION)\" -DCAISSA_EVALFILE=\"$(EVALFILE)\"

SSE2FLAGS     = $(COMMONFLAGS) -DUSE_SSE -DUSE_SSE2
SSE4FLAGS     = $(SSE2FLAGS) -DUSE_SSE4 -DUSE_POPCNT
AVX2FLAGS     = $(SSE4FLAGS) -DUSE_AVX2
BMI2FLAGS     = $(AVX2FLAGS) -DUSE_BMI2
AVX512FLAGS   = $(AVX2FLAGS) -mavx512f -mavx512bw -mavx512dq -DUSE_AVX512

# for OpenBench
ob:
	$(CC)   $(SRC) -march=native $(AVX2FLAGS) -o $(EXE)$(EXT) $(PGO_GENERATE)
	$(BENCH)
	$(PGO_MERGE)
	$(CC)   $(SRC) -march=native $(AVX2FLAGS) -o $(EXE)$(EXT) $(PGO_USE)
	$(CLEAN)

bmi2:
	$(CC)   $(SRC) -march=core-avx2 $(BMI2FLAGS) -o $(EXE_NAME)-x64-bmi2$(EXT)
avx2:
	$(CC)   $(SRC) -march=core-avx2 $(AVX2FLAGS) -o $(EXE_NAME)-x64-avx2$(EXT)
sse4:
	$(CC)   $(SRC) -march=core2 $(SSE4FLAGS) -o $(EXE_NAME)-x64-sse4-popcnt$(EXT)
sse2:
	$(CC)   $(SRC) -march=core2 $(SSE2FLAGS) -o $(EXE_NAME)-x64-sse2$(EXT)
avx512:
	$(CC)   $(SRC) -march=core-avx2 $(AVX512FLAGS) -o $(EXE_NAME)-x64-avx512$(EXT)
legacy:
	$(CC)   $(SRC) -march=core2 $(SSE2FLAGS) -o $(EXE_NAME)-x64-legacy$(EXT)

# Profile-guided optimization builds
bmi2_pgo:
	$(CC)   $(SRC) -march=core-avx2 $(BMI2FLAGS) -o $(EXE)$(EXT) $(PGO_GENERATE)
	$(BENCH)
	$(PGO_MERGE)
	$(CC)   $(SRC) -march=core-avx2 $(BMI2FLAGS) -o $(EXE)$(EXT) $(PGO_USE)
	$(CLEAN)
avx2_pgo:
	$(CC)   $(SRC) -march=core-avx2 $(AVX2FLAGS) -o $(EXE)$(EXT) $(PGO_GENERATE)
	$(BENCH)
	$(PGO_MERGE)
	$(CC)   $(SRC) -march=core-avx2 $(AVX2FLAGS) -o $(EXE)$(EXT) $(PGO_USE)
	$(CLEAN)

release:
	make -j sse2 sse4 avx2 bmi2 avx512
