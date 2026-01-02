EXE                = caissa
VERSION            = 1.24.9
EXE_NAME           = $(EXE)-$(VERSION)
DEFAULT_EVALFILE   = ../data/neuralNets/eval-67.pnn
EVALFILE           = $(DEFAULT_EVALFILE)
CC                 = g++
SRC                = $(wildcard backend/*.cpp) $(wildcard frontend/*.cpp) backend/syzygy/tbprobe.cpp
OBJ                = $(SRC:.cpp=.o)

ifeq ($(OS), Windows_NT)
	EXT = .exe
else
	EXT = 
endif

WFLAGS = -Wall -Wno-unused-function -Wno-switch -Wno-attributes -Wno-missing-field-initializers -Wno-multichar
FLAGS = $(WFLAGS) -s -flto -flto-partition=one -std=c++20 -O3 -funroll-loops
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
	BENCH = $(EXE)$(EXT) bench 15 > nul 2>&1
	CLEAN = rmdir /s /q $(PGO_DIR)
	CLEAN_OBJS = del /s /q $(subst /,\,$(OBJ))
else
	BENCH = ./$(EXE) bench 15 > /dev/null 2>&1
	CLEAN = $(RM) -rf $(PGO_DIR)
	CLEAN_OBJS = $(RM) -f $(OBJ)
endif

COMMONFLAGS   = $(FLAGS) $(LIBS) -DCONFIGURATION_FINAL -DNDEBUG -DCAISSA_VERSION=\"$(VERSION)\" -DCAISSA_EVALFILE=\"$(EVALFILE)\"

SSE2FLAGS     = $(COMMONFLAGS) -DUSE_SSE -DUSE_SSE2
SSE4FLAGS     = $(SSE2FLAGS) -DUSE_SSE4 -DUSE_POPCNT
AVX2FLAGS     = $(SSE4FLAGS) -DUSE_AVX2
BMI2FLAGS     = $(AVX2FLAGS) -DUSE_BMI2
AVX512FLAGS   = $(AVX2FLAGS) -mavx512f -mavx512bw -mavx512dq -DUSE_AVX512

%.o: %.cpp
	$(CC) -march=native $(AVX2FLAGS) -c $< -o $@
%.o: %.c
	$(CC) -march=native $(AVX2FLAGS) -c $< -o $@

# for OpenBench
ob: $(OBJ)
	$(CC) $(OBJ) -march=native $(AVX2FLAGS) -o $(EXE)$(EXT)
	$(CLEAN_OBJS)

bmi2:
	$(CC) $(SRC) -march=core-avx2 $(BMI2FLAGS) -o $(EXE_NAME)-x64-bmi2$(EXT)
avx2:
	$(CC) $(SRC) -march=core-avx2 $(AVX2FLAGS) -o $(EXE_NAME)-x64-avx2$(EXT)
sse4:
	$(CC) $(SRC) -march=core2 $(SSE4FLAGS) -o $(EXE_NAME)-x64-sse4-popcnt$(EXT)
sse2:
	$(CC) $(SRC) -march=core2 $(SSE2FLAGS) -o $(EXE_NAME)-x64-sse2$(EXT)
avx512:
	$(CC) $(SRC) -march=skylake-avx512 $(AVX512FLAGS) -o $(EXE_NAME)-x64-avx512$(EXT)
legacy:
	$(CC) $(SRC) -march=core2 $(SSE2FLAGS) -o $(EXE_NAME)-x64-legacy$(EXT)

# Profile-guided optimization builds
bmi2_pgo:
	$(CC) $(SRC) -march=core-avx2 $(BMI2FLAGS) -o $(EXE)$(EXT) $(PGO_GENERATE)
	$(BENCH)
	$(PGO_MERGE)
	$(CC) $(SRC) -march=core-avx2 $(BMI2FLAGS) -o $(EXE)$(EXT) $(PGO_USE)
	$(CLEAN)
avx2_pgo:
	$(CC) $(SRC) -march=core-avx2 $(AVX2FLAGS) -o $(EXE)$(EXT) $(PGO_GENERATE)
	$(BENCH)
	$(PGO_MERGE)
	$(CC) $(SRC) -march=core-avx2 $(AVX2FLAGS) -o $(EXE)$(EXT) $(PGO_USE)
	$(CLEAN)
avx512_pgo:
	$(CC) $(SRC) -march=skylake-avx512 $(AVX512FLAGS) -o $(EXE)$(EXT) $(PGO_GENERATE)
	$(BENCH)
	$(PGO_MERGE)
	$(CC) $(SRC) -march=core-avx2 $(AVX512FLAGS) -o $(EXE)$(EXT) $(PGO_USE)
	$(CLEAN)

release:
	make -j sse2 sse4 avx2 bmi2 avx512
