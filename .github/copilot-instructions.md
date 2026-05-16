# Caissa Chess Engine – Copilot Instructions

Caissa is a strong UCI-compatible chess engine written in C++20 with a custom NNUE-style neural network evaluator. It targets rated play at 3600+ ELO and supports standard chess, FRC (Chess960), and DFRC.

## Build & Test

### Linux (CMake – recommended)
```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..   # or Final / Debug
make -j$(nproc)
```

Build types:
- `Final` – production: no asserts, maximum optimizations
- `Release` – development: asserts on, optimizations on
- `Debug` – development: asserts on, optimizations off

Architecture (default is `x64-bmi2` on x86-64):
```bash
cmake -DTARGET_ARCH=x64-avx512 -DCMAKE_BUILD_TYPE=Final ..
# Other options: x64-bmi2, x64-avx2, x64-sse4-popcnt, x64-legacy, aarch64-neon, aarch64
```

### Linux (Makefile – quick, builds AVX2/BMI2)
```bash
cd src && make -j$(nproc)
```

### Windows
Run `GenerateVisualStudioSolution.bat`, then open `build_<arch>/caissa.sln` in Visual Studio 2022.

### Running tests
```bash
# Unit tests (from build directory)
bin/utils unittest

# Smoke test / benchmark
bin/caissa "bench" "quit"

# Performance tests
bin/utils perftest [paths...]
```

### Neural network file
Neural network files (`.pnn`) are **not stored in this repository**. They are hosted in [Witek902/Caissa-Nets](https://github.com/Witek902/Caissa-Nets) as GitHub Release assets.

- **CMake**: automatically downloads the net to `data/neuralNets/` at configure time and copies it to the build output directory. No manual steps needed.
- **Makefile**: automatically downloads the net to `data/neuralNets/` via `curl` before compilation (required for INCBIN embedding). No manual steps needed.
- **`data/neuralNets/*.pnn` is in `.gitignore`** — never commit net files to this repo.

To update to a new net version, change `DEFAULT_NEURAL_NET_FILE_NAME` / `NET_VERSION` in `CMakeLists.txt`, `DEFAULT_EVALFILE` in `src/makefile`, and `c_DefaultEvalFile` in `src/backend/Evaluate.cpp`.

---

## Architecture

The project has three CMake targets:

| Target | Type | Location | Purpose |
|---|---|---|---|
| `backend` | static library | `src/backend/` | Engine core: search, eval, move gen, position |
| `caissa` (frontend) | executable | `src/frontend/` | UCI protocol wrapper |
| `utils` | executable | `src/utils/` | Trainer, self-play, unit tests, perf tests |

`utils` depends on `backend`; `frontend` also depends on `backend`.

### backend – key files
- `Search.cpp/.hpp` – negamax with alpha-beta, PVS, LMR, null-move pruning, singular extensions, correction history
- `Position.cpp/.hpp` – board state; `SidePosition` holds per-color bitboards + piece array
- `MoveGen.hpp`, `MoveList.hpp` – move generation; max 280 moves per position (`MaxAllowedMoves`)
- `PackedNeuralNetwork.cpp/.hpp` – runtime NNUE inference (manually SIMD-vectorized)
- `NeuralNetworkEvaluator.cpp/.hpp` – incremental accumulator updates; `AccumulatorCache`
- `TranspositionTable.cpp/.hpp` – shared TT with large-page support
- `Evaluate.cpp/.hpp` – static eval entry point; piece values, WLD model
- `MoveOrderer.cpp/.hpp`, `MovePicker.cpp/.hpp` – move ordering
- `TimeManager.cpp/.hpp` – time management
- `Endgame.cpp/.hpp` – special endgame routines
- `Tablebase.cpp/.hpp` – Syzygy (enabled by default) and Gaviota (opt-in) probing
- `Tuning.cpp/.hpp` – `DEFINE_PARAM` macro for exposing search params to UCI (requires `ENABLE_TUNING` build flag)

### Neural network (runtime)
Architecture: `(32×768 → 1024) × 2 → 1` (dual-perspective, one accumulator per king side, 32 king buckets, 768 = 12 piece types × 64 squares). The last layer has 8 variants selected by piece count. Network files use the `.pnn` extension.

### utils – subcommands (invoked as `bin/utils <command>`)
`unittest`, `perftest`, `selfplay`, `prepareTrainingData`, `plainTextToTrainingData`, `dumpGames`, `testNetwork`, `trainNetwork`, `trainCudaNetwork` (CUDA only), `validateEndgame`, `generateEndgamePositions`, `analyzeGames`

CUDA trainer is optional: auto-detected by CMake; compiled only when CUDA Toolkit is found (`USE_CUDA` define).

---

## Key Conventions

### Macros and inlining
- `INLINE` / `NO_INLINE` / `INLINE_LAMBDA` – cross-compiler inlining (`__forceinline` on MSVC, `__attribute__((always_inline))` on GCC/Clang)
- `ASSERT(x)` – fires in Debug/Release, compiled out in Final
- `VERIFY(x)` – same condition but always executes the expression; use where side-effects must survive Final builds
- `UNUSED(x)` – suppress unused-variable warnings

### Build configuration defines
- `CONFIGURATION_DEBUG` / `CONFIGURATION_RELEASE` / `CONFIGURATION_FINAL` – set by CMake per build type
- SIMD capability defines: `USE_SSE`, `USE_SSE2`, `USE_SSE4`, `USE_POPCNT`, `USE_AVX`, `USE_AVX2`, `USE_BMI2`, `USE_AVX512`, `USE_ARM_NEON`
- Architecture: `ARCHITECTURE_X64` or `ARCHITECTURE_AARCH64`
- Always guard SIMD code with the appropriate `#ifdef USE_*` / `#ifdef NN_USE_*` blocks

### Types and constants (defined in `Common.hpp` / `Score.hpp`)
- `ScoreType` = `int16_t`; scores are in centipawns
- `InfValue = 32767`, `CheckmateValue = 32000`, `TablebaseWinValue = 31000`, `KnownWinValue = 20000`
- `Color` = `uint8_t`; `White = 0`, `Black = 1`
- `PieceScore` = `TPieceScore<int16_t>` with `.mg` (midgame) and `.eg` (endgame) fields
- `MaxSearchDepth = 256`, `MaxAllowedMoves = 280`

### Move representation
- `Move` – full 32-bit struct (from/to squares, piece, flags); used during search
- `PackedMove` – 16-bit compact form (from/to + promotion only); used for TT storage and history tables
- Use `static_assert(sizeof(PackedMove) == 2)` guards to detect unintended size changes

### Tunable search parameters
Use the `DEFINE_PARAM(Name, Value, MinValue, MaxValue)` macro. Without `ENABLE_TUNING`, this becomes a `static constexpr int32_t`. With it, the parameter is registered in `g_TunableParameters` and exposed via UCI `setoption`.

Search parameters live in `Search.cpp`; move ordering parameters live in `MoveOrderer.cpp`.

---

## SPSA Parameter Tuning

Caissa uses SPSA (Simultaneous Perturbation Stochastic Approximation) for automated search parameter tuning. Tuning results are saved as HTML files in `scripts/tuning_html/`.

### Applying tuned results
- Tuned float values must be **rounded to the nearest integer** before updating `DEFINE_PARAM`.
- Only update the value (second argument); preserve `MinValue`/`MaxValue` unless explicitly extending bounds.
- Check whether any parameter converged at its min or max bound — this is a sign the range may be too narrow, or (with few games) noise is dominating.

### Tuning workflow
- **Tune in focused groups** of ~20–30 related parameters per run, not all parameters at once. Suggested groups:
  1. LMR parameters (`Search.cpp` lines 26–43)
  2. Pruning: NMP, RFP, futility, razoring, probcut
  3. Singular extensions
  4. History bonus/malus + continuation weights (`MoveOrderer.cpp`)
  5. Move ordering scores (MVV, threat bonuses, NodeCacheBonus)
- **600–800K games per group** is a good target. Fewer than ~4K games per parameter risks noisy convergence.
- A final "consolidation" run with all groups together (1M+ games) catches cross-group interactions.

### Semantic checks before applying
Some LMR parameters have non-obvious sign semantics — verify the usage in code before constraining bounds:
- `if (!isImproving) r += LmrQuietImproving` — a negative value means *less* reduction when not improving (defensible).
- `if (childNode.isInCheck) r -= LmrQuietInCheck` — a negative value means *more* reduction on check-giving moves (almost certainly wrong; clamp min to 0).
- `if (isBadCapture) r += LmrCaptureBad` — positive increases reduction for bad captures (expected).

### RTTI is disabled
`-fno-rtti` / `/GR-` everywhere. Do not use `dynamic_cast` or `typeid`.

### Compiler warnings
Warnings are errors (`-Werror` / `/WX`). Do not introduce new warnings. Approved warning suppressions are listed in the root `CMakeLists.txt`.

### Namespace usage
The neural network runtime lives in namespace `nn`. The trainer lives in `src/utils/net/`. The thread pool utility is in namespace `threadpool`.
