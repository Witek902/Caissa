# Caissa Chess Engine

[![C++ Standard](https://img.shields.io/badge/C++-20-blue.svg)](https://en.cppreference.com/w/cpp/20)
[![Linux Build Status](https://github.com/Witek902/Caissa/workflows/Linux/badge.svg)](https://github.com/Witek902/Caissa/actions/workflows/linux.yml)
[![GitHub License](https://img.shields.io/github/license/Witek902/Caissa?logo=github)](https://github.com/Witek902/Caissa/blob/master/LICENSE)
[![GitHub Release](https://img.shields.io/github/v/release/Witek902/Caissa?logo=github)](https://github.com/Witek902/Caissa/releases/latest)

![ArtImage](https://user-images.githubusercontent.com/5882734/193368109-abce432b-85e9-4f11-bb3c-57fd3d27db22.jpg?raw=true)
<p align="right"><em>(image generated with DALL·E 2)</em></p>

## Overview

**Caissa** is a strong, UCI-compatible chess engine written from scratch in C++ since early 2021 by Michał Witanowski, released under the MIT license. It uses a custom neural network evaluation trained on 20.5+ billion self-play positions and is rated **3600+ ELO** on major chess engine rating lists, placing it at around top-16 spot.

Supported variants:
- **Regular Chess** — standard chess rules
- **FRC (Fischer Random Chess)** — Chess960
- **DFRC (Double Fischer Random Chess)**

---

## Playing Strength

Caissa consistently ranks among the top chess engines on major rating lists:

### CCRL (Computer Chess Rating Lists)
| List | Rating | Rank | Version | Notes |
|------|--------|------|---------|-------|
| [CCRL 40/2 FRC](https://www.computerchess.org.uk/ccrl/404FRC/) | **4037** | #12 | 1.26 | Fischer Random Chess |
| [CCRL Chess324](https://www.computerchess.org.uk/ccrl/Chess324/rating_list_all.html) | **3749** | #15 | 1.25 | Chess324 variant |
| [CCRL 40/15](https://www.computerchess.org.uk/ccrl/4040/) | **3633** | #11 | 1.26 | 4 CPU |
| [CCRL Blitz](https://www.computerchess.org.uk/ccrl/404/) | **3749** | #12 | 1.22 | 8 CPU |

### SPCC (Stefan Pohl Computer Chess)
| List | Rating | Rank | Version |
|------|--------|------|---------|
| [SPCC UHO-Top15](https://www.sp-cc.de) | **3749** | around #18 | Caissa 1.26 avx512 |

### IpMan Chess
| List | Rating | Rank | Version | Architecture |
|------|--------|------|---------|--------------|
| [10+1 (R9-7945HX)](https://ipmanchess.yolasite.com/r9-7945hx.php) | **3532** | #18 | 1.25 | AVX-512 |
| [10+1 (i9-13700H)](https://ipmanchess.yolasite.com/i7-13700h.php) | **3546** | #16 | 1.25 | AVX-512 |

### CEGT (Chess Engine Grand Tournament)
| List | Rating | Rank | Version |
|------|--------|------|---------|
| [CEGT 40/20](http://www.cegt.net/40_40%20Rating%20List/40_40%20SingleVersion/rangliste.html) | **3570** | #12 | 1.25 |
| [CEGT 40/4](http://www.cegt.net/40_4_Ratinglist/40_4_single/rangliste.html) | **3614** | #8 | 1.22 |
| [CEGT 5+3](http://www.cegt.net/5Plus3Rating/BestVersionsNEW/rangliste.html) | **3618** | #5 | 1.22 |

> **Note**: The rankings above may be outdated.

---

## Features

### General
- ✅ **UCI Protocol** — full Universal Chess Interface support
- ✅ **Neural Network Evaluation** — custom NNUE-style evaluation, see [Neural Network](#neural-network)
- ✅ **Endgame Tablebases** — Syzygy support (up to 7 pieces)
- ✅ **Chess960 Support** — Fischer Random Chess (FRC) and Double FRC

### Search Algorithm
- ✅ **Negamax** with alpha-beta pruning
- ✅ **Iterative Deepening** with aspiration windows
- ✅ **Principal Variation Search (PVS)**
- ✅ **Quiescence Search** for tactical positions
- ✅ **Transposition Table** with large pages support
- ✅ **Multi-PV Search** — analyze multiple lines simultaneously
- ✅ **Multithreaded Search** — parallel search with shared TT
- ✅ **Late Move Reductions (LMR)**
- ✅ **Null-Move Pruning**, **ProbCut**, **razoring**, **futility pruning**
- ✅ **Singular Extensions**
- ✅ **Correction History** — pawn and non-pawn correction tables improve static eval accuracy
- ✅ **Cuckoo Hashing** for fast repetition detection

### Performance Optimizations
- **Magic Bitboards** — efficient move generation
- **Large Pages** — transposition table uses large pages for better performance
- **Node Caching** — per-move node counts cached across iterations to improve move ordering
- **Accumulator Caching** — neural network accumulator cache
- **NUMA Support** — memory allocation and thread pinning respect NUMA topology on multi-socket systems (Linux, requires `libnuma`)
- **Ultra-Fast** — outstanding performance at ultra-short time controls (sub-second games)
- **Special Endgame Routines** — enhanced endgame evaluation

---

## Neural Network

### Architecture

`(32×768 → 1024) × 2 → 1` — dual-perspective (one accumulator per king side), 32 king buckets,
768 features per perspective (12 piece types × 64 squares).

- **Features**: absolute piece coordinates with horizontal symmetry
- **Activation**: Squared-Clipped-ReLU (SCReLU)
- **Output**: 8 variants of the last layer, selected by piece count
- **Incremental Updates** — efficiently updated first layer
- **Vectorized Code** — hand-written SIMD for AVX-512, AVX2 (with optional VNNI), SSE4 and ARM NEON

### Training

- **Custom CUDA Trainer** written from scratch, using AdamW optimizer
- **Highly Optimized** — manual CUDA kernel optimizations for speed
- **Self-Play Data** — 20.5+ billion positions from self-generated games
- **Progressive Training** — older games are purged, so networks are trained only on data from the latest engine versions

### Network files

Network files use the `.pnn` extension and are **not stored in this repository** — they are hosted in [Witek902/Caissa-Nets](https://github.com/Witek902/Caissa-Nets) as release assets and downloaded automatically at build time (CMake at configure time, the Makefile via `curl`/`wget` before compiling). No manual download or copy step is needed. A different network can be loaded at runtime with the `EvalFile` UCI option.

---

## Quick Start

1. Download the executable for your CPU from the [Releases](https://github.com/Witek902/Caissa/releases) page — see [Architecture Variants](#architecture-variants) below. If unsure, use **BMI2**.
2. Run the engine in any UCI-compatible chess GUI.

To build instead, see [Compilation](#compilation).

## Architecture Variants

The same variant names are used for the release binaries and for the CMake `TARGET_ARCH` option.

| Variant | `TARGET_ARCH` | CPU Requirements | Recommended For |
|---------|---------------|------------------|-----------------|
| **AVX-512** | `x64-avx512` | AVX-512F + AVX-512BW | Latest Intel Xeon, AMD Zen 4/5 |
| **BMI2** | `x64-bmi2` | AVX2 + BMI2 | Most modern CPUs (2015+) — *default* |
| **AVX2** | `x64-avx2` | AVX2 | Intel Haswell, early AMD Ryzen |
| **POPCNT** | `x64-sse4-popcnt` | SSE4.2 + POPCNT | Older CPUs (2008–2014) |
| **Legacy** | `x64-legacy` | x64 + SSE2 only | Very old x64 CPUs |
| **NEON** | `aarch64-neon` | ARMv8-A + NEON | Modern ARM hardware |
| **AArch64** | `aarch64` | ARMv8-A | ARM without NEON intrinsics |

> **Note**: The Makefile additionally provides a `-march=native` default build, which is tuned for the host CPU.

## Compilation

### Prerequisites

- **C++ Compiler** with C++20 support:
  - GCC 10+ or Clang 12+ (Linux)
  - Visual Studio 2022 (Windows)
- **CMake** 3.24 or later
- **Make** (Linux) or Visual Studio (Windows)
- **curl** or **wget** — used by the Makefile to fetch the network file
- *Optional*: **CUDA Toolkit** — enables the CUDA network trainer in `utils` (auto-detected)
- *Optional*: **libnuma** (Linux) — enables NUMA-aware allocation and thread pinning for multi-socket systems

### Linux

#### Using CMake (recommended)

```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Final ..
make -j$(nproc)
```

**Build configurations:**
- `Final` — production build, no asserts, maximum optimizations
- `Release` — development build, asserts on, optimizations on
- `Debug` — development build, asserts on, optimizations off

**Architecture selection** — pass `TARGET_ARCH` with any name from the [Architecture Variants](#architecture-variants) table (defaults to `x64-bmi2` on x86-64, `aarch64-neon` on ARM):

```bash
cmake -DTARGET_ARCH=x64-avx512 -DCMAKE_BUILD_TYPE=Final ..
```

The binary is written to `build/bin/`, together with the neural network file.

#### Using the Makefile (quick build)

```bash
cd src
make -j$(nproc)
```

> **Note**: The default goal (`ob`, used by OpenBench) builds with `-march=native`, tuned for the host CPU. For a portable binary pick an explicit target: `bmi2`, `avx512`, `avx2`, `avx2-vnni`, `sse4`, `sse2`, `legacy`, `release` (builds all of them), or the PGO variants `bmi2_pgo`, `avx2_pgo`, `avx512_pgo`.
> Pass `EVALFILE=<path>` to build against a local network file instead of downloading one.

### Windows

1. Run `GenerateVisualStudioSolution.bat` to generate the Visual Studio solution
2. Open `build_<arch>/caissa.sln` in Visual Studio 2022
3. Select the desired configuration (Debug/Release/Final)
4. Build the solution (Ctrl+Shift+B)

The binary is written to `build_<arch>/bin/x64/<Configuration>/`.

> **Note**: Visual Studio 2022 is the only tested version. Using CMake directly from within
> Visual Studio has not been tested.

### ARM / AArch64

```bash
mkdir build && cd build
cmake -DTARGET_ARCH=aarch64-neon -DCMAKE_BUILD_TYPE=Final ..
make -j$(nproc)
```

Use `-DTARGET_ARCH=aarch64` for a build without NEON intrinsics.

---

## Custom Commands

In addition to the standard UCI protocol, the engine supports these non-standard commands, useful for development and debugging:

| Command | Description |
|---------|-------------|
| `bench [depth]` | Run the signature benchmark over a fixed position set (default depth 12). Also available as `benchmark` |
| `perft <depth>` | Count leaf nodes to a given depth (move generation test) |
| `eval` | Display evaluation of the current position, with WDL probabilities |
| `eval detailed [start\|stop\|reset]` | Print NNUE accumulator statistics, optionally gathered over a search |
| `print` | Pretty-print the current board |
| `scoremoves` | Show move ordering scores for the current position |
| `threats` | Show threat information for the current position |
| `ttinfo` | Print transposition table statistics |
| `ttprobe` | Probe the transposition table for the current position |
| `tbprobe` | Probe tablebases for the current position |
| `cacheprobe` | Probe the node cache for the current position |
| `help` | List all available commands |
| `printparams` | Print all tunable search/eval parameters (`ENABLE_TUNING` builds only) |
| `moveordererstats` | Print move orderer statistics (non-`Final` builds only) |

## UCI Options

### Search
- **Hash** (int) — transposition table size in megabytes
- **Threads** (int) — number of search threads
- **MultiPV** (int) — number of principal variation lines to search
- **Ponder** (bool) — enable pondering mode

### Time Management
- **MoveOverhead** (int) — move overhead in milliseconds (increase if the engine loses on time)

### Evaluation
- **EvalFile** (string) — path to the neural network evaluation file (`.pnn`)
- **EvalRandomization** (int) — evaluation randomization range (weakens the engine, introduces non-determinism)

### Tablebases
- **SyzygyPath** (string) — path to Syzygy tablebases (multiple paths separated by `;` on Windows, `:` elsewhere)
- **SyzygyProbeLimit** (int) — maximum number of pieces for tablebase probing

### Display
- **UCI_Chess960** (bool) — enable Chess960 mode (castling encoded as "king captures rook")
- **UCI_ShowWDL** (bool) — show win/draw/loss probabilities with the evaluation
- **UseSAN** (bool) — use Standard Algebraic Notation (FIDE standard)
- **ColorConsoleOutput** (bool) — enable colored console output

---

## History & Originality

Caissa has been written **from the ground up** since early 2021:

1. **Early versions** — simple PeSTO evaluation
2. **Version 0.6** — temporarily used Stockfish NNUE
3. **Version 0.7+** — custom neural network evaluation, initially based on Stockfish's architecture and trained on a few million positions, since grown into the architecture described in [Neural Network](#neural-network)

### Key Components

- **Runtime evaluation**: [`PackedNeuralNetwork.cpp`](src/backend/PackedNeuralNetwork.cpp)
  - Inspired by [nnue.md](https://github.com/glinscott/nnue-pytorch/blob/master/docs/nnue.md)
  - Highly optimized with manual SIMD vectorization
- **Network trainer**: [`CudaNetworkTrainer.cpp`](src/utils/CudaNetworkTrainer.cpp), [`CudaNetwork.cu`](src/utils/cudaTrainer/CudaNetwork.cu) — written completely from scratch
- **Self-play generator**: [`SelfPlay.cpp`](src/utils/SelfPlay.cpp)
  - Generates games with fixed nodes/depth
  - Custom binary format for efficient storage
  - Uses Stefan Pohl's [UHO books](https://www.sp-cc.de/downloads--links.htm) or DFRC openings

---

## Project Structure

```
src/
├── backend/     # Core engine library: search, evaluation, move generation, position
│   ├── Search.*                 # Search algorithms
│   ├── Position.*               # Position representation
│   ├── MoveGen.*                # Move generation
│   ├── PackedNeuralNetwork.*    # Neural network evaluation
│   ├── TranspositionTable.*     # Position caching
│   └── ...
│
├── frontend/    # UCI wrapper executable (caissa)
│   └── UCI.*                    # UCI protocol implementation and entry point
│
└── utils/       # Development and training tools executable (utils)
    ├── CudaNetworkTrainer.*     # Neural network training
    ├── cudaTrainer/             # CUDA kernels
    ├── SelfPlay.*               # Self-play game generation
    ├── Tests.*                  # Unit tests
    └── ...
```

The `utils` executable bundles the development tooling, invoked as `utils <command>`:
`unittest`, `selfplay`, `trainCudaNetwork` (CUDA builds only), `prepareTrainingData`, `plainTextToTrainingData`, `pgnToTrainingData`, and more.

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
