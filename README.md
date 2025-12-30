# Caissa Chess Engine

[![C++ Standard](https://img.shields.io/badge/C++-20-blue.svg)](https://en.cppreference.com/w/cpp/20)
[![Linux Build Status](https://github.com/Witek902/Caissa/workflows/Linux/badge.svg)](https://github.com/Witek902/Caissa/actions/workflows/linux.yml)
[![GitHub License](https://img.shields.io/github/license/Witek902/Caissa?logo=github)](https://github.com/Witek902/Caissa/blob/master/LICENSE)
[![GitHub Release](https://img.shields.io/github/v/release/Witek902/Caissa?logo=github)](https://github.com/Witek902/Caissa/releases/latest)

![ArtImage](https://user-images.githubusercontent.com/5882734/193368109-abce432b-85e9-4f11-bb3c-57fd3d27db22.jpg?raw=true)
<p style='text-align: right;'><em>(image generated with DALL·E 2)</em></p>

## Overview

**Caissa** is a strong, UCI-compatible chess engine written from scratch in C++ since early 2021. It features a custom neural network evaluation system trained on over 17 billion self-play positions, achieving ratings of **3600+ ELO** on major chess engine rating lists, placing it at around top-10 spot.

The engine is optimized for:
- **Regular Chess** - Standard chess rules
- **FRC (Fischer Random Chess)** - Chess960 variant
- **DFRC (Double Fischer Random Chess)** - Extended FRC variant

## Table of Contents

- [Playing Strength](#playing-strength)
- [Features](#features)
- [Quick Start](#quick-start)
- [Compilation](#compilation)
  - [Prerequisites](#prerequisites)
  - [Linux](#linux)
  - [Windows](#windows)
- [Architecture Variants](#architecture-variants)
- [UCI Options](#uci-options)
- [History & Originality](#history--originality)
- [Project Structure](#project-structure)
- [License](#license)

## Playing Strength

Caissa consistently ranks among the top chess engines on major rating lists:

### CCRL (Computer Chess Rating Lists)
| List | Rating | Rank | Version | Notes |
|------|--------|------|---------|-------|
| [CCRL 40/2 FRC](https://www.computerchess.org.uk/ccrl/404FRC/) | **4022** | #6 | 1.23 | Fischer Random Chess |
| [CCRL Chess324](https://www.computerchess.org.uk/ccrl/Chess324/rating_list_all.html) | **3770** | #6 | 1.23 | Chess324 variant |
| [CCRL 40/15](https://www.computerchess.org.uk/ccrl/4040/) | **3622** | #9 | 1.23 | 4 CPU |
| [CCRL Blitz](https://www.computerchess.org.uk/ccrl/404/) | **3755** | #10 | 1.22 | 8 CPU |

### SPCC (Schachprogramm-Computer-Chess)
| List | Rating | Rank | Version |
|------|--------|------|---------|
| [SPCC UHO-Top15](https://www.sp-cc.de) | **3697** | #10 | Caissa 1.24 avx512 |

### IpMan Chess
| List | Rating | Rank | Version | Architecture |
|------|--------|------|---------|--------------|
| [10+1 (R9-7945HX)](https://ipmanchess.yolasite.com/r9-7945hx.php) | **3542** | #16 | 1.24 | AVX-512 |
| [10+1 (i9-7980XE)](https://ipmanchess.yolasite.com/i9-7980xe.php) | **3526** | #14 | 1.21 | AVX-512 |
| [10+1 (i9-13700H)](https://ipmanchess.yolasite.com/i7-13700h.php) | **3544** | #17 | 1.22 | AVX2-BMI2 |

### CEGT (Chess Engine Grand Tournament)
| List | Rating | Rank | Version |
|------|--------|------|---------|
| [CEGT 40/20](http://www.cegt.net/40_40%20Rating%20List/40_40%20SingleVersion/rangliste.html) | **3576** | #8 | 1.24 |
| [CEGT 40/4](http://www.cegt.net/40_4_Ratinglist/40_4_single/rangliste.html) | **3614** | #8 | 1.22 |
| [CEGT 5+3](http://www.cegt.net/5Plus3Rating/BestVersionsNEW/rangliste.html) | **3618** | #5 | 1.22 |

> **Note**: The rankings above may be outdated.

## Features

### General
- ✅ **UCI Protocol** - Full Universal Chess Interface support
- ✅ **Neural Network Evaluation** - Custom NNUE-style evaluation
- ✅ **Endgame Tablebases** - Syzygy and Gaviota support
- ✅ **Chess960 Support** - Fischer Random Chess (FRC) and Double FRC

### Search Algorithm
- ✅ **Negamax** with alpha-beta pruning
- ✅ **Iterative Deepening** with aspiration windows
- ✅ **Principal Variation Search (PVS)**
- ✅ **Quiescence Search** for tactical positions
- ✅ **Transposition Table** with large pages support
- ✅ **Multi-PV Search** - Analyze multiple lines simultaneously
- ✅ **Multithreaded Search** - Parallel search with shared TT

### Neural Network Evaluation
- **Architecture**: (11×768→1024)×2→1
- **Incremental Updates** - Efficiently updated first layer
- **Vectorized Code** - Manual SIMD optimization for:
  - AVX-512 (fastest)
  - AVX2
  - SSE2
  - ARM NEON
- **Activation**: Clipped-ReLU
- **Variants**: 8 variants of last layer weights (piece count dependent)
- **Features**: Absolute piece coordinates with horizontal symmetry, 11 king buckets
- **Special Endgame Routines** - Enhanced endgame evaluation

### Neural Network Trainer
- **Custom CPU-based Trainer** using Adam algorithm
- **Highly Optimized** - Exploits AVX instructions, multithreading, and network sparsity
- **Self-Play Training** - Trained on 17+ billion positions from self-generated games
- **Progressive Training** - Older games purged, networks trained on latest engine versions

### Performance Optimizations
- **Magic Bitboards** - Efficient move generation
- **Large Pages** - Transposition table uses large pages for better performance
- **Node Caching** - Evaluation result caching
- **Accumulator Caching** - Neural network accumulator caching
- **Ultra-Fast** - Outstanding performance at ultra-short time controls (sub-second games)

## Quick Start

### Using Pre-built Binaries

1. Download the appropriate executable from the [Releases](https://github.com/Witek902/Caissa/releases) page
2. Choose the version matching your CPU:
   - **AVX-512**: Latest Intel Xeon/AMD EPYC (fastest)
   - **BMI2**: Most modern CPUs (recommended)
   - **AVX2**: Older CPUs with AVX2 support
   - **POPCNT**: Older CPUs with SSE4.2
   - **Legacy**: Very old x64 CPUs
3. Copy the neural network file (`.pnn`) from `data/neuralNets/` to the same directory as the executable
4. Run the engine with any UCI-compatible chess GUI

### Running from Source

See the [Compilation](#compilation) section below for detailed build instructions.

## Compilation

### Prerequisites

- **C++ Compiler** with C++20 support:
  - GCC 10+ or Clang 12+ (Linux)
  - Visual Studio 2022 (Windows)
- **CMake** 3.15 or later
- **Make** (Linux) or Visual Studio (Windows)

### Linux

#### Using Makefile (Quick Build)

```bash
cd src
make -j$(nproc)
```

> **Note**: This compiles the default AVX2/BMI2 version.

#### Using CMake (Recommended)

```bash
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Final ..
make -j$(nproc)
```

**Build Configurations:**
- `Final` - Production build, no asserts, maximum optimizations
- `Release` - Development build with asserts, optimizations enabled
- `Debug` - Development build with asserts, optimizations disabled

**Architecture Selection:**

To build for a specific architecture, set the `TARGET_ARCH` variable:

```bash
# AVX-512 (requires AVX-512 support)
cmake -DTARGET_ARCH=x64-avx512 -DCMAKE_BUILD_TYPE=Final ..
# BMI2 (recommended for modern CPUs)
cmake -DTARGET_ARCH=x64-bmi2 -DCMAKE_BUILD_TYPE=Final ..
# AVX2
cmake -DTARGET_ARCH=x64-avx2 -DCMAKE_BUILD_TYPE=Final ..
# SSE4-POPCNT
cmake -DTARGET_ARCH=x64-sse4-popcnt -DCMAKE_BUILD_TYPE=Final ..
# Legacy (fallback)
cmake -DTARGET_ARCH=x64-legacy -DCMAKE_BUILD_TYPE=Final ..
```

### Windows

1. Run `GenerateVisualStudioSolution.bat` to generate the Visual Studio solution
2. Open `build_<arch>/caissa.sln` in Visual Studio 2022
3. Select the desired configuration (Debug/Release/Final)
4. Build the solution (Ctrl+Shift+B)

> **Note**: Visual Studio 2022 is the only tested version. CMake directly in Visual Studio has not been tested.

### Post-Compilation

After compilation, copy the appropriate neural network file from `data/neuralNets/` to:
- **Linux**: `build/bin/`
- **Windows**: `build\bin\x64\<Configuration>\`

## Architecture Variants

| Variant | CPU Requirements | Performance | Recommended For |
|---------|-----------------|-------------|-----------------|
| **AVX-512** | AVX-512 instruction set | Fastest | Latest Intel Xeon, AMD EPYC |
| **BMI2** | AVX2 + BMI2 | Fast | Most modern CPUs (2015+) |
| **AVX2** | AVX2 instruction set | Fast | Intel Haswell, AMD Ryzen |
| **POPCNT** | SSE4.2 + POPCNT | Moderate | Older CPUs (2008-2014) |
| **Legacy** | x64 only | Slowest | Very old x64 CPUs |

> **Tip**: If unsure, try BMI2 first. It's supported by most modern CPUs and offers excellent performance.

## UCI Options

The engine supports the following UCI options:

### Search Options
- **Hash** (int) - Transposition table size in megabytes
- **Threads** (int) - Number of search threads
- **MultiPV** (int) - Number of principal variation lines to search
- **Ponder** (bool) - Enable pondering mode

### Time Management
- **MoveOverhead** (int) - Move overhead in milliseconds (increase if engine loses time)

### Evaluation
- **EvalFile** (string) - Path to neural network evaluation file (`.pnn`)
- **EvalRandomization** (int) - Evaluation randomization range (weakens engine, introduces non-determinism)

### Tablebases
- **SyzygyPath** (string) - Semicolon-separated paths to Syzygy tablebases
- **SyzygyProbeLimit** (int) - Maximum number of pieces for tablebase probing

### Display Options
- **UCI_AnalyseMode** (bool) - Analysis mode (full PV lines, no depth constraints)
- **UCI_Chess960** (bool) - Enable Chess960 mode (castling as "king captures rook")
- **UCI_ShowWDL** (bool) - Show win/draw/loss probabilities with evaluation
- **UseSAN** (bool) - Use Standard Algebraic Notation (FIDE standard)
- **ColorConsoleOutput** (bool) - Enable colored console output

## History & Originality

Caissa has been written **from the ground up** since early 2021. The development journey:

1. **Early Versions** - Used simple PeSTO evaluation
2. **Version 0.6** - Temporarily used Stockfish NNUE
3. **Version 0.7+** - Custom neural network evaluation system

### Neural Network Development

The engine's neural network has evolved significantly:
- **Initial Network**: Based on Stockfish's architecture, trained on a few million positions
- **Current Network** (v1.24+): Trained on **17+ billion positions** from self-play
- **Progressive Training**: Older games are purged, ensuring networks are trained only on the latest engine versions

### Key Components

- **Runtime Evaluation**: [`PackedNeuralNetwork.cpp`](https://github.com/Witek902/Caissa/blob/master/src/backend/PackedNeuralNetwork.cpp)
  - Inspired by [nnue.md](https://github.com/glinscott/nnue-pytorch/blob/master/docs/nnue.md)
  - Highly optimized with manual SIMD vectorization

- **Network Trainer**: [`NetworkTrainer.cpp`](https://github.com/Witek902/Caissa/blob/master/src/utils/NetworkTrainer.cpp), [`NeuralNetwork.cpp`](https://github.com/Witek902/Caissa/blob/master/src/utils/net/Network.cpp)
  - Written completely from scratch
  - CPU-based, heavily optimized with AVX and multithreading
  - Exploits network sparsity for performance

- **Self-Play Generator**: [`SelfPlay.cpp`](https://github.com/Witek902/Caissa/blob/master/src/utils/SelfPlay.cpp)
  - Generates games with fixed nodes/depth
  - Custom binary format for efficient storage
  - Uses Stefan's Pohl [UHO books](https://www.sp-cc.de/downloads--links.htm) or DFRC openings

## Project Structure

The project is organized into three main modules:

```
src/
├── backend/     # Core engine library
│   ├── Search.*           # Search algorithms
│   ├── Position.*         # Position representation
│   ├── MoveGen.*          # Move generation
│   ├── PackedNeuralNetwork.*  # Neural network evaluation
│   ├── TranspositionTable.*   # Position caching
│   └── ...
│
├── frontend/    # UCI interface executable
│   ├── Main.cpp           # Entry point
│   └── UCI.*              # UCI protocol implementation
│
└── utils/       # Development and training tools
    ├── NetworkTrainer.*    # Neural network training
    ├── SelfPlay.*          # Self-play game generation
    ├── Tests.*            # Unit tests
    └── ...
```

### Module Descriptions

- **backend** (library) - Engine core: search, evaluation, move generation, position management
- **frontend** (executable) - UCI wrapper providing command-line interface
- **utils** (executable) - Utilities: network trainer, self-play generator, unit tests, performance tests

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Author**: Michał Witanowski  
**Started**: Early 2021  
**Language**: C++20  
**License**: MIT
