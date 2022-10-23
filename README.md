# Caissa Chess Engine

[![LinuxBuildStatus](https://github.com/Witek902/Caissa/workflows/Linux/badge.svg)](https://github.com/Witek902/Caissa/actions/workflows/linux.yml)

![ArtImage](https://user-images.githubusercontent.com/5882734/193368109-abce432b-85e9-4f11-bb3c-57fd3d27db22.jpg?raw=true)

## Overview

UCI command-line chess engine written in C++ from scratch. In development since beginning of 2021.

### Playing strength

* CCRL 40/15 Score: **3288** (#38) (version 1.1)
* CCRL 2+1 Score: **3352** (#42) (version 1.0)
* CCRL FRC 40/2 Score: **3414** (#17) (version 1.1)
* CEGT 40/4 Score: **3211** (#34) (version 0.9)
* CEGT 40/20 Score: **3245** (#40) (version 1.0)

### Supported UCI options

* **Hash** (int) Sets transposition table size in megabytes.
* **MultiPV** (int) Specifies number of searched and printed PV lines.
* **MoveOverhead** (int) Sets move overhead in miliseconds. Should be increased if the engine is loosing on time.
* **Threads** (int) Sets number of threads used for searching.
* **Ponder** (bool) Enables pondering.
* **EvalFile** (string) Neural network evaluation file.
* **SyzygyPath** (string) Semicolon-separated list of paths to Syzygy endgame tablebases.
* **GaviotaTbPath** (string) Path to Gaviota endgame tablebases.
* **GaviotaTbCache** (int) Gaviota cache size in megabytes.
* **UCI_AnalyseMode** (bool) Enables analysis mode: search full PV lines and disable any depth constrains.
* **UseSAN** (bool) Enables short algebraic notation output (FIDE standard) instead of default long algebraic notation.
* **ColorConsoleOutput** (bool) Enables colorful console output for better readibility.


### Provided EXE versions

* **AVX2/BMI2** Fastest, requires a x64 CPU with AVX2 and BMI2 instruction set support.
* **POPCNT** Slower, requires a x64 CPU with SSE4 and POPCNT instruction set support.
* **Legacy** Slowest, requires any x64 CPU.


## Features

#### General
* UCI protocol
* Neural network evaluation
* Syzygy and Gaviota endgame tablebases support
* Chess960 (Fischer Random) support

#### Search Algorithm
* Negamax with alpha-beta pruning
* Iterative Deepening with Aspiration Windows
* Principal Variation Search (PVS)
* Zero Window Search
* Quiescence Search
* Transposition Table
* Multi-PV search
* Multithreaded search via shared transposition table

#### Evaluation
* Custom neural network
  * 704&rarr;512&rarr;16&rarr;32&rarr;1 layers architecture
  * effectively updated first layer, AVX2/SSE accelerated
  * clipped-ReLU activation function
  * absolute piece coordinates with horizontal symmetry, no king-relative features
  * custom CPU-based trainer using Adagrad SGD algorithm
  * dedicated network for endgame positions (5 or less pieces on board)
* Endgame evaluation
* Simple classic evaluation function based purely on Piece Square Tables
* NN and PSQT trained on data generated during self-play matches

#### Selectivity
* Null Move Reductions
* Late Move Reductions & Prunning
* Futility Pruning
* Mate Distance Prunning
* Singular Move Extensions
* Upcoming repetition detection

#### Move Ordering
* MVV/LVA
* Winning/Losing Captures (Static Exchange Evaluation)
* Killer/History/Counter/Followup Move Heuristic
* Sacrifice penalty / threat bonus

#### Time Management
* Heuristics based on approximate move count left and score fluctuations.
* Reducing search time for singular root moves

#### Misc
* Large Pages Support for Transposition Table
* Magic Bitboards

## Modules

The projects comprises folowing modules:
  * _backend_ (library) - engine's core
  * _frontend_ (executable) - UCI wrapper for the backend
  * _utils_ (executable) - various utilities, such as unit tests, neural network trainer, self-play data generator, etc.
  
## TODO

  * Better neural network architecture
  * Better classic evaluation (king safety, mobility, pawns, etc.)
  * Search tuning and improvement
  * More platforms support
