# Caissa Chess Engine
UCI command-line chess engine written in C++. Currently compiles only for Windows, Linux version is on the way.

Estimated ELO: **3100** (10+1 time control, single threaded, with NNUE)

## Features

#### General
* UCI protocol
* Neural network evaluation
* Syzygy endgame tablebases support

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
  * 736&rarr;512&rarr;32&rarr;64&rarr;1 layers architecture
  * effectively updated first layer, AVX2 accelerated
  * absolute piece coordinates (no symmetry, no king-relative features)
* Endgame evaluation
* Simple classic evaluation function based on Piece Square Tables
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
* Heuristics based on approximate move count left
* Reducing search time for singular root moves

#### Misc
* Large Pages Support for Transposition Table
* Magic Bitboards
* Analysis Mode (search full PV lines and disable any time and depth limits)


## Modules

The projects comprises folowing modules:
  * _backend_ (library) - engine's core
  * _frontend_ (executable) - UCI wrapper for the backend
  * _utils_ (executable) - various utilities, such as unit tests, neural network trainer, self-play data generator, etc.
  
## TODO

  * More platforms support
