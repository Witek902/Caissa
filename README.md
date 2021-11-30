# Caissa Chess Engine
UCI command-line chess engine written in C++. Currently compiles only for Windows, Linux version is on the way.

Estimated ELO: **2800** (10+1 time control, single threaded, with NNUE)

## Features

#### General
* UCI protocol
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
* Stockfish NNUE support
* Endgame evaluation
* Simple classic evaluation function based on Piece Square Tables

#### Selectivity
* Null Move Reductions
* Late Move Reductions & Prunning
* Futility Pruning (both for alpha and beta)
* Mate Distance Prunning
* Singular Move Extensions

#### Move Ordering
* MVV/LVA
* Winning/Losing Captures (Static Exchange Evaluation)
* Killer/History/Counter/Followup Move Heuristic

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
  * _tests_ (executable) - unit tests and search tests
  * _utils_ (executable) - various utilities, such as neural network trainer
  
## TODO

  * Better evaluation function
  * Custom-trained neural network
  * More platforms support
