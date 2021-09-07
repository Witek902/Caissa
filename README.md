# Chess
UCI command-line chess engine written in C++. Currently compiles only for Windows.

## Example
```
position fen 1r1bk2r/2R2ppp/p3p3/1b2P2q/4QP2/4N3/1B4PP/3R2K1 w k - 0 1
go depth 12
info depth 1 seldepth 5 time 0 score cp 44 nodes 334 pv g2g4
info depth 2 seldepth 7 time 0 score cp 0 nodes 692 pv g2g4 h5h3
info depth 3 seldepth 9 time 0 score cp -25 nodes 4093 pv c7c2 e8g8 b2d4
info depth 4 seldepth 12 time 2 score cp -43 nodes 9616 pv c7c1 e8g8 b2a3 f8e8
info depth 5 seldepth 12 time 2 score cp -42 nodes 10220 pv c7c1 d8b6 b2d4 e8g8 d4b6
info depth 6 seldepth 14 time 3 score cp -33 nodes 16167 pv c7c1 d8b6 d1d6 e8g8 b2d4 b6d4
info depth 7 seldepth 14 time 10 score cp -28 nodes 41015 pv c7c1 d8b6 b2a3 h5g4 a3c5 b6c5 c1c5
info depth 8 seldepth 17 time 29 score cp -19 nodes 123948 pv c7c1 d8e7 f4f5 b5e2 e4c6 e8f8 d1d2 e6f5 e3f5
info depth 9 seldepth 18 time 33 score cp -25 nodes 161117 pv c7c1 d8e7 f4f5 h5e2 b2d4 e6f5 e4f5 b5d7 f5d3
info depth 10 seldepth 19 time 109 score cp -54 nodes 488852 pv c7c1 d8b6 b2d4 b6d4 d1d4 e8g8
info depth 11 seldepth 21 time 117 score cp -57 nodes 574409 pv c7c1 d8b6 b2d4 b6d4 e4d4 e8g8 c1c7 b5e2 d1c1 f8d8 d4e4
info depth 12 seldepth 21 time 305 score cp -64 nodes 1434136 pv c7c1 d8b6 b2d4 b6d4 d1d4 e8g8 e3c4 b5c4 d4c4 b8b2
bestmove c7c1
```

## Features

  * UCI protocol
  * Alpha-Beta prunning
  * Principal Variation Search
  * Transposition Table
  * MVV/LVA
  * Killer/history heuristics
  * Iterative Deepening
  * Aspiration Window
  * Late Move Reduction
  * Multi-PV search
  * Null Move Prunning
  * Magic Bitboards
  * Syzygy endgame tables support

## Modules

The projects comprises folowing modules:
  * _backend_ (library) - engine's core
  * _frontend_ (executable) - UCI wrapper for the backend
  * _tests_ (executable) - unit tests and search tests
  
## TODO

  * Better evaluation function
  * More platforms support
  * Time management
  * Parallel search
