# What is it

nnue-probe is library for probing NNUE neural networks for chess.
The core nnue probing code is taken from [CFish](https://github.com/syzygy1/Cfish) and modified a bit.

# How to build

To compile

    make clean; make COMP=gcc 

Cross-compiling for windows from linux using mingw is possible by setting `COMP=win`

# Probing from python

    from __future__ import print_function
    from ctypes import *
    nnue = cdll.LoadLibrary("libnnueprobe.so")
    nnue.nnue_init(b"nn-04cf2b4ed1da.nnue")
    score = nnue.nnue_evaluate_fen(b"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    print("Score = ", score)

The result

    Loading NNUE : nn-04cf2b4ed1da.nnue
    NNUE loaded !
    Score =  42

