#include <iostream>
#include "Position.hpp"
#include "Move.hpp"
#include "Search.hpp"
#include "Evaluate.hpp"
#include "UCI.hpp"

int main(int argc, const char* argv[])
{
    InitBitboards(); 
    InitZobristHash();
    LoadNeuralNetwork("network.dat");

    UniversalChessInterface uci(argc, argv);
    uci.Loop();

    return 0;
}
