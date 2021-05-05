#include <iostream>
#include "Position.hpp"
#include "Move.hpp"
#include "Search.hpp"
#include "UCI.hpp"

int main()
{
    InitBitboards(); 
    InitZobristHash();

    bool uciLoopResult = false;
    {
        UniversalChessInterface uci;
        uciLoopResult = uci.Loop();
    }

    return uciLoopResult ? 0 : 1;
}
