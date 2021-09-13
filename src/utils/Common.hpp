#pragma once

#include <inttypes.h>

struct PositionEntry
{
    uint64_t whiteKing;
    uint64_t whitePawns;
    uint64_t whiteKnights;
    uint64_t whiteBishops;
    uint64_t whiteRooks;
    uint64_t whiteQueens;

    uint64_t blackKing;
    uint64_t blackPawns;
    uint64_t blackKnights;
    uint64_t blackBishops;
    uint64_t blackRooks;
    uint64_t blackQueens;

    uint8_t sideToMove : 1;
    uint8_t whiteCastlingRights : 2;
    uint8_t blackCastlingRights : 2;

    int32_t eval;
    int32_t gameResult;
    uint16_t moveNumber;
    uint16_t totalMovesInGame;
};


void SelfPlay();

bool Train();

bool TrainEndgame();