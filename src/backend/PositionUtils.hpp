#pragma once

#include "Bitboard.hpp"

#include <random>

#pragma pack(push, 1)
struct PackedPosition
{
    Bitboard occupied;          // bitboard of occupied squares
    uint16_t moveCount;
    uint8_t sideToMove : 1;     // 0 - white, 1 - black
    uint8_t halfMoveCount : 7;
    uint8_t castlingRights : 4;
    uint8_t enPassantFile : 4;
    uint8_t piecesData[16];     // 4 bits per occupied square
};
#pragma pack(pop)

bool PackPosition(const Position& inPos, PackedPosition& outPos);
bool UnpackPosition(const PackedPosition& inPos, Position& outPos);

void GenerateRandomPosition(std::mt19937& randomGenerator, const MaterialKey& material, Position& outPosition);

// generate random starting position for Transcendental Chess variant
void GenerateTranscendentalChessPosition(std::mt19937& randomGenerator, Position& outPosition);
