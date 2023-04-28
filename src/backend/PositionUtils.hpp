#pragma once

#include "Bitboard.hpp"
#include "Material.hpp"

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
bool UnpackPosition(const PackedPosition& inPos, Position& outPos, bool computeHash = true);

struct RandomPosDesc
{
    MaterialKey materialKey;
    // bitboards restricting piece placement
    Bitboard allowedWhiteKing = Bitboard::Full();
    Bitboard allowedWhitePawns = Bitboard::Full();
    Bitboard allowedWhiteKnights = Bitboard::Full();
    Bitboard allowedWhiteBishops = Bitboard::Full();
    Bitboard allowedWhiteRooks = Bitboard::Full();
    Bitboard allowedWhiteQueens = Bitboard::Full();
    Bitboard allowedBlackKing = Bitboard::Full();
    Bitboard allowedBlackPawns = Bitboard::Full();
    Bitboard allowedBlackKnights = Bitboard::Full();
    Bitboard allowedBlackBishops = Bitboard::Full();
    Bitboard allowedBlackRooks = Bitboard::Full();
    Bitboard allowedBlackQueens = Bitboard::Full();
};

void GenerateRandomPosition(std::mt19937& randomGenerator, const RandomPosDesc& desc, Position& outPosition);
