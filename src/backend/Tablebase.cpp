#include "Tablebase.hpp"
#include "Position.hpp"
#include "Move.hpp"

#ifdef USE_TABLE_BASES

#include "tablebase/tbprobe.h"

#include <iostream>

void LoadTablebase(const char* path)
{
    if (tb_init(path))
    {
        std::cout << "Tablebase loaded successfully. Size = " << TB_LARGEST << std::endl;
    }
    else
    {
        std::cout << "Failed to load tablebase" << std::endl;
    }
}

void UnloadTablebase()
{
    tb_free();
}

bool HasTablebases()
{
    return TB_LARGEST > 0u;
}

static Piece TranslatePieceType(uint32_t tbPromotes)
{
    switch (tbPromotes)
    {
    case TB_PROMOTES_QUEEN:     return Piece::Queen;
    case TB_PROMOTES_ROOK:      return Piece::Rook;
    case TB_PROMOTES_BISHOP:    return Piece::Bishop;
    case TB_PROMOTES_KNIGHT:    return Piece::Knight;
    }
    return Piece::None;
}

Move ProbeTablebase_Root(const Position& pos)
{
    if (!HasTablebases())
    {
        return Move::Invalid();
    }

    const uint32_t probeResult = tb_probe_root(
        pos.Whites().Occupied(),
        pos.Blacks().Occupied(),
        pos.Whites().king | pos.Blacks().king,
        pos.Whites().queens | pos.Blacks().queens,
        pos.Whites().rooks | pos.Blacks().rooks,
        pos.Whites().bishops | pos.Blacks().bishops,
        pos.Whites().knights | pos.Blacks().knights,
        pos.Whites().pawns | pos.Blacks().pawns,
        pos.GetHalfMoveCount(),
        0, // TODO castling rights
        pos.GetEnPassantSquare().mIndex,
        pos.GetSideToMove() == Color::White,
        nullptr);

    if (probeResult == TB_RESULT_FAILED)
    {
        return Move::Invalid();
    }

    PackedMove packedMove(Square(TB_GET_FROM(probeResult)), TB_GET_TO(probeResult), TranslatePieceType(TB_GET_PROMOTES(probeResult)));

    return pos.MoveFromPacked(packedMove);
}

#else // !USE_TABLE_BASES

bool HasTablebases()
{
    return false;
}

void LoadTablebase(const char*) { }
void UnloadTablebase() { }

Move ProbeTablebase_Root(const Position&)
{
    return Move::Invalid();
}

#endif // USE_TABLE_BASES
