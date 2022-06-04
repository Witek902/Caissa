#include "Tablebase.hpp"
#include "Position.hpp"
#include "Move.hpp"

#ifdef USE_TABLE_BASES

#include "tablebase/tbprobe.h"

#include <iostream>
#include <mutex>

static std::mutex g_tbMutex;

void LoadTablebase(const char* path)
{
    std::unique_lock lock(g_tbMutex);

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
    std::unique_lock lock(g_tbMutex);

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

bool ProbeTablebase_Root(const Position& pos, Move& outMove, uint32_t* outDistanceToZero, int32_t* outWDL)
{
    if (!HasTablebases() ||
        pos.GetNumPieces() > TB_LARGEST)
    {
        return false;
    }

    std::unique_lock lock(g_tbMutex);

    uint32_t castlingRights = 0;
    if (pos.GetWhitesCastlingRights() & CastlingRights_ShortCastleAllowed)  castlingRights |= TB_CASTLING_K;
    if (pos.GetWhitesCastlingRights() & CastlingRights_LongCastleAllowed)   castlingRights |= TB_CASTLING_Q;
    if (pos.GetBlacksCastlingRights() & CastlingRights_ShortCastleAllowed)  castlingRights |= TB_CASTLING_k;
    if (pos.GetBlacksCastlingRights() & CastlingRights_LongCastleAllowed)   castlingRights |= TB_CASTLING_q;

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
        castlingRights,
        pos.GetEnPassantSquare().IsValid() ? pos.GetEnPassantSquare().Index() : 0,
        pos.GetSideToMove() == Color::White,
        nullptr);

    if (probeResult == TB_RESULT_FAILED)
    {
        return false;
    }

    const uint32_t tbFrom = TB_GET_FROM(probeResult);
    const uint32_t tbTo = TB_GET_TO(probeResult);
    const uint32_t tbPromo = TB_GET_PROMOTES(probeResult);

    outMove = pos.MoveFromPacked(PackedMove(Square(tbFrom), Square(tbTo), TranslatePieceType(tbPromo)));

    if (!outMove.IsValid())
    {
        return false;
    }

    if (outDistanceToZero)
    {
        *outDistanceToZero = TB_GET_DTZ(probeResult);
    }

    if (outWDL)
    {
        *outWDL = 0;

        if (TB_GET_WDL(probeResult) == TB_WIN) *outWDL = 1;
        if (TB_GET_WDL(probeResult) == TB_LOSS) *outWDL = -1;
    }

    return true;
}

bool ProbeTablebase_WDL(const Position& pos, int32_t* outWDL)
{
    ASSERT(pos.IsValid());
    ASSERT(!pos.IsInCheck(GetOppositeColor(pos.GetSideToMove())));

    if (!HasTablebases() ||
        pos.GetNumPieces() > TB_LARGEST)
    {
        return false;
    }

    uint32_t castlingRights = 0;
    if (pos.GetWhitesCastlingRights() & CastlingRights_ShortCastleAllowed)  castlingRights |= TB_CASTLING_K;
    if (pos.GetWhitesCastlingRights() & CastlingRights_LongCastleAllowed)   castlingRights |= TB_CASTLING_Q;
    if (pos.GetBlacksCastlingRights() & CastlingRights_ShortCastleAllowed)  castlingRights |= TB_CASTLING_k;
    if (pos.GetBlacksCastlingRights() & CastlingRights_LongCastleAllowed)   castlingRights |= TB_CASTLING_q;

    // TODO skip if too many pieces, obvious wins, etc.
    const uint32_t probeResult = tb_probe_wdl(
        pos.Whites().Occupied(),
        pos.Blacks().Occupied(),
        pos.Whites().king | pos.Blacks().king,
        pos.Whites().queens | pos.Blacks().queens,
        pos.Whites().rooks | pos.Blacks().rooks,
        pos.Whites().bishops | pos.Blacks().bishops,
        pos.Whites().knights | pos.Blacks().knights,
        pos.Whites().pawns | pos.Blacks().pawns,
        castlingRights,
        pos.GetEnPassantSquare().IsValid() ? pos.GetEnPassantSquare().Index() : 0,
        pos.GetSideToMove() == Color::White);

    if (probeResult != TB_RESULT_FAILED)
    {
#ifdef COLLECT_SEARCH_STATS
        ctx.stats.tbHits++;
#endif // COLLECT_SEARCH_STATS

        // wins/losses are certain only if half move count is zero
        if (pos.GetHalfMoveCount() == 0)
        {
            if (probeResult == TB_LOSS)
            {
                *outWDL = -1;
                return true;
            }
            else if (probeResult == TB_WIN)
            {
                *outWDL = 1;
                return true;
            }
        }

        // draws are certain, no matter the half move counter
        if (probeResult != TB_LOSS && probeResult != TB_WIN)
        {
            *outWDL = 0;
            return true;
        }
    }

    return false;
}

#else // !USE_TABLE_BASES

bool HasTablebases()
{
    return false;
}

void LoadTablebase(const char*) { }
void UnloadTablebase() { }

bool ProbeTablebase_Root(const Position&, Move&, uint32_t*, int32_t*)
{
    return false;
}

bool ProbeTablebase_WDL(const Position&, int32_t*)
{
    return false;
}

#endif // USE_TABLE_BASES
