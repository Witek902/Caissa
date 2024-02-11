#include "Endgame.hpp"
#include "Evaluate.hpp"
#include "Position.hpp"
#include "Material.hpp"
#include "Move.hpp"
#include "MoveList.hpp"
#include "MoveGen.hpp"
#include "Math.hpp"
#include "PackedNeuralNetwork.hpp"

#include <bitset>
#include <vector>
#include <mutex>

// KPK evaluation is based on Stockfish bitbase:
// https://github.com/official-stockfish/Stockfish/blob/master/src/bitbase.cpp
namespace KPKEndgame {

// 2 - side to move
// 24 - pawn squares (files A-D, ranks 2-7)
// 64 - white king position
// 64 - black king position
constexpr uint32_t MaxIndex = 2 * 24 * 64 * 64;

static std::bitset<MaxIndex> lookupTable;

// encode bitbase index
uint32_t EncodeIndex(Color sideToMove, Square blackKingSq, Square whiteKingSq, Square pawnSq)
{
    ASSERT(blackKingSq.IsValid());
    ASSERT(whiteKingSq.IsValid());
    ASSERT(pawnSq.File() <= 3);
    ASSERT(pawnSq.Rank() >= 1);
    ASSERT(pawnSq.Rank() <= 6);

    return
        uint32_t(whiteKingSq.Index()) |
        (uint32_t(blackKingSq.Index()) << 6) |
        (uint32_t(sideToMove) << 12) |
        (uint32_t(pawnSq.File()) << 13) |
        (uint32_t(6u - pawnSq.Rank()) << 15);
}

enum Result : uint8_t
{
    INVALID = 0,
    UNKNOWN = 1,
    DRAW = 2,
    WIN = 4
};

Result& operator|=(Result& r, Result v) { return r = Result(r | v); }

struct KPKPosition
{
    KPKPosition() = default;
    explicit KPKPosition(uint32_t idx);
    operator Result() const { return result; }
    Result Classify(const std::vector<KPKPosition>& db);

    Color sideToMove : 1;
    Result result : 3;
    Square kingSquare[2];
    Square pawnSquare;

    // TODO: distance to queening?
};

static_assert(sizeof(KPKPosition) <= 4, "Invalid KPKPosition size");

bool Probe(Square whiteKingSq, Square pawnSq, Square blackKingSq, Color sideToMove)
{
    if (pawnSq.File() >= 4)
    {
        whiteKingSq = whiteKingSq.FlippedFile();
        blackKingSq = blackKingSq.FlippedFile();
        pawnSq = pawnSq.FlippedFile();
    }

    const uint32_t index = EncodeIndex(sideToMove, blackKingSq, whiteKingSq, pawnSq);
    ASSERT(index < MaxIndex);

    return lookupTable[index];
}

void Init()
{
    std::vector<KPKPosition> db(MaxIndex);

    for (uint32_t i = 0; i < MaxIndex; ++i)
    {
        db[i] = KPKPosition(i);
    }

    // iterate until all positions are visited
    {
        uint32_t repeat = 1;

        while (repeat)
        {
            repeat = 0;
            for (uint32_t i = 0; i < MaxIndex; ++i)
            {
                repeat |= (db[i] == UNKNOWN && db[i].Classify(db) != UNKNOWN);
            }
        }
    }

    uint32_t numWinPositions = 0;

    for (uint32_t i = 0; i < MaxIndex; ++i)
    {
        if (db[i] == WIN)
        {
            lookupTable.set(i);
            numWinPositions++;
        }
    }

    // number of winning KPK positions is known exactly
    ASSERT(numWinPositions == 111282);
    UNUSED(numWinPositions);
}

KPKPosition::KPKPosition(uint32_t idx)
{
    // decode position from index
    kingSquare[0] = Square(idx & 0x3F);
    kingSquare[1] = Square((idx >> 6) & 0x3F);
    sideToMove = Color((idx >> 12) & 0x1);
    pawnSquare = Square((idx >> 13) & 0x3, 6u - ((idx >> 15) & 0x7));

    const Bitboard pawnAttacks = Bitboard::GetPawnAttacks(pawnSquare, White);

    // Invalid if two pieces are on the same square or if a king can be captured
    if (Square::Distance(kingSquare[0], kingSquare[1]) <= 1
        || kingSquare[0] == pawnSquare
        || kingSquare[1] == pawnSquare
        || (sideToMove == White && (pawnAttacks & kingSquare[1].GetBitboard())))
    {
        result = INVALID;
    }
    // Win if the pawn can be promoted without getting captured
    else if (sideToMove == White
        && pawnSquare.Rank() == 6
        && kingSquare[0] != pawnSquare.North()
        && (    Square::Distance(kingSquare[1], pawnSquare.North()) > 1
            || (Square::Distance(kingSquare[0], pawnSquare.North()) == 1)))
    {
        result = WIN;
    }
    // Draw if it is stalemate or the black king can capture the pawn
    else if (sideToMove == Black
        && (!(Bitboard::GetKingAttacks(kingSquare[1]) & ~(Bitboard::GetKingAttacks(kingSquare[0]) | pawnAttacks))
            || (Bitboard::GetKingAttacks(kingSquare[1]) & ~Bitboard::GetKingAttacks(kingSquare[0]) & pawnSquare.GetBitboard())))
    {
        result = DRAW;
    }
    // Position will be classified later
    else
    {
        result = UNKNOWN;
    }
}

Result KPKPosition::Classify(const std::vector<KPKPosition>& db)
{
    const Result Good = sideToMove == White ? WIN : DRAW;
    const Result Bad = sideToMove == White ? DRAW : WIN;

    Result r = INVALID;
    const Bitboard b = Bitboard::GetKingAttacks(kingSquare[(uint32_t)sideToMove]);

    b.Iterate([&](uint32_t square) INLINE_LAMBDA
    {
        r |= sideToMove == White ?
            db[EncodeIndex(Black, kingSquare[1], square, pawnSquare)] :
            db[EncodeIndex(White, square, kingSquare[0], pawnSquare)];
    });

    if (sideToMove == White)
    {
        // single push
        if (pawnSquare.Rank() < 6)
        {
            r |= db[EncodeIndex(Black, kingSquare[1], kingSquare[0], pawnSquare.North())];
        }
        // double push
        if (pawnSquare.Rank() == 1 && pawnSquare.North() != kingSquare[0] && pawnSquare.North() != kingSquare[1])
        {
            r |= db[EncodeIndex(Black, kingSquare[1], kingSquare[0], pawnSquare.North().North())];
        }
    }

    return result = r & Good ? Good : r & UNKNOWN ? UNKNOWN : Bad;
}

} // KPKEndgame

using EndgameEvaluationFunc = bool (*)(const Position&, int32_t&, int32_t&);

// map: material mask -> function index
static uint8_t s_endgameEvaluationMap[MaterialMask_MAX] = { UINT8_MAX };

// all registered functions
static EndgameEvaluationFunc s_endgameEvaluationFunctions[UINT8_MAX];
static uint8_t s_numEndgameEvaluationFunctions = 0;

static void RegisterEndgame(MaterialMask materialMask, const EndgameEvaluationFunc& func)
{
    ASSERT(materialMask < MaterialMask_MAX);
    ASSERT(s_numEndgameEvaluationFunctions < UINT8_MAX);

    // function already registered
    ASSERT(s_endgameEvaluationMap[materialMask] == UINT8_MAX);
    ASSERT(s_endgameEvaluationMap[FlipColor(materialMask)] == UINT8_MAX);

    const uint8_t functionIndex = s_numEndgameEvaluationFunctions++;
    s_endgameEvaluationFunctions[functionIndex] = func;
    s_endgameEvaluationMap[materialMask] = functionIndex;
}

static void RegisterEndgame(MaterialMask materialMask, const uint8_t functionIndex)
{
    ASSERT(materialMask < MaterialMask_MAX);

    // function already registered
    ASSERT(s_endgameEvaluationMap[materialMask] == UINT8_MAX);
    ASSERT(s_endgameEvaluationMap[FlipColor(materialMask)] == UINT8_MAX);

    s_endgameEvaluationMap[materialMask] = functionIndex;
}

static const Bitboard winningFilesKQvKP = Bitboard::FileBitboard<1>() | Bitboard::FileBitboard<3>() | Bitboard::FileBitboard<4>() | Bitboard::FileBitboard<6>();

// Rook(s) and/or Queen(s) vs. lone king
static bool EvaluateEndgame_KXvK(const Position& pos, int32_t& outScore, int32_t& outScale)
{
    UNUSED(outScale);

    ASSERT(pos.Blacks().OccupiedExcludingKing().Count() == 0);

    const Square strongKing(FirstBitSet(pos.Whites().king));
    const Square weakKing(FirstBitSet(pos.Blacks().king));

    if (pos.GetSideToMove() == Black)
    {
        MoveList moves;
        GenerateKingMoveList(pos, Bitboard::GetKingAttacks(pos.GetOpponentSide().GetKingSquare()), moves);

        // TODO this does not handle all cases
        // detect stalemate
        if (moves.Size() == 0)
        {
            outScore = 0;
            return true;
        }

        // check if a piece can be captured immediately
        if (pos.Whites().Occupied().Count() == 2)
        {
            for (uint32_t i = 0; i < moves.Size(); ++i)
            {
                if (moves.GetMove(i).IsCapture())
                {
                    outScore = 0;
                    return true;
                }
            }
        }
    }

    const Bitboard occupied = pos.Whites().Occupied();

    int32_t materialScore = 
        c_queenValue.eg * pos.Whites().queens.Count() +
        c_rookValue.eg * pos.Whites().rooks.Count() +
        c_bishopValue.eg * pos.Whites().bishops.Count() / 4 +
        c_knightValue.eg * pos.Whites().knights.Count() +
        c_pawnValue.eg * pos.Whites().pawns.Count();

    if (materialScore > 4000) materialScore = 4000 + (materialScore - 4000) / 16;

    outScore = KnownWinValue + materialScore;
    outScore += 256 * (3 - weakKing.EdgeDistance());
    outScore -= 8 * Square::Distance(weakKing, strongKing); // push kings close

    pos.Whites().knights.Iterate([&](uint32_t square) INLINE_LAMBDA
    {
        // penalty for lack of mobility
        const Bitboard attacks = Bitboard::GetKnightAttacks(Square(square)) & ~occupied;
        const uint32_t mobility = attacks.Count();
        outScore += std::min(4u, mobility);

        outScore -= Square::Distance(weakKing, Square(square)); // push knight towards weak king
    });

    pos.Whites().rooks.Iterate([&](uint32_t square) INLINE_LAMBDA
    {
        // penalty for lack of mobility
        const Bitboard attacks = Bitboard::GenerateRookAttacks(Square(square), occupied) & ~occupied;
        const uint32_t mobility = attacks.Count();
        if (mobility == 0) outScore -= 512;
        outScore += std::min(8u, mobility);
    });

    outScore = std::clamp(outScore, 0, TablebaseWinValue - 1);

    return true;
}

// knight(s) vs. lone king
static bool EvaluateEndgame_KNvK(const Position& pos, int32_t& outScore, int32_t& outScale)
{
    UNUSED(outScale);

    ASSERT(pos.Blacks().OccupiedExcludingKing().Count() == 0);

    const Square strongKing(FirstBitSet(pos.Whites().king));
    const Square weakKing(FirstBitSet(pos.Blacks().king));
    const int32_t numKnights = pos.Whites().knights.Count();

    if (numKnights <= 2)
    {
        // NOTE: there are checkmates with two knights, but they cannot be forced from all positions
        outScore = 0;
    }
    else // whiteKnights >= 3
    {
        outScore = KnownWinValue;
        outScore += c_knightValue.eg * (numKnights - 2); // prefer keeping the knights
        outScore += 8 * (3 - weakKing.AnyCornerDistance()); // push king to corner
        outScore -= Square::Distance(weakKing, strongKing); // push kings close

        // limit weak king movement
        const Bitboard kingLegalSquares =
            Bitboard::GetKingAttacks(weakKing) &
            ~Bitboard::GetKnightAttacks(pos.Whites().knights) &
            ~Bitboard::GetKingAttacks(strongKing);
        outScore -= kingLegalSquares.Count();

        // push knights towards king
        pos.Whites().knights.Iterate([&](uint32_t square) INLINE_LAMBDA
        {
            outScore -= Square::Distance(weakKing, Square(square));
        });
    }

    return true;
}

// knight(s) vs. knight(s)
static bool EvaluateEndgame_KNvKN(const Position& pos, int32_t& outScore, int32_t& outScale)
{
    UNUSED(outScale);

    ASSERT(pos.Whites().pawns == 0 && pos.Whites().bishops == 0 && pos.Whites().knights != 0 && pos.Whites().rooks == 0 && pos.Whites().queens == 0);
    ASSERT(pos.Blacks().pawns == 0 && pos.Blacks().bishops == 0 && pos.Blacks().knights != 0 && pos.Blacks().rooks == 0 && pos.Blacks().queens == 0);

    const uint32_t numWhiteKnights = pos.Whites().knights.Count();
    const uint32_t numBlackKnights = pos.Blacks().knights.Count();

    if (numWhiteKnights == 1 && numBlackKnights == 1)
    {
        outScore = 0;
        return true;
    }
    else if (numWhiteKnights <= 2 && numBlackKnights <= 2)
    {
        const Square strongKing = pos.Whites().GetKingSquare();
        const Square weakKing = pos.Blacks().GetKingSquare();

        outScore = 4 * (strongKing.AnyCornerDistance() - weakKing.AnyCornerDistance());

        // push knights towards enemy king
        pos.Whites().knights.Iterate([&](uint32_t square) INLINE_LAMBDA
        {
            outScore += 4 - Square::Distance(weakKing, Square(square)) / 2; 
        });
        pos.Blacks().knights.Iterate([&](uint32_t square) INLINE_LAMBDA
        {
            outScore -= 4 - Square::Distance(strongKing, Square(square)) / 2;
        });

        return true;
    }

    return false;
}

// bishop(s) vs. lone king
static bool EvaluateEndgame_KBvK(const Position& pos, int32_t& outScore, int32_t& outScale)
{
    UNUSED(outScale);

    ASSERT(pos.Blacks().pawns == 0 && pos.Blacks().bishops == 0 && pos.Blacks().rooks == 0 && pos.Blacks().queens == 0);

    const Square strongKing(FirstBitSet(pos.Whites().king));
    const Square weakKing(FirstBitSet(pos.Blacks().king));
    const int32_t whiteBishops = pos.Whites().bishops.Count();
    const int32_t blackKnights = pos.Blacks().knights.Count();

    const uint32_t numLightSquareBishops = (pos.Whites().bishops & Bitboard::LightSquares()).Count();
    const uint32_t numDarkSquareBishops = (pos.Whites().bishops & Bitboard::DarkSquares()).Count();

    if (blackKnights <= 1 && (numLightSquareBishops == 0 || numDarkSquareBishops == 0))
    {
        outScore = 0;
        return true;
    }
    else if (blackKnights == 0 && (numLightSquareBishops >= 1 || numDarkSquareBishops >= 1))
    {
        outScore = KnownWinValue;
        outScore += 64 * (whiteBishops - 2); // prefer keeping the bishops on board
        outScore += 8 * (3 - weakKing.AnyCornerDistance()); // push king to corner
        outScore += (7 - Square::Distance(weakKing, strongKing)); // push kings close
        return true;
    }
    return false;
}

// bishop(s) vs. bishop(s)
static bool EvaluateEndgame_KBvKB(const Position& pos, int32_t& outScore, int32_t& outScale)
{
    UNUSED(outScale);

    ASSERT(pos.Whites().pawns == 0 && pos.Whites().bishops != 0 && pos.Whites().knights == 0 && pos.Whites().rooks == 0 && pos.Whites().queens == 0);
    ASSERT(pos.Blacks().pawns == 0 && pos.Blacks().bishops != 0 && pos.Blacks().knights == 0 && pos.Blacks().rooks == 0 && pos.Blacks().queens == 0);

    const int32_t whiteLightSquaresBishops = (pos.Whites().bishops & Bitboard::LightSquares()).Count();
    const int32_t whiteDarkSquaresBishops = (pos.Whites().bishops & ~Bitboard::LightSquares()).Count();

    const int32_t blackLightSquaresBishops = (pos.Blacks().bishops & Bitboard::LightSquares()).Count();
    const int32_t blackDarkSquaresBishops = (pos.Blacks().bishops & ~Bitboard::LightSquares()).Count();

    const int32_t whiteBishops = whiteLightSquaresBishops + whiteDarkSquaresBishops;
    const int32_t blackBishops = blackLightSquaresBishops + blackDarkSquaresBishops;

    if (whiteBishops <= 1 && blackBishops <= 1)
    {
        outScore = 0;
        return true;
    }

    if ((whiteLightSquaresBishops == 0 || whiteDarkSquaresBishops == 0) &&
        (blackLightSquaresBishops == 0 || blackDarkSquaresBishops == 0))
    {
        outScore = 0;
        return true;
    }

    return false;
}

// knight + bishop vs. lone king
static bool EvaluateEndgame_KNBvK(const Position& pos, int32_t& outScore, int32_t& outScale)
{
    UNUSED(outScale);

    ASSERT(pos.Whites().bishops > 0 && pos.Whites().knights > 0 && pos.Whites().rooks == 0 && pos.Whites().queens == 0);
    ASSERT(pos.Blacks().OccupiedExcludingKing().Count() == 0);

    const Square strongKing(FirstBitSet(pos.Whites().king));
    const Square weakKing(FirstBitSet(pos.Blacks().king));

    // push king to 'right' board corner
    const Square kingSquare = (pos.Whites().bishops & Bitboard::DarkSquares()) ? weakKing : weakKing.FlippedFile();

    outScore = KnownWinValue;
    outScore += c_pawnValue.eg * pos.Whites().pawns.Count(); // prefer keeping pawns
    outScore += c_knightValue.eg * (pos.Whites().bishops.Count() - 1); // prefer keeping bishops
    outScore += c_bishopValue.eg * (pos.Whites().knights.Count() - 1); // prefer keeping knights
    outScore += 4 * (3 - kingSquare.EdgeDistance()); // push king to edge
    outScore += 4 * (7 - kingSquare.DarkCornerDistance()); // push king to right corner
    outScore += (7 - Square::Distance(weakKing, strongKing)); // push kings close

    // limit weak king movement
    const Bitboard kingLegalSquares =
        Bitboard::GetKingAttacks(weakKing) &
        ~Bitboard::GetKnightAttacks(pos.Whites().knights) &
        ~Bitboard::GetKingAttacks(strongKing);
    outScore -= kingLegalSquares.Count();

    return true;
}

// pawn(s) vs. lone king
static bool EvaluateEndgame_KPvK(const Position& pos, int32_t& outScore, int32_t& outScale)
{
    UNUSED(outScale);

    ASSERT(pos.Whites().pawns.Count() > 0);
    ASSERT(pos.Blacks().OccupiedExcludingKing().Count() == 0);

    const Square strongKing(FirstBitSet(pos.Whites().king));
    const Square weakKing(FirstBitSet(pos.Blacks().king));
    const int32_t numPawns = pos.Whites().pawns.Count();
    const int32_t blackToMove = pos.GetSideToMove() == Black;

    Square strongKingSq = strongKing;
    Square weakKingSq = weakKing;
    Square pawnSquare = LastBitSet(pos.Whites().pawns);

    if (numPawns == 1)
    {
        if (!KPKEndgame::Probe(strongKingSq, pawnSquare, weakKingSq, pos.GetSideToMove()))
        {
            // bitbase draw
            outScore = 0;
            return true;
        }

        ASSERT(pawnSquare.Rank() < 7);
        Square keySquare = Square(pawnSquare.File(), pawnSquare.Rank() + 1);
        if (pawnSquare.Rank() < 6) keySquare = Square(pawnSquare.File(), pawnSquare.Rank() + 2);

        outScore = KnownWinValue + c_pawnValue.eg;
        outScore += 8 * pawnSquare.Rank();
        outScore -= (int32_t)Square::Distance(keySquare, strongKingSq); // put strong king in front of pawn
        outScore += (int32_t)Square::Distance(pawnSquare, weakKingSq); // try to capture pawn
        return true;
    }
    else if (numPawns == 2)
    {
        const Square secondPawnSquare = FirstBitSet(pos.Whites().pawns & ~pawnSquare.GetBitboard());
        ASSERT(secondPawnSquare.Rank() <= pawnSquare.Rank());

        bool isWin = false;

        // connected passed pawns
        if ((Square::Distance(pawnSquare, secondPawnSquare) <= 1) &&
            (pawnSquare.File() != secondPawnSquare.File()) &&
            (pos.GetSideToMove() == White || (pawnSquare.Rank() != secondPawnSquare.Rank())))
        {
            isWin = true;
        }

        // losing side can't capture both pawns
        if (std::abs(int32_t(pawnSquare.File()) - int32_t(secondPawnSquare.File())) == 2 &&
            Square::Distance(pawnSquare, weakKing) > 2 + blackToMove &&
            Square::Distance(secondPawnSquare, weakKing) > 2 + blackToMove)
        {
            isWin = true;
        }

        // losing side can't capture both pawns
        if (std::abs(int32_t(pawnSquare.File()) - int32_t(secondPawnSquare.File())) >= 3 &&
            Square::Distance(pawnSquare, weakKing) > 3 + blackToMove &&
            Square::Distance(secondPawnSquare, weakKing) > 3 + blackToMove)
        {
            isWin = true;
        }

        // losing side can't capture both pawns
        if (Square::Distance(pawnSquare, secondPawnSquare) > 5 && pos.GetSideToMove() == White)
        {
            isWin = true;
        }

        // bitbase win if weak king is not in front of pawns
        if (!isWin &&
            pos.GetSideToMove() == White &&
            (weakKingSq.Rank() < 7 || weakKingSq.File() != pawnSquare.File()) &&
            KPKEndgame::Probe(strongKingSq, pawnSquare, weakKingSq, pos.GetSideToMove()))
        {
            isWin = true;
        }

        if (isWin)
        {
            outScore = KnownWinValue + 2 * c_pawnValue.eg;
            outScore += 8 * pawnSquare.Rank();
            outScore += 6 * secondPawnSquare.Rank();
            outScore += 7 - std::max(0, (int32_t)Square::Distance(pawnSquare, strongKingSq) - 1); // push kings close to pawn
            outScore += std::max(0, (int32_t)Square::Distance(pawnSquare, weakKingSq) - 1); // push kings close to pawn
            return true;
        }
    }

    // if all pawns are on A/H file, then it's a draw
    // if the weak king is already blocking promotion or will reach promotion square faster than a pawn or oponent's king
    {
        const Square promotionSquare(pawnSquare.File(), 7);

        const uint32_t weakKingDistance = Square::Distance(weakKing, promotionSquare);
        const uint32_t strongKingDistance = Square::Distance(strongKing, promotionSquare);
        const uint32_t pawnDistance = Square::Distance(pawnSquare, promotionSquare);

        if (((pos.Whites().pawns & ~Bitboard::FileBitboard<0>()) == 0) &&
            ((Square::Distance(weakKing, Square_a8) <= 1) ||
             (weakKingDistance < pawnDistance + blackToMove &&
              weakKingDistance + 1 < strongKingDistance + blackToMove)))
        {
            outScore = 0;
            return true;
        }

        if (((pos.Whites().pawns & ~Bitboard::FileBitboard<7>()) == 0) &&
            ((Square::Distance(weakKing, Square_h8) <= 1) ||
             (weakKingDistance < pawnDistance + blackToMove &&
              weakKingDistance + 1 < strongKingDistance + blackToMove)))
        {
            outScore = 0;
            return true;
        }
    }

    // win if king can't reach promoting pawn
    {
        uint32_t mostAdvancedPawnRank = 0;
        for (uint32_t rank = 1; rank < 7; ++rank)
        {
            if (pos.Whites().pawns & Bitboard::RankBitboard(rank))
            {
                mostAdvancedPawnRank = rank;
            }
        }

        if (weakKing.Rank() + (uint32_t)pos.GetSideToMove() < mostAdvancedPawnRank)
        {
            outScore = KnownWinValue + numPawns * c_pawnValue.eg;
            outScore += 8 * mostAdvancedPawnRank;
            return true;
        }
    }

    return false;
}

// pawn(s) vs. pawn(s)
static bool EvaluateEndgame_KPvKP(const Position& pos, int32_t& outScore, int32_t& outScale)
{
    UNUSED(outScale);

    ASSERT(pos.Whites().pawns != 0 && pos.Whites().bishops == 0 && pos.Whites().knights == 0 && pos.Whites().rooks == 0 && pos.Whites().queens == 0);
    ASSERT(pos.Blacks().pawns != 0 && pos.Blacks().bishops == 0 && pos.Blacks().knights == 0 && pos.Blacks().rooks == 0 && pos.Blacks().queens == 0);

    Square whiteKing(FirstBitSet(pos.Whites().king));
    Square blackKing(FirstBitSet(pos.Blacks().king));
    const uint32_t numWhitePawns = pos.Whites().pawns.Count();
    const uint32_t numBlackPawns = pos.Blacks().pawns.Count();

    if (numWhitePawns == 1 && numBlackPawns == 1)
    {
        Square whitePawn = FirstBitSet(pos.Whites().pawns);
        Square blackPawn = FirstBitSet(pos.Blacks().pawns);

        if (whitePawn.File() >= 4)
        {
            whiteKing = whiteKing.FlippedFile();
            blackKing = blackKing.FlippedFile();
            whitePawn = whitePawn.FlippedFile();
            blackPawn = blackPawn.FlippedFile();
        }
        ASSERT(whitePawn.File() < 4);

        if (whitePawn.Rank() < 4 || whitePawn.File() == 0 || whitePawn.File() == 7)
        {
            if (!KPKEndgame::Probe(whiteKing, whitePawn, blackKing, pos.GetSideToMove()))
            {
                // bitbase draw
                outScore = 0;
                return true;
            }
        }

        const Square whiteQueeningSquare(whitePawn.File(), 7);
        const Square blackQueeningSquare(blackPawn.File(), 0);

        const int32_t whiteToQueen = 7 - whitePawn.Rank();
        const int32_t blackToQueen = blackPawn.Rank();

        // if passed pawn
        if (whitePawn.Rank() >= blackPawn.Rank() || (whitePawn.File() > blackPawn.File() + 1) || (blackPawn.File() > whitePawn.File() + 1))
        {
            const bool whiteBlockedByKing = whiteKing.File() == whitePawn.File() && whiteKing.Rank() > whitePawn.Rank();
            const bool blackBlockedByKing = blackKing.File() == blackPawn.File() && blackKing.Rank() < blackPawn.Rank();

            // incorrect opponent pawn may lead to drawing KQvKP endgame
            const bool correctWhitePawn = whitePawn.GetBitboard() & winningFilesKQvKP;
            const bool correctBlackPawn = blackPawn.GetBitboard() & winningFilesKQvKP;

            if (whiteToQueen + 2 + whiteBlockedByKing - correctBlackPawn < blackToQueen &&
                Square::Distance(blackKing, whitePawn) > whiteToQueen + 1 - blackBlockedByKing)
            {
                outScore = KnownWinValue + whitePawn.Rank();
                return true;
            }

            if (blackToQueen + 3 + blackBlockedByKing - correctWhitePawn < whiteToQueen &&
                Square::Distance(whiteKing, blackPawn) > whiteToQueen - whiteBlockedByKing)
            {
                outScore = -KnownWinValue - (7 - blackPawn.Rank());
                return true;
            }
        }
    }

    return false;
}

// bishop(s) + pawn(s) vs. lone king
static bool EvaluateEndgame_KBPvK(const Position& pos, int32_t& outScore, int32_t& outScale)
{
    UNUSED(outScale);

    ASSERT(pos.Whites().pawns != 0 && pos.Whites().bishops != 0 && pos.Whites().knights == 0 && pos.Whites().rooks == 0 && pos.Whites().queens == 0);
    ASSERT(pos.Blacks().OccupiedExcludingKing().Count() == 0);

    const Square strongKing(FirstBitSet(pos.Whites().king));
    const Square weakKing(FirstBitSet(pos.Blacks().king));

    const int32_t blackToMove = pos.GetSideToMove() == Black;

    // if all pawns are on A/H file and we have a wrong bishop, then it's a draw
    // if the weak king is already blocking promotion or will reach promotion square faster than a pawn or oponent's king
    {
        const Square pawnSquare(LastBitSet(pos.Whites().pawns));
        const Square promotionSquare(pawnSquare.File(), 7);

        if (((pos.Whites().pawns & ~Bitboard::FileBitboard<0>()) == 0) &&
            ((pos.Whites().bishops & Bitboard::LightSquares()) == 0) &&
            ((Square::Distance(weakKing, Square_a8) <= 1)))
        {
            outScore = 0;
            return true;
        }

        if (((pos.Whites().pawns & ~Bitboard::FileBitboard<7>()) == 0) &&
            ((pos.Whites().bishops & Bitboard::DarkSquares()) == 0) &&
            ((Square::Distance(weakKing, Square_h8) <= 1)))
        {
            outScore = 0;
            return true;
        }
    }

    if (pos.Whites().pawns.Count() == 1)
    {
        const Square pawnSquare(FirstBitSet(pos.Whites().pawns));
        const Square promotionSquare(pawnSquare.File(), 7);

        const Square bishopSquare(FirstBitSet(pos.Whites().bishops));

        const bool bishopOnLightSquare = pos.Whites().bishops & Bitboard::LightSquares();
        const bool promotionOnLightSquare = promotionSquare.GetBitboard() & Bitboard::LightSquares();

        // if we have a "good" bishop and weak king can't easily capture pawn, then it's a win
        if (bishopOnLightSquare == promotionOnLightSquare &&
            Square::Distance(strongKing, weakKing) > 2 &&
            Square::Distance(weakKing, pawnSquare) > 2 + blackToMove &&
            Square::Distance(strongKing, bishopSquare) > 1 && !bishopSquare.IsCorner())
        {
            outScore = KnownWinValue;
            outScore += 16 * pawnSquare.Rank();
            outScore += Square::Distance(weakKing, pawnSquare);
            outScore -= Square::Distance(strongKing, pawnSquare);
            return true;
        }

        // bishop blocked on a7
        if (bishopSquare == Square_a7 && pawnSquare == Square_b6 &&
            Square::Distance(weakKing, Square_b7) <= 1 &&
            Square::Distance(strongKing, Square_b7) + blackToMove > 2)
        {
            outScore = 0;
            return true;
        }

        // bishop blocked on h7
        if (bishopSquare == Square_h7 && pawnSquare == Square_g6 &&
            Square::Distance(weakKing, Square_g7) <= 1 &&
            Square::Distance(strongKing, Square_g7) + blackToMove > 2)
        {
            outScore = 0;
            return true;
        }
    }

    return false;
}

// knight(s) + pawn(s) vs. lone king
static bool EvaluateEndgame_KNPvK(const Position& pos, int32_t& outScore, int32_t& outScale)
{
    UNUSED(outScale);

    ASSERT(pos.Whites().pawns > 0 && pos.Whites().bishops == 0 && pos.Whites().knights > 0 && pos.Whites().rooks == 0 && pos.Whites().queens == 0);
    ASSERT(pos.Blacks().OccupiedExcludingKing().Count() == 0);

    const Square strongKingSq(FirstBitSet(pos.Whites().king));
    const Square weakKingSq(FirstBitSet(pos.Blacks().king));

    // a knight protecting a rook pawn on the seventh rank is a draw 
    if (pos.Whites().pawns.Count() == 1 && pos.Whites().knights.Count() == 1)
    {
        const Square pawnSquare(FirstBitSet(pos.Whites().pawns));
        const Square knightSquare(FirstBitSet(pos.Whites().knights));

        if ((pawnSquare == Square_a7 || pawnSquare == Square_h7) &&
            Square::Distance(pawnSquare, weakKingSq) == 1 &&
            (Bitboard::GetKnightAttacks(knightSquare) & pawnSquare.GetBitboard()))
        {
            outScore = 0;
            return true;
        }

        if ((pawnSquare.File() != knightSquare.File() || pawnSquare.Rank() > knightSquare.Rank()) &&
            KPKEndgame::Probe(strongKingSq, pawnSquare, weakKingSq, pos.GetSideToMove()))
        {
            ASSERT(pawnSquare.Rank() < 7);
            Square keySquare = Square(pawnSquare.File(), pawnSquare.Rank() + 1);
            if (pawnSquare.Rank() < 6) keySquare = Square(pawnSquare.Rank() + 2, pawnSquare.File());

            outScore = KnownWinValue + c_pawnValue.eg + c_knightValue.eg;
            outScore += 8 * pawnSquare.Rank();
            outScore -= (int32_t)Square::Distance(keySquare, strongKingSq); // put strong king in front of pawn
            outScore += (int32_t)Square::Distance(pawnSquare, weakKingSq); // try to capture pawn
            outScore -= (int32_t)Square::Distance(pawnSquare, knightSquare); // move knight towards pawn
            return true;
        }
    }

    return false;
}

// Queen vs. Pawn
static bool EvaluateEndgame_KQvKP(const Position& pos, int32_t& outScore, int32_t& outScale)
{
    UNUSED(outScale);

    ASSERT(pos.Whites().pawns == 0 && pos.Whites().bishops == 0 && pos.Whites().knights == 0 && pos.Whites().rooks == 0 && pos.Whites().queens != 0);
    ASSERT(pos.Blacks().pawns != 0 && pos.Blacks().bishops == 0 && pos.Blacks().knights == 0 && pos.Blacks().rooks == 0 && pos.Blacks().queens == 0);

    if (pos.Whites().queens.Count() == 1 && pos.Blacks().pawns.Count() == 1)
    {
        const Square strongKing(FirstBitSet(pos.Whites().king));
        const Square weakKing(FirstBitSet(pos.Blacks().king));
        const Square queenSquare(FirstBitSet(pos.Whites().queens));
        const Square pawnSquare(FirstBitSet(pos.Blacks().pawns));

        // push kings closer
        outScore = 7 - Square::Distance(weakKing, strongKing);

        if (pawnSquare.Rank() >= 3)
        {
            // if pawn if 3 squares from promotion (or more) it's 100% win for white
            if (pos.GetSideToMove() == White && !pos.IsInCheck(White))
            {
                outScore += KnownWinValue;
            }
            else
            {
                outScore += 800;
            }
        }
        else if (pawnSquare.Rank() != 1 || Square::Distance(weakKing, pawnSquare) != 1 || (pawnSquare.GetBitboard() & winningFilesKQvKP))
        {
            // if the pawn is about to promote, is not on rook or bishop file, then it's most likely a win
            outScore += 800;
        }

        return true;
    }

    return false;
}

// Queen vs. Knight+Pawn
static bool EvaluateEndgame_KQvKNP(const Position& pos, int32_t& outScore, int32_t& outScale)
{
    UNUSED(outScore);

    ASSERT(pos.Whites().pawns == 0 && pos.Whites().bishops == 0 && pos.Whites().knights == 0 && pos.Whites().rooks == 0 && pos.Whites().queens != 0);
    ASSERT(pos.Blacks().pawns != 0 && pos.Blacks().bishops == 0 && pos.Blacks().knights != 0 && pos.Blacks().rooks == 0 && pos.Blacks().queens == 0);

    if (pos.Whites().queens.Count() == 1 && pos.Blacks().pawns.Count() == 1 && pos.Blacks().knights.Count() == 1)
    {
        const Square strongKing(FirstBitSet(pos.Whites().king));
        const Square weakKing(FirstBitSet(pos.Blacks().king));
        const Square queenSquare(FirstBitSet(pos.Whites().queens));
        const Square pawnSquare(FirstBitSet(pos.Blacks().pawns));
        const Square knightSquare(FirstBitSet(pos.Blacks().knights));

        if (strongKing.Rank() >= 6 &&
            pawnSquare.Rank() <= 2 &&
            weakKing.Rank() <= 2 &&
            Square::Distance(pawnSquare, weakKing) <= 1 &&
            Square::Distance(knightSquare, weakKing) <= 1 &&
            Square::Distance(knightSquare, pawnSquare) <= 2)
        {
            outScale = c_endgameScaleMax / 4;
        }
    }

    return false;
}

// Queen vs. Knight+Bishop
static bool EvaluateEndgame_KQvKBN(const Position& pos, int32_t& outScore, int32_t& outScale)
{
    UNUSED(outScore);

    ASSERT(pos.Whites().pawns == 0 && pos.Whites().bishops == 0 && pos.Whites().knights == 0 && pos.Whites().rooks == 0 && pos.Whites().queens != 0);
    ASSERT(pos.Blacks().pawns == 0 && pos.Blacks().bishops != 0 && pos.Blacks().knights != 0 && pos.Blacks().rooks == 0 && pos.Blacks().queens == 0);

    // rare Q vs. BN fortress
    // for example: 2Q5/8/8/8/3n4/8/1b6/k2K4 b - - 0 1
    if (pos.Whites().queens.Count() == 1 && pos.Blacks().bishops.Count() == 1 && pos.Blacks().knights.Count() == 1)
    {
        Square strongKing = pos.Whites().GetKingSquare();
        Square weakKing = pos.Blacks().GetKingSquare();
        Square queenSquare(FirstBitSet(pos.Whites().queens));
        Square bishopSquare(FirstBitSet(pos.Blacks().bishops));
        Square knightSquare(FirstBitSet(pos.Blacks().knights));

        // normalize position so the weak king is in left-bottom quadrant
        if (weakKing.Rank() >= 4)
        {
            strongKing = strongKing.FlippedRank();
            weakKing = weakKing.FlippedRank();
            queenSquare = queenSquare.FlippedRank();
            bishopSquare = bishopSquare.FlippedRank();
            knightSquare = knightSquare.FlippedRank();
        }
        if (weakKing.File() >= 4)
        {
            strongKing = strongKing.FlippedFile();
            weakKing = weakKing.FlippedFile();
            queenSquare = queenSquare.FlippedFile();
            bishopSquare = bishopSquare.FlippedFile();
            knightSquare = knightSquare.FlippedFile();
        }

        if (knightSquare == Square_d4 &&
            (weakKing == Square_a1 || weakKing == Square_b1 || weakKing == Square_a2) &&
            (bishopSquare == Square_a1 || bishopSquare == Square_b2))
        {
            outScale = c_endgameScaleMax / 4;
            if (Square::Distance(weakKing, strongKing) > 2)
            {
                outScore = 0;
                return true;
            }
        }
    }

    return false;
}

// Rook vs. Pawn
static bool EvaluateEndgame_KRvKP(const Position& pos, int32_t& outScore, int32_t& outScale)
{
    UNUSED(outScale);

    ASSERT(pos.Whites().pawns == 0 && pos.Whites().bishops == 0 && pos.Whites().knights == 0 && pos.Whites().rooks != 0 && pos.Whites().queens == 0);
    ASSERT(pos.Blacks().pawns != 0 && pos.Blacks().bishops == 0 && pos.Blacks().knights == 0 && pos.Blacks().rooks == 0 && pos.Blacks().queens == 0);

    if (pos.Whites().rooks.Count() == 1 && pos.Blacks().pawns.Count() == 1 && pos.GetSideToMove() == White)
    {
        const Square strongKing(FirstBitSet(pos.Whites().king));
        const Square weakKing(FirstBitSet(pos.Blacks().king));
        const Square strongRook(FirstBitSet(pos.Whites().rooks));
        const Square weakPawn(FirstBitSet(pos.Blacks().pawns));
        const Square pushedPawnSquare = weakPawn.South();
        const Square promotionSquare(weakPawn.File(), 0);

        bool win = false;

        //const bool hangingRook = Square::Distance(weakKing, strongRook) < 2;

        // win if strong king is in front of pawn
        if (strongKing.Rank() < weakPawn.Rank() && strongKing.File() == weakPawn.File())
        {
            win = true;
        }
        // win if strong king is in front of pawn
        if (weakPawn.Rank() > 2 && strongKing.Rank() < weakPawn.Rank() &&
            std::abs((int32_t)strongKing.File() - (int32_t)weakPawn.File()) <= 1)
        {
            win = true;
        }
        // win if pawn is not much advanced or the weak king is too far from pawn and the rook
        if ((weakKing.Rank() + 1 >= weakPawn.Rank() && weakPawn.Rank() > 5) ||
            (weakKing.Rank() > weakPawn.Rank() && weakPawn.Rank() > 4) ||
            (Square::Distance(weakKing, weakPawn) >= 6 && weakPawn.Rank() > 1) ||
            (Square::Distance(weakKing, weakPawn) >= 4 && weakKing.Rank() >= weakPawn.Rank() && weakPawn.Rank() > 1) ||
            (Square::Distance(weakKing, weakPawn) >= 3 && weakKing.Rank() >= weakPawn.Rank() && weakPawn.Rank() > 3))
        {
            win = true;
        }

        if (win)
        {
            outScore = KnownWinValue + 300;
            outScore -= 16 * weakPawn.Rank();
            outScore -= Square::Distance(weakPawn, strongKing);
            outScore += Square::Distance(weakPawn, weakKing);
            return true;
        }

        // piece square tables generated with "validate endgame" utility
        const int16_t blackPawnPsqt[] =
        {
           374,    308,    283,    290,
           249,    231,    240,    279,
           376,    365,    387,    410,
           566,    566,    593,    598,
          1016,    961,   1037,   1085,
          1003,    952,   1013,   1052,
        };

        const uint8_t normalizedFile = weakPawn.File() < 4 ? weakPawn.File() : (7 - weakPawn.File());
        const uint32_t psqtIndex = 4 * (weakPawn.Rank() - 1) + normalizedFile;
        ASSERT(psqtIndex < 24);

        outScore = blackPawnPsqt[psqtIndex];
        outScore -= 64 * Square::Distance(strongKing, promotionSquare);
        outScore += 64 * Square::Distance(weakKing, pushedPawnSquare);
        return true;
    }

    return false;
}

// Rook vs. Knight
static bool EvaluateEndgame_KRvKN(const Position& pos, int32_t& outScore, int32_t& outScale)
{
    UNUSED(outScale);

    ASSERT(pos.Whites().pawns == 0 && pos.Whites().bishops == 0 && pos.Whites().knights == 0 && pos.Whites().rooks != 0 && pos.Whites().queens == 0);
    ASSERT(pos.Blacks().pawns == 0 && pos.Blacks().bishops == 0 && pos.Blacks().knights != 0 && pos.Blacks().rooks == 0 && pos.Blacks().queens == 0);

    if (pos.Whites().rooks.Count() == 1 && pos.Blacks().knights.Count() == 1)
    {
        const Square strongKing(FirstBitSet(pos.Whites().king));
        const Square weakKing(FirstBitSet(pos.Blacks().king));
        const Square strongRook(FirstBitSet(pos.Whites().rooks));
        const Square weakKnight(FirstBitSet(pos.Blacks().knights));

        static const uint8_t blackKingPsqt[] =
        {
           107,     66,     41,     34,     34,     40,     66,    107,
            66,     31,     20,     16,     16,     20,     31,     66,
            40,     20,     11,      8,      8,     11,     20,     40,
            34,     16,      8,      5,      5,      8,     16,     34,
            34,     16,      8,      5,      5,      8,     16,     34,
            40,     20,     11,      8,      8,     11,     20,     40,
            66,     31,     20,     16,     16,     20,     31,     66,
           107,     66,     40,     34,     34,     40,     66,    107,
        };

        outScore = blackKingPsqt[weakKing.mIndex];
        outScore += 7 * (3 - weakKnight.AnyCornerDistance());
        outScore += 16 * (Square::Distance(weakKing, weakKnight) - 1);
        outScore -= 3 * Square::Distance(strongKing, weakKing);
        outScore -= 5 * Square::Distance(strongKing, weakKnight);
        return true;
    }

    return false;
}

// Rook vs. Bishop
static bool EvaluateEndgame_KRvKB(const Position& pos, int32_t& outScore, int32_t& outScale)
{
    UNUSED(outScale);

    ASSERT(pos.Whites().pawns == 0 && pos.Whites().bishops == 0 && pos.Whites().knights == 0 && pos.Whites().rooks != 0 && pos.Whites().queens == 0);
    ASSERT(pos.Blacks().pawns == 0 && pos.Blacks().bishops != 0 && pos.Blacks().knights == 0 && pos.Blacks().rooks == 0 && pos.Blacks().queens == 0);

    if (pos.Whites().rooks.Count() == 1 && pos.Blacks().bishops.Count() == 1)
    {
        const Square strongKing(FirstBitSet(pos.Whites().king));
        const Square weakKing(FirstBitSet(pos.Blacks().king));
        const Square strongRook(FirstBitSet(pos.Whites().rooks));
        const Square weakBishop(FirstBitSet(pos.Blacks().bishops));

        outScore = 8 * (3 - weakKing.EdgeDistance());
        outScore += 2 * weakBishop.AnyCornerDistance();
        return true;
    }

    return false;
}

// Queen vs. Rook
static bool EvaluateEndgame_KQvKR(const Position& pos, int32_t& outScore, int32_t& outScale)
{
    UNUSED(outScale);

    ASSERT(pos.Whites().pawns == 0 && pos.Whites().bishops == 0 && pos.Whites().knights == 0 && pos.Whites().rooks == 0 && pos.Whites().queens != 0);
    ASSERT(pos.Blacks().pawns == 0 && pos.Blacks().bishops == 0 && pos.Blacks().knights == 0 && pos.Blacks().rooks != 0 && pos.Blacks().queens == 0);

    if (pos.Whites().queens.Count() == 1 && pos.Blacks().rooks.Count() == 1)
    {
        const Square strongKing(FirstBitSet(pos.Whites().king));
        const Square weakKing(FirstBitSet(pos.Blacks().king));
        const Square queenSquare(FirstBitSet(pos.Whites().queens));
        const Square rookSquare(FirstBitSet(pos.Blacks().rooks));

        outScore = 400;
        outScore += 8 * (3 - weakKing.EdgeDistance()); // push king to edge
        outScore += (7 - Square::Distance(weakKing, strongKing)); // push kings close
        return true;
    }

    return false;
}

// Queen vs. Knight
static bool EvaluateEndgame_KQvKN(const Position& pos, int32_t& outScore, int32_t& outScale)
{
    UNUSED(outScale);

    ASSERT(pos.Whites().pawns == 0 && pos.Whites().bishops == 0 && pos.Whites().knights == 0 && pos.Whites().rooks == 0 && pos.Whites().queens != 0);
    ASSERT(pos.Blacks().pawns == 0 && pos.Blacks().bishops == 0 && pos.Blacks().knights != 0 && pos.Blacks().rooks == 0 && pos.Blacks().queens == 0);

    if (pos.Whites().queens.Count() == 1 && pos.Blacks().knights.Count() == 1)
    {
        const Square strongKing(FirstBitSet(pos.Whites().king));
        const Square weakKing(FirstBitSet(pos.Blacks().king));
        const Square queenSquare(FirstBitSet(pos.Whites().queens));
        const Square knightSquare(FirstBitSet(pos.Blacks().knights));

        if (pos.GetSideToMove() == Black)
        {
            // detect fork
            if (Bitboard::GetKnightAttacks(knightSquare) & Bitboard::GetKnightAttacks(queenSquare) & Bitboard::GetKnightAttacks(strongKing))
            {
                outScore = 0;
                return true;
            }
        }

        outScore = KnownWinValue + (pos.GetSideToMove() == White ? 100 : -100);
        outScore -= 8 * weakKing.EdgeDistance(); // push king to edge
        outScore -= 2 * Square::Distance(weakKing, strongKing); // push kings close
        outScore += Square::Distance(weakKing, knightSquare); // push losing king and knight close
        return true;
    }

    return false;
}

static bool EvaluateEndgame_KRvKR(const Position& pos, int32_t& outScore, int32_t& outScale)
{
    UNUSED(outScale);

    ASSERT(pos.Whites().pawns == 0 && pos.Whites().bishops == 0 && pos.Whites().knights == 0 && pos.Whites().rooks > 0 && pos.Whites().queens == 0);
    ASSERT(pos.Blacks().pawns == 0 && pos.Blacks().bishops == 0 && pos.Blacks().knights == 0 && pos.Blacks().rooks > 0 && pos.Blacks().queens == 0);

    if (pos.Whites().rooks.Count() == 1 && pos.Blacks().rooks.Count() == 1)
    {
        // assume white is to move
        Square strongKing(FirstBitSet(pos.GetCurrentSide().king));
        Square whiteRook(FirstBitSet(pos.GetCurrentSide().rooks));
        Square weakKing(FirstBitSet(pos.GetOpponentSide().king));
        Square blackRook(FirstBitSet(pos.GetOpponentSide().rooks));

        // right skewer
        if (weakKing.Rank() == blackRook.Rank() && // losing rook and king in same line
            weakKing.File() >= blackRook.File() + 3 && // losing king can't protect rook
            whiteRook.File() >= weakKing.File() + 2 && // losing king can't protect against skewer
            (strongKing.Rank() != weakKing.Rank() || strongKing.File() > whiteRook.File()) && // strong king isn't blocking skewer
            (strongKing.File() != whiteRook.File() || strongKing.Rank() < std::min(whiteRook.Rank(), weakKing.Rank()) || strongKing.Rank() > std::max(whiteRook.Rank(), weakKing.Rank())) // strong king isn't blocking rook from moving
            )
        {
            outScore = KnownWinValue;
            return true;
        }

        // left skewer
        if (weakKing.Rank() == blackRook.Rank() && // losing rook and king in same line
            weakKing.File() + 3 <= blackRook.File() && // losing king can't protect rook
            whiteRook.File() + 2 <= weakKing.File() && // losing king can't protect against skewer
            (strongKing.Rank() != weakKing.Rank() || strongKing.File() < whiteRook.File()) && // strong king isn't blocking skewer
            (strongKing.File() != whiteRook.File() || strongKing.Rank() < std::min(whiteRook.Rank(), weakKing.Rank()) || strongKing.Rank() > std::max(whiteRook.Rank(), weakKing.Rank())) // strong king isn't blocking rook from moving
            )
        {
            outScore = KnownWinValue;
            return true;
        }

        // top skewer
        if (weakKing.File() == blackRook.File() && // losing rook and king in same line
            weakKing.Rank() >= blackRook.Rank() + 3 && // losing king can't protect rook
            whiteRook.Rank() >= weakKing.Rank() + 2 && // losing king can't protect against skewer
            (strongKing.File() != weakKing.File() || strongKing.Rank() > whiteRook.Rank()) && // strong king isn't blocking skewer
            (strongKing.Rank() != whiteRook.Rank() || strongKing.File() < std::min(whiteRook.File(), weakKing.File()) || strongKing.File() > std::max(whiteRook.File(), weakKing.File())) // strong king isn't blocking rook from moving
            )
        {
            outScore = KnownWinValue;
            return true;
        }

        // bottom skewer
        if (weakKing.File() == blackRook.File() && // losing rook and king in same line
            weakKing.Rank() + 3 <= blackRook.Rank() && // losing king can't protect rook
            whiteRook.Rank() + 2 <= weakKing.Rank() && // losing king can't protect against skewer
            (strongKing.File() != weakKing.File() || strongKing.Rank() < whiteRook.Rank()) && // strong king isn't blocking skewer
            (strongKing.Rank() != whiteRook.Rank() || strongKing.File() < std::min(whiteRook.File(), weakKing.File()) || strongKing.File() > std::max(whiteRook.File(), weakKing.File())) // strong king isn't blocking rook from moving
            )
        {
            outScore = KnownWinValue;
            return true;
        }

        // TODO rook skewer at the edge
        // TODO mate on edge (black king blocked by white king)

        outScore = 0;
        return true;
    }

    return false;
}

static bool EvaluateEndgame_KQvKQ(const Position& pos, int32_t& outScore, int32_t& outScale)
{
    UNUSED(outScale);

    ASSERT(pos.Whites().pawns == 0 && pos.Whites().bishops == 0 && pos.Whites().knights == 0 && pos.Whites().rooks == 0 && pos.Whites().queens > 0);
    ASSERT(pos.Blacks().pawns == 0 && pos.Blacks().bishops == 0 && pos.Blacks().knights == 0 && pos.Blacks().rooks == 0 && pos.Blacks().queens > 0);

    if (pos.Whites().queens.Count() == 1 && pos.Blacks().queens.Count() == 1)
    {
        if (pos.Whites().GetKingSquare().EdgeDistance() > 0 &&
            pos.Blacks().GetKingSquare().EdgeDistance() > 0)
        {
            outScore = 0;
            return true;
        }
    }

    return false;
}

// Rook+Pawn vs. Rook
static bool EvaluateEndgame_KRPvKR(const Position& pos, int32_t& outScore, int32_t& outScale)
{
    UNUSED(outScale);

    ASSERT(pos.Whites().pawns != 0 && pos.Whites().bishops == 0 && pos.Whites().knights == 0 && pos.Whites().rooks != 0 && pos.Whites().queens == 0);
    ASSERT(pos.Blacks().pawns == 0 && pos.Blacks().bishops == 0 && pos.Blacks().knights == 0 && pos.Blacks().rooks != 0 && pos.Blacks().queens == 0);

    // KRP vs KR
    if (pos.Whites().rooks.Count() == 1 && pos.Whites().pawns.Count() == 1 && pos.Blacks().rooks.Count() == 1)
    {
        const Square strongKing(FirstBitSet(pos.Whites().king));
        const Square strongRook(FirstBitSet(pos.Whites().rooks));
        const Square strongPawn(FirstBitSet(pos.Whites().pawns));
        const Square weakKing(FirstBitSet(pos.Blacks().king));
        const Square weakRook(FirstBitSet(pos.Blacks().rooks));

        const Square queeningSquare(strongPawn.File(), 7);

        // Lucena position
        if (strongPawn.File() > 0 && strongPawn.File() < 7 &&   // pawn on files B-G
            strongPawn.Rank() >= 6 &&
            strongKing.File() == strongPawn.File() && strongKing.Rank() > strongPawn.Rank() && // king in front of pawn
            ((strongKing.File() < strongRook.File() && strongRook.File() < weakKing.File()) ||
             (strongKing.File() > strongRook.File() && strongRook.File() > weakKing.File())))
        {
            outScore = KnownWinValue;
            return true;
        }

        // Philidor position
        if (Square::Distance(weakKing, queeningSquare) <= 1 &&
            strongPawn.Rank() < 5 &&
            strongKing.Rank() < 5 &&
            weakRook.Rank() == 5)
        {
            outScore = 0;
            return true;
        }
    }

    return false;
}

// Queen vs. Rook+Pawn
static bool EvaluateEndgame_KQvKRP(const Position& pos, int32_t& outScore, int32_t& outScale)
{
    UNUSED(outScale);

    ASSERT(pos.Whites().pawns == 0 && pos.Whites().bishops == 0 && pos.Whites().knights == 0 && pos.Whites().rooks == 0 && pos.Whites().queens != 0);
    ASSERT(pos.Blacks().pawns != 0 && pos.Blacks().bishops == 0 && pos.Blacks().knights == 0 && pos.Blacks().rooks != 0 && pos.Blacks().queens == 0);

    if (pos.Whites().queens.Count() == 1 && pos.Blacks().rooks.Count() == 1 && pos.Blacks().pawns.Count() == 1)
    {
        const Square strongKing(FirstBitSet(pos.Whites().king));
        const Square weakKing(FirstBitSet(pos.Blacks().king));
        const Square queenSquare(FirstBitSet(pos.Whites().queens));
        const Square rookSquare(FirstBitSet(pos.Blacks().rooks));
        const Square pawnSquare(FirstBitSet(pos.Blacks().pawns));

        if (pawnSquare.Rank() == 6 &&
            pawnSquare.File() > 0 && pawnSquare.File() < 7 &&
            Square::Distance(pawnSquare, weakKing) <= 1 &&
            weakKing.Rank() > 5 &&
            strongKing.Rank() < 5 &&
            (Bitboard::GetPawnAttacks<Black>(pawnSquare) & rookSquare.GetBitboard()))
        {
            outScore = 0;
            return true;
        }

        // pawn on b/g file
        // king behind pawn
        // rook protected by the pawn
        if ((pawnSquare.File() == 1 || pawnSquare.File() == 6) &&
            (weakKing.South() == pawnSquare || weakKing.File() == 0 || weakKing.File() == 7) &&
            weakKing.Rank() == pawnSquare.Rank() + 1 &&
            Square::Distance(pawnSquare, weakKing) <= 1 &&
            strongKing.Rank() < rookSquare.Rank() &&
            IsAscendingOrDescending(pawnSquare.File(), rookSquare.File(), strongKing.File()) &&
            (Bitboard::GetPawnAttacks<Black>(pawnSquare) & rookSquare.GetBitboard()))
        {
            outScore = 0;
            return true;
        }

        if (pawnSquare == Square_a6 &&
            weakKing.Rank() > 5 && weakKing.File() == 0 &&
            strongKing.File() > 1 && strongKing.Rank() < 5 &&
            rookSquare.File() == 1 && rookSquare.Rank() >= 4)
        {
            outScore = 0;
            return true;
        }

        if (pawnSquare == Square_h6 &&
            weakKing.Rank() > 5 && weakKing.File() == 7 &&
            strongKing.File() < 6 && strongKing.Rank() < 5 &&
            rookSquare.File() == 6 && rookSquare.Rank() >= 4)
        {
            outScore = 0;
            return true;
        }
    }

    return false;
}

void InitEndgame()
{
    KPKEndgame::Init();

    // clear map
    for (uint32_t i = 0; i < MaterialMask_MAX; ++i)
    {
        s_endgameEvaluationMap[i] = UINT8_MAX;
    }

    // Rook/Queen + anything vs. lone king
    {
        const uint8_t functionIndex = s_numEndgameEvaluationFunctions++;
        s_endgameEvaluationFunctions[functionIndex] = EvaluateEndgame_KXvK;

        for (uint32_t mask = 0; mask < MaterialMask_WhitesMAX; ++mask)
        {
            if (mask & (MaterialMask_WhiteRook | MaterialMask_WhiteQueen))
            {
                RegisterEndgame((MaterialMask)mask, functionIndex);
            }
        }
    }

    RegisterEndgame(MaterialMask_WhiteKnight, EvaluateEndgame_KNvK);
    RegisterEndgame(MaterialMask_WhiteBishop, EvaluateEndgame_KBvK);
    RegisterEndgame(MaterialMask_WhiteBishop|MaterialMask_BlackKnight, EvaluateEndgame_KBvK);
    RegisterEndgame(MaterialMask_WhiteBishop|MaterialMask_WhiteKnight, EvaluateEndgame_KNBvK);
    RegisterEndgame(MaterialMask_WhiteBishop|MaterialMask_WhiteKnight|MaterialMask_WhitePawn, EvaluateEndgame_KNBvK);
    RegisterEndgame(MaterialMask_WhiteBishop|MaterialMask_WhitePawn, EvaluateEndgame_KBPvK);
    RegisterEndgame(MaterialMask_WhiteKnight|MaterialMask_WhitePawn, EvaluateEndgame_KNPvK);
    RegisterEndgame(MaterialMask_WhitePawn, EvaluateEndgame_KPvK);
    RegisterEndgame(MaterialMask_WhiteKnight|MaterialMask_BlackKnight, EvaluateEndgame_KNvKN);
    RegisterEndgame(MaterialMask_WhitePawn|MaterialMask_BlackPawn, EvaluateEndgame_KPvKP);
    RegisterEndgame(MaterialMask_WhiteQueen|MaterialMask_BlackPawn, EvaluateEndgame_KQvKP);
    RegisterEndgame(MaterialMask_WhiteRook|MaterialMask_BlackPawn, EvaluateEndgame_KRvKP);
    RegisterEndgame(MaterialMask_WhiteBishop|MaterialMask_BlackBishop, EvaluateEndgame_KBvKB);
    RegisterEndgame(MaterialMask_WhiteRook|MaterialMask_BlackKnight, EvaluateEndgame_KRvKN);
    RegisterEndgame(MaterialMask_WhiteRook|MaterialMask_BlackBishop, EvaluateEndgame_KRvKB);
    RegisterEndgame(MaterialMask_WhiteQueen|MaterialMask_BlackRook, EvaluateEndgame_KQvKR);
    RegisterEndgame(MaterialMask_WhiteQueen|MaterialMask_BlackKnight, EvaluateEndgame_KQvKN);
    RegisterEndgame(MaterialMask_WhiteQueen|MaterialMask_BlackKnight|MaterialMask_BlackPawn, EvaluateEndgame_KQvKNP);
    RegisterEndgame(MaterialMask_WhiteQueen|MaterialMask_BlackBishop|MaterialMask_BlackKnight, EvaluateEndgame_KQvKBN);
    RegisterEndgame(MaterialMask_WhiteRook|MaterialMask_BlackRook, EvaluateEndgame_KRvKR);
    RegisterEndgame(MaterialMask_WhiteQueen|MaterialMask_BlackQueen, EvaluateEndgame_KQvKQ);
    RegisterEndgame(MaterialMask_WhiteRook|MaterialMask_WhitePawn|MaterialMask_BlackRook, EvaluateEndgame_KRPvKR);
    RegisterEndgame(MaterialMask_WhiteQueen|MaterialMask_BlackRook|MaterialMask_BlackPawn, EvaluateEndgame_KQvKRP);
}

#ifdef COLLECT_ENDGAME_STATISTICS
#include <unordered_map>
static std::mutex s_matKeyOccurencesMutex;
static std::unordered_map<MaterialKey, uint64_t> s_matKeyOccurences;
#endif // COLLECT_ENDGAME_STATISTICS

bool EvaluateEndgame(const Position& pos, int32_t& outScore, int32_t& outScale)
{
    MaterialMask materialMask = BuildMaterialMask(pos);
    ASSERT(materialMask < MaterialMask_MAX);

    // King vs King
    if (materialMask == 0)
    {
        outScore = 0;
        return true;
    }

#ifdef COLLECT_ENDGAME_STATISTICS
    if (pos.GetNumPieces() <= 6)
    {
        std::unique_lock<std::mutex> lock(s_matKeyOccurencesMutex);
        s_matKeyOccurences[pos.GetMaterialKey()]++;
    }
#endif // COLLECT_ENDGAME_STATISTICS

    // find registered function (regular)
    uint8_t evaluationFuncIndex = s_endgameEvaluationMap[materialMask];
    if (evaluationFuncIndex != UINT8_MAX)
    {
        const EndgameEvaluationFunc& func = s_endgameEvaluationFunctions[evaluationFuncIndex];
        outScore = InvalidValue;
        const bool result = func(pos, outScore, outScale);
        if (result) { ASSERT(outScore != InvalidValue); }
        return result;
    }

    // find registered function (flipped color)
    evaluationFuncIndex = s_endgameEvaluationMap[FlipColor(materialMask)];
    if (evaluationFuncIndex != UINT8_MAX)
    {
        const EndgameEvaluationFunc& func = s_endgameEvaluationFunctions[evaluationFuncIndex];
        int32_t score = InvalidValue;
        const bool result = func(pos.SwappedColors(), score, outScale);
        if (result) { ASSERT(score != InvalidValue); }
        outScore = -score;
        return result;
    }

    return false;
}

#ifdef COLLECT_ENDGAME_STATISTICS
void PrintEndgameStatistics()
{
    std::unique_lock<std::mutex> lock(s_matKeyOccurencesMutex);
    for (const auto& iter : s_matKeyOccurences)
    {
        std::cout << iter.first.ToString() << " " << iter.second << std::endl;
    }
}
#endif // COLLECT_ENDGAME_STATISTICS
