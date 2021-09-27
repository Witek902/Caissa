#include "Endgame.hpp"
#include "Evaluate.hpp"
#include "Position.hpp"
#include "Move.hpp"
#include "PackedNeuralNetwork.hpp"

#include <bitset>
#include <vector>

static nn::PackedNeuralNetwork g_pawnsEndgameNeuralNetwork;

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
    Result classify(const std::vector<KPKPosition>& db);

    Color sideToMove : 1;
    Result result : 3;
    Square kingSquare[2];
    Square pawnSquare;

    // TODO: distance to queening?
};

static_assert(sizeof(KPKPosition) <= 4, "Invalid KPKPosition size");

bool Probe(Square whiteKingSq, Square pawnSq, Square blackKingSq, Color sideToMove)
{
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
                repeat |= (db[i] == UNKNOWN && db[i].classify(db) != UNKNOWN);
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
}

KPKPosition::KPKPosition(uint32_t idx)
{
    // decode position from index
    kingSquare[0] = Square(idx & 0x3F);
    kingSquare[1] = Square((idx >> 6) & 0x3F);
    sideToMove = Color((idx >> 12) & 0x1);
    pawnSquare = Square((idx >> 13) & 0x3, 6u - ((idx >> 15) & 0x7));

    const Bitboard pawnAttacks = Bitboard::GetPawnAttacks(pawnSquare, Color::White);

    // Invalid if two pieces are on the same square or if a king can be captured
    if (Square::Distance(kingSquare[0], kingSquare[1]) <= 1
        || kingSquare[0] == pawnSquare
        || kingSquare[1] == pawnSquare
        || (sideToMove == Color::White && (pawnAttacks & kingSquare[1].GetBitboard())))
    {
        result = INVALID;
    }
    // Win if the pawn can be promoted without getting captured
    else if (sideToMove == Color::White
        && pawnSquare.Rank() == 6
        && kingSquare[0] != pawnSquare.North()
        && (    Square::Distance(kingSquare[1], pawnSquare.North()) > 1
            || (Square::Distance(kingSquare[0], pawnSquare.North()) == 1)))
    {
        result = WIN;
    }
    // Draw if it is stalemate or the black king can capture the pawn
    else if (sideToMove == Color::Black
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

Result KPKPosition::classify(const std::vector<KPKPosition>& db)
{
    const Result Good = sideToMove == Color::White ? WIN : DRAW;
    const Result Bad = sideToMove == Color::White ? DRAW : WIN;

    Result r = INVALID;
    const Bitboard b = Bitboard::GetKingAttacks(kingSquare[(uint32_t)sideToMove]);

    b.Iterate([&](uint32_t square) INLINE_LAMBDA
    {
        r |= sideToMove == Color::White ?
            db[EncodeIndex(Color::Black, kingSquare[1], square, pawnSquare)] :
            db[EncodeIndex(Color::White, square, kingSquare[0], pawnSquare)];
    });

    if (sideToMove == Color::White)
    {
        // single push
        if (pawnSquare.Rank() < 6)
        {
            r |= db[EncodeIndex(Color::Black, kingSquare[1], kingSquare[0], pawnSquare.North())];
        }
        // double push
        if (pawnSquare.Rank() == 1 && pawnSquare.North() != kingSquare[0] && pawnSquare.North() != kingSquare[1])
        {
            r |= db[EncodeIndex(Color::Black, kingSquare[1], kingSquare[0], pawnSquare.North().North())];
        }
    }

    return result = r & Good ? Good : r & UNKNOWN ? UNKNOWN : Bad;
}

} // KPKEndgame

enum MaterialMask : uint32_t
{
    MaterialMask_WhitePawn   = 1 << 0,
    MaterialMask_WhiteKnight = 1 << 1,
    MaterialMask_WhiteBishop = 1 << 2,
    MaterialMask_WhiteRook   = 1 << 3,
    MaterialMask_WhiteQueen  = 1 << 4,

    MaterialMask_BlackPawn   = 1 << 5,
    MaterialMask_BlackKnight = 1 << 6,
    MaterialMask_BlackBishop = 1 << 7,
    MaterialMask_BlackRook   = 1 << 8,
    MaterialMask_BlackQueen  = 1 << 9,
};

INLINE static constexpr uint32_t BuildMaterialMask(
    uint32_t wp, uint32_t wk, uint32_t wb, uint32_t wr, uint32_t wq,
    uint32_t bp, uint32_t bk, uint32_t bb, uint32_t br, uint32_t bq)
{
    uint32_t mask = 0;

    if (wp > 0) mask |= 1 << 0;
    if (wk > 0) mask |= 1 << 1;
    if (wb > 0) mask |= 1 << 2;
    if (wr > 0) mask |= 1 << 3;
    if (wq > 0) mask |= 1 << 4;

    if (bp > 0) mask |= 1 << 5;
    if (bk > 0) mask |= 1 << 6;
    if (bb > 0) mask |= 1 << 7;
    if (br > 0) mask |= 1 << 8;
    if (bq > 0) mask |= 1 << 9;

    return mask;
}

void InitEndgame()
{
    KPKEndgame::Init();

    g_pawnsEndgameNeuralNetwork.Load("D:/DEV/CURRENT/Chess/src/frontend/pawns.nn");
}

bool EvaluateEndgame(const Position& pos, int32_t& outScore)
{
    const int32_t whiteQueens   = pos.Whites().queens.Count();
    const int32_t whiteRooks    = pos.Whites().rooks.Count();
    const int32_t whiteBishops  = pos.Whites().bishops.Count();
    const int32_t whiteKnights  = pos.Whites().knights.Count();
    const int32_t whitePawns    = pos.Whites().pawns.Count();

    const int32_t blackQueens   = pos.Blacks().queens.Count();
    const int32_t blackRooks    = pos.Blacks().rooks.Count();
    const int32_t blackBishops  = pos.Blacks().bishops.Count();
    const int32_t blackKnights  = pos.Blacks().knights.Count();
    const int32_t blackPawns    = pos.Blacks().pawns.Count();

    const uint32_t mask = BuildMaterialMask(
        whitePawns, whiteKnights, whiteBishops, whiteRooks, whiteQueens,
        blackPawns, blackKnights, blackBishops, blackRooks, blackQueens);

    Square whiteKing(FirstBitSet(pos.Whites().king));
    Square blackKing(FirstBitSet(pos.Blacks().king));

    switch (mask)
    {

    // King vs King
    case 0:
    {
        outScore = 0;
        return true;
    }

    // Knight(s) vs King
    case MaterialMask_WhiteKnight:
    {
        if (whiteKnights <= 2)
        {
            // NOTE: there are checkmates with two knights, but they cannot be forced from all positions
            outScore = 0;
            return true;
        }
        else // whiteKnights >= 3
        {
            outScore = whiteKnights > 3 ? KnownWinValue : 0;
            outScore += 8 * (whiteKnights - 3); // prefer keeping the knights
            outScore += (3 - blackKing.AnyCornerDistance()); // push king to corner
            return true;
        }
        break;
    }

    // King vs Knight(s)
    case MaterialMask_BlackKnight:
    {
        if (blackKnights <= 2)
        {
            // NOTE: there are checkmates with two knights, but they cannot be forced from all positions
            outScore = 0;
            return true;
        }
        else // blackKnights >= 3
        {
            outScore = blackKnights > 3 ? -KnownWinValue : 0;
            outScore -= 8 * (blackKnights - 3); // prefer keeping the knights
            outScore -= (3 - whiteKing.AnyCornerDistance()); // push king to corner
            return true;
        }
        break;
    }

    // Bishop(s) vs Knight
    case MaterialMask_WhiteBishop:
    case MaterialMask_WhiteBishop | MaterialMask_BlackKnight:
    {
        const uint32_t numLightSquareBishops = (pos.Whites().bishops & Bitboard::LightSquares()).Count();
        const uint32_t numDarkSquareBishops = (pos.Whites().bishops & Bitboard::DarkSquares()).Count();

        if (blackKnights <= 1 && (numLightSquareBishops == 0 || numDarkSquareBishops == 0))
        {
            outScore = 0;
            return true;
        }
        else if (blackKnights <= 1 && (numLightSquareBishops >= 1 || numDarkSquareBishops >= 1))
        {
            outScore = KnownWinValue;
            if (blackKnights) outScore = 0; // drawish score when opponent have a knight
            outScore += 64 * (whiteBishops - 2); // prefer keeping the bishops on board
            outScore += 8 * (3 - blackKing.AnyCornerDistance()); // push king to corner
            outScore += (7 - Square::Distance(blackKing, whiteKing)); // push kings close
            return true;
        }
        break;
    }

    // Knight vs Bishop(s)
    case MaterialMask_BlackBishop:
    case MaterialMask_BlackBishop | MaterialMask_WhiteKnight:
    {
        const uint32_t numLightSquareBishops = (pos.Blacks().bishops & Bitboard::LightSquares()).Count();
        const uint32_t numDarkSquareBishops = (pos.Blacks().bishops & Bitboard::DarkSquares()).Count();

        if (whiteKnights <= 1 && (numLightSquareBishops == 0 || numDarkSquareBishops == 0))
        {
            outScore = 0;
            return true;
        }
        else if (whiteKnights <= 1 && (numLightSquareBishops >= 1 || numDarkSquareBishops >= 1))
        {
            outScore = -KnownWinValue;
            if (whiteKnights) outScore = 0; // drawish score when opponent have a knight
            outScore -= 64 * (blackBishops - 2); // prefer keeping the bishops on board
            outScore -= 8 * (3 - whiteKing.AnyCornerDistance()); // push king to corner
            outScore -= (7 - Square::Distance(blackKing, whiteKing)); // push kings close
            return true;
        }
        break;
    }

    // Queens/Rooks vs King
    // TODO extend to "Queen/Rook+anything vs King"
    case BuildMaterialMask(0, 0, 0, 1, 0, 0, 0, 0, 0, 0):
    case BuildMaterialMask(0, 0, 0, 0, 1, 0, 0, 0, 0, 0):
    case BuildMaterialMask(0, 0, 0, 1, 1, 0, 0, 0, 0, 0):
    {
        outScore = KnownWinValue + 1000;
        outScore += 8 * (3 - blackKing.EdgeDistance()); // push king to edge
        outScore += (7 - Square::Distance(blackKing, whiteKing)); // push kings close
        return true;
    }

    // King vs Queens/Rooks
    // TODO extend to "King vs Queen/Rook+anything"
    case BuildMaterialMask(0, 0, 0, 0, 0, 0, 0, 0, 1, 0):
    case BuildMaterialMask(0, 0, 0, 0, 0, 0, 0, 0, 0, 1):
    case BuildMaterialMask(0, 0, 0, 0, 0, 0, 0, 0, 1, 1):
    {
        outScore = -(KnownWinValue + 1000);
        outScore -= 8 * (3 - whiteKing.EdgeDistance()); // push king to edge
        outScore -= (7 - Square::Distance(blackKing, whiteKing)); // push kings close
        return true;
    }

    // Knight+Bishop vs King
    case BuildMaterialMask(0, 1, 1, 0, 0, 0, 0, 0, 0, 0):
    {
        // push king to 'right' board corner
        const Square kingSquare = (pos.Whites().bishops & Bitboard::DarkSquares()) ? blackKing : blackKing.FlippedFile();

        outScore = KnownWinValue;
        outScore += 8 * (7 - kingSquare.DarkCornerDistance()); // push king to corner
        outScore += (7 - Square::Distance(blackKing, whiteKing)); // push kings close
        return true;
    }

    // King vs Knight+Bishop
    case BuildMaterialMask(0, 0, 0, 0, 0, 0, 1, 1, 0, 0):
    {
        // push king to 'right' board corner
        const Square kingSquare = (pos.Blacks().bishops & Bitboard::DarkSquares()) ? whiteKing : whiteKing.FlippedFile();

        outScore = -KnownWinValue;
        outScore -= 8 * (7 - kingSquare.DarkCornerDistance()); // push king to corner
        outScore -= (7 - Square::Distance(blackKing, whiteKing)); // push kings close
        return true;
    }

    // Pawn vs King
    case MaterialMask_WhitePawn:
    {
        if (whitePawns == 1)
        {
            Square strongKingSq = whiteKing;
            Square weakKingSq = blackKing;
            Square pawnSquare = FirstBitSet(pos.Whites().pawns);

            if (pawnSquare.File() >= 4)
            {
                strongKingSq = strongKingSq.FlippedFile();
                weakKingSq = weakKingSq.FlippedFile();
                pawnSquare = pawnSquare.FlippedFile();
            }

            if (!KPKEndgame::Probe(strongKingSq, pawnSquare, weakKingSq, pos.GetSideToMove()))
            {
                // bitbase draw
                outScore = 0;
                return true;
            }

            outScore = KnownWinValue;
            outScore += 8 * pawnSquare.Rank();
            outScore += 7 - std::max(0, (int32_t)Square::Distance(pawnSquare, strongKingSq) - 1); // push kings close to pawn
            return true;
        }
        break;
    }

    // King vs Pawn
    case MaterialMask_BlackPawn:
    {
        if (blackPawns == 1)
        {
            // swap black/white, because KPKEndgame::Probe assumes that white is up a pawn
            Square strongKingSq = blackKing.FlippedRank();
            Square weakKingSq = whiteKing.FlippedRank();
            Square pawnSquare = Square(FirstBitSet(pos.Blacks().pawns)).FlippedRank();

            if (pawnSquare.File() >= 4)
            {
                strongKingSq = strongKingSq.FlippedFile();
                weakKingSq = weakKingSq.FlippedFile();
                pawnSquare = pawnSquare.FlippedFile();
            }

            if (!KPKEndgame::Probe(strongKingSq, pawnSquare, weakKingSq, GetOppositeColor(pos.GetSideToMove())))
            {
                // bitbase draw
                outScore = 0;
                return true;
            }

            outScore = -KnownWinValue;
            outScore -= 8 * pawnSquare.Rank();
            outScore -= 7 - std::max(0, (int32_t)Square::Distance(pawnSquare, strongKingSq) - 1); // push kings close to pawn
            return true;
        }
        break;
    }

    // TODO WIP
    /*
    // Pawns vs Pawns
    case MaterialMask_WhitePawn|MaterialMask_BlackPawn:
    {
        if (g_pawnsEndgameNeuralNetwork.IsValid())
        {
            Square kingA = whiteKing;
            Square kingB = blackKing;
            Bitboard pawnsA = pos.Whites().pawns;
            Bitboard pawnsB = pos.Blacks().pawns;

            if (pos.GetSideToMove() == Color::Black)
            {
                kingA = blackKing.FlippedRank();
                kingB = whiteKing.FlippedRank();
                pawnsA = pos.Blacks().pawns.FlippedVertically();
                pawnsB = pos.Whites().pawns.FlippedVertically();
            }

            if (kingA.File() >= 4)
            {
                kingA = kingA.FlippedFile();
                kingB = kingB.FlippedFile();
                pawnsA = pawnsA.MirroredHorizontally();
                pawnsB = pawnsB.MirroredHorizontally();
            }

            constexpr uint32_t maxFeatures = 18; // kings + max pawns
            uint32_t features[18];

            uint32_t numFeatures = 0;
            uint32_t inputOffset = 0;

            // white king
            {
                const uint32_t whiteKingIndex = 4 * kingA.Rank() + kingA.File();
                features[numFeatures++] = whiteKingIndex;
                inputOffset += 32;
            }

            // black king
            {
                features[numFeatures++] = inputOffset + kingB.Index();
                inputOffset += 64;
            }

            {
                for (uint32_t i = 0; i < 48u; ++i)
                {
                    const uint32_t squreIndex = i + 8u;
                    if ((pawnsA >> squreIndex) & 1) features[numFeatures++] = inputOffset + i;
                }
                inputOffset += 48;
            }

            {
                for (uint32_t i = 0; i < 48u; ++i)
                {
                    const uint32_t squreIndex = i + 8u;
                    if ((pawnsB >> squreIndex) & 1) features[numFeatures++] = inputOffset + i;
                }
                inputOffset += 48;
            }

            ASSERT(numFeatures >= 4);
            ASSERT(numFeatures <= maxFeatures);

            const int32_t rawNetworkOutput = g_pawnsEndgameNeuralNetwork.Run(numFeatures, features);
            const float winProbability = (float)rawNetworkOutput / (float)nn::WeightScale / (float)nn::OutputScale;
            const float pawnsValue = WinProbabilityToPawns(winProbability);

            outScore = (int32_t)(0.5f + 100.0f * std::clamp(pawnsValue, -64.0f, 64.0f));

            if (pos.GetSideToMove() == Color::Black) outScore = -outScore;

            //return true;
        }
        break;
    }
    */

    }

    return false;
}