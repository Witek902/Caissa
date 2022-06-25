#include "Endgame.hpp"
#include "Evaluate.hpp"
#include "Position.hpp"
#include "Material.hpp"
#include "Move.hpp"
#include "MoveList.hpp"
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

Result KPKPosition::Classify(const std::vector<KPKPosition>& db)
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
    MaterialMask_WhitePawn      = 1 << 0,
    MaterialMask_WhiteKnight    = 1 << 1,
    MaterialMask_WhiteBishop    = 1 << 2,
    MaterialMask_WhiteRook      = 1 << 3,
    MaterialMask_WhiteQueen     = 1 << 4,

    MaterialMask_BlackPawn      = 1 << 5,
    MaterialMask_BlackKnight    = 1 << 6,
    MaterialMask_BlackBishop    = 1 << 7,
    MaterialMask_BlackRook      = 1 << 8,
    MaterialMask_BlackQueen     = 1 << 9,

    MaterialMask_MAX            = 1 << 10,
    MaterialMask_WhitesMAX      = MaterialMask_BlackPawn,
};

INLINE constexpr MaterialMask operator | (MaterialMask a, MaterialMask b)
{
    return static_cast<MaterialMask>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}

INLINE static MaterialMask BuildMaterialMask(const Position& pos)
{
    MaterialMask mask = (MaterialMask)0;

    if (pos.Whites().pawns)     mask = mask | MaterialMask_WhitePawn;
    if (pos.Whites().knights)   mask = mask | MaterialMask_WhiteKnight;
    if (pos.Whites().bishops)   mask = mask | MaterialMask_WhiteBishop;
    if (pos.Whites().rooks)     mask = mask | MaterialMask_WhiteRook;
    if (pos.Whites().queens)    mask = mask | MaterialMask_WhiteQueen;

    if (pos.Blacks().pawns)     mask = mask | MaterialMask_BlackPawn;
    if (pos.Blacks().knights)   mask = mask | MaterialMask_BlackKnight;
    if (pos.Blacks().bishops)   mask = mask | MaterialMask_BlackBishop;
    if (pos.Blacks().rooks)     mask = mask | MaterialMask_BlackRook;
    if (pos.Blacks().queens)    mask = mask | MaterialMask_BlackQueen;

    return mask;
}

INLINE static MaterialMask FlipColor(const MaterialMask mask)
{
    return MaterialMask((mask >> 5) | ((mask & 0x1F) << 5));
}

using EndgameEvaluationFunc = bool (*)(const Position&, int32_t&);

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
static bool EvaluateEndgame_KXvK(const Position& pos, int32_t& outScore)
{
    ASSERT(pos.Blacks().OccupiedExcludingKing().Count() == 0);

    const Square strongKing(FirstBitSet(pos.Whites().king));
    const Square weakKing(FirstBitSet(pos.Blacks().king));

    if (pos.GetSideToMove() == Color::Black)
    {
        MoveList moves;
        pos.GenerateKingMoveList(moves);

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

    outScore = KnownWinValue;
    outScore += c_queenValue.eg * pos.Whites().queens.Count();
    outScore += c_rookValue.eg * pos.Whites().rooks.Count();
    outScore += c_bishopValue.eg * pos.Whites().bishops.Count();
    outScore += c_knightValue.eg * pos.Whites().knights.Count();
    outScore += c_pawnValue.eg * pos.Whites().pawns.Count();
    outScore += 8 * (3 - weakKing.EdgeDistance()); // push king to edge
    outScore += (7 - Square::Distance(weakKing, strongKing)); // push kings close
    outScore = std::clamp(outScore, -TablebaseWinValue + 1, TablebaseWinValue - 1);

    // TODO put rook on a rank/file that limits king movement

    return true;
}

// knight(s) vs. lone king
static bool EvaluateEndgame_KNvK(const Position& pos, int32_t& outScore)
{
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
        outScore = numKnights > 3 ? KnownWinValue : 0;
        outScore += c_knightValue.eg * (numKnights - 3); // prefer keeping the knights
        outScore += 8 * (3 - weakKing.AnyCornerDistance()); // push king to corner
        outScore += (7 - Square::Distance(weakKing, strongKing)); // push kings close
    }

    return true;
}

// bishop(s) vs. lone king
static bool EvaluateEndgame_KBvK(const Position& pos, int32_t& outScore)
{
    ASSERT(pos.Blacks().pawns == 0);
    ASSERT(pos.Blacks().bishops == 0);
    ASSERT(pos.Blacks().rooks == 0);
    ASSERT(pos.Blacks().queens == 0);

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
    else if (blackKnights <= 1 && (numLightSquareBishops >= 1 || numDarkSquareBishops >= 1))
    {
        outScore = KnownWinValue;
        if (blackKnights) outScore = 0; // drawish score when opponent have a knight
        outScore += 64 * (whiteBishops - 2); // prefer keeping the bishops on board
        outScore += 8 * (3 - weakKing.AnyCornerDistance()); // push king to corner
        outScore += (7 - Square::Distance(weakKing, strongKing)); // push kings close
        return true;
    }
    return false;
}

// knight + bishop vs. lone king
static bool EvaluateEndgame_KNBvK(const Position& pos, int32_t& outScore)
{
    ASSERT(pos.Whites().pawns == 0);
    ASSERT(pos.Whites().bishops > 0);
    ASSERT(pos.Whites().knights > 0);
    ASSERT(pos.Whites().rooks == 0);
    ASSERT(pos.Whites().queens == 0);

    ASSERT(pos.Blacks().OccupiedExcludingKing().Count() == 0);

    const Square strongKing(FirstBitSet(pos.Whites().king));
    const Square weakKing(FirstBitSet(pos.Blacks().king));
    const int32_t whiteBishops = pos.Whites().bishops.Count();
    const int32_t whiteKnights = pos.Whites().knights.Count();

    // push king to 'right' board corner
    const Square kingSquare = (pos.Whites().bishops & Bitboard::DarkSquares()) ? weakKing : weakKing.FlippedFile();

    outScore = KnownWinValue;
    outScore += c_knightValue.eg * (whiteBishops - 1); // prefer keeping the knights
    outScore += c_bishopValue.eg * (whiteKnights - 1); // prefer keeping the knights
    outScore += 8 * (7 - kingSquare.DarkCornerDistance()); // push king to corner
    outScore += (7 - Square::Distance(weakKing, strongKing)); // push kings close

    return true;
}

// pawn(s) vs. lone king
static bool EvaluateEndgame_KPvK(const Position& pos, int32_t& outScore)
{
    ASSERT(pos.Whites().pawns.Count() > 0);
    ASSERT(pos.Blacks().OccupiedExcludingKing().Count() == 0);

    const Square strongKing(FirstBitSet(pos.Whites().king));
    const Square weakKing(FirstBitSet(pos.Blacks().king));
    const int32_t numPawns = pos.Whites().pawns.Count();

    Square strongKingSq = strongKing;
    Square weakKingSq = weakKing;
    Square pawnSquare = FirstBitSet(pos.Whites().pawns);

    if (numPawns == 1)
    {
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

        ASSERT(pawnSquare.Rank() < 7);
        Square keySquare = Square(pawnSquare.File(), pawnSquare.Rank() + 1);
        if (pawnSquare.Rank() < 6) keySquare = Square(pawnSquare.Rank() + 2, pawnSquare.File());

        outScore = KnownWinValue + c_pawnValue.eg;
        outScore += 8 * pawnSquare.Rank();
        outScore -= (int32_t)Square::Distance(keySquare, strongKingSq); // put strong king in front of pawn
        outScore += (int32_t)Square::Distance(pawnSquare, weakKingSq); // try to capture pawn
        return true;
    }
    else if (numPawns == 2)
    {
        const Square secondPawnSquare = FirstBitSet(pos.Whites().pawns & ~pawnSquare.GetBitboard());
        const bool areConnected = (Square::Distance(pawnSquare, secondPawnSquare) <= 1) && (pawnSquare.File() != secondPawnSquare.File());

        // connected passed pawns
        if ((areConnected && pos.GetSideToMove() == Color::White) ||
            (Square::Distance(pawnSquare, secondPawnSquare) > 5 && pos.GetSideToMove() == Color::White))
        {
            outScore = KnownWinValue + 2 * c_pawnValue.eg;
            outScore += 8 * pawnSquare.Rank();
            outScore += 7 - std::max(0, (int32_t)Square::Distance(pawnSquare, strongKingSq) - 1); // push kings close to pawn
            outScore += std::max(0, (int32_t)Square::Distance(pawnSquare, weakKingSq) - 1); // push kings close to pawn
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
static bool EvaluateEndgame_KPvKP(const Position& pos, int32_t& outScore)
{
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

        if (whitePawn.Rank() < 4 || whitePawn.File() == 0)
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

        const uint32_t whiteToQueen = 7 - whitePawn.Rank();
        const uint32_t blackToQueen = blackPawn.Rank();

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

        // piece square tables generated with "validate endgame" utility
        static const int16_t whitePawnPsqt[] =
        {
             0,      0,      0,      0,      0,      0,      0,      0,
          -116,    -94,   -107,   -116,      0,      0,      0,      0,
          -123,   -122,   -139,   -147,      0,      0,      0,      0,
           -47,    -52,    -73,    -76,      0,      0,      0,      0,
            44,     21,     31,     17,      0,      0,      0,      0,
           157,    133,    137,    141,      0,      0,      0,      0,
           -34,    -11,     -7,     -7,      0,      0,      0,      0,
        };
        static const int16_t whiteKingPsqt[] =
        {
           -38,     11,     61,     95,    103,     79,     24,    -36,
           -11,     25,     73,    105,    106,     84,     26,    -25,
           -16,     20,     67,     97,    100,     77,     19,    -31,
           -41,    -22,      9,     30,     30,      8,    -39,    -71,
           -52,    -44,    -27,    -27,    -23,    -47,    -89,   -109,
           -52,    -47,    -45,    -51,    -50,    -85,   -131,   -143,
           -37,    -25,    -11,    -35,    -45,    -88,   -145,   -161,
           -38,    -20,    -15,    -42,    -67,   -105,   -146,   -173,
        };
        static const int16_t blackPawnPsqt[] =
        {
             0,      0,      0,      0,      0,      0,      0,      0,
          -397,   -402,   -356,   -362,   -358,   -353,   -388,   -411,
          -152,   -131,   -123,   -113,   -141,   -171,   -177,   -182,
            12,     58,     61,     66,     34,    -21,    -26,    -38,
            98,     93,    161,    128,    123,    120,     75,     86,
           101,     85,    114,    151,    177,    192,    194,    179,
            77,     35,     45,    101,    143,    188,    204,    178,
        };
        static const int16_t blackKingPsqt[] =
        {
            47,     29,     15,     15,     10,     14,     37,     62,
            39,     16,     -8,     -9,    -17,     -6,     32,     58,
            25,     11,      1,    -12,    -15,     -5,     21,     45,
            -9,    -23,    -35,    -36,    -31,    -18,     19,     52,
           -58,    -91,   -111,    -96,    -70,    -33,     20,     57,
          -142,   -230,   -306,   -201,   -114,    -44,     23,     65,
          -143,   -206,   -262,   -180,   -101,    -36,     31,     68,
          -120,   -175,   -194,   -140,    -72,     -8,     55,     97,
        };

        outScore =
            whitePawnPsqt[whitePawn.mIndex] +
            whiteKingPsqt[whiteKing.mIndex] +
            blackPawnPsqt[blackPawn.mIndex] +
            blackKingPsqt[blackKing.mIndex];

        outScore += 16 * (
            Square::Distance(blackKing, whitePawn) + Square::Distance(blackKing, blackPawn) -
            Square::Distance(whiteKing, whitePawn) - Square::Distance(whiteKing, blackPawn));

        return true;
    }

    return false;
}

// bishop(s) + pawn(s) vs. lone king
static bool EvaluateEndgame_KBPvK(const Position& pos, int32_t& outScore)
{
    ASSERT(pos.Whites().pawns > 0);
    ASSERT(pos.Whites().bishops > 0);
    ASSERT(pos.Whites().knights == 0);
    ASSERT(pos.Whites().rooks == 0);
    ASSERT(pos.Whites().queens == 0);

    ASSERT(pos.Blacks().OccupiedExcludingKing().Count() == 0);

    const Square strongKing(FirstBitSet(pos.Whites().king));
    const Square weakKing(FirstBitSet(pos.Blacks().king));

    // if all pawns are on A/H file and we have a wrong bishop, then it's a draw
    {
        if (((pos.Whites().pawns & ~Bitboard::FileBitboard<0>()) == 0) &&
            ((pos.Whites().bishops & Bitboard::LightSquares()) == 0) &&
            (Square::Distance(weakKing, Square_a8) <= 1))
        {
            outScore = 0;
            return true;
        }
        if (((pos.Whites().pawns & ~Bitboard::FileBitboard<7>()) == 0) &&
            ((pos.Whites().bishops & Bitboard::DarkSquares()) == 0) &&
            (Square::Distance(weakKing, Square_h8) <= 1))
        {
            outScore = 0;
            return true;
        }
    }

    // if we have a "good" bishop and weak king can't easily capture pawn, then it's a win
    if (pos.Whites().pawns.Count() == 1)
    {
        const Square pawnSquare(FirstBitSet(pos.Whites().pawns));
        const Square bishopSquare(FirstBitSet(pos.Whites().bishops));
        const Square promotionSquare(pawnSquare.File(), 7);

        const bool bishopOnLightSquare = pos.Whites().bishops & Bitboard::LightSquares();
        const bool promotionOnLightSquare = promotionSquare.GetBitboard() & Bitboard::LightSquares();

        const uint32_t blackToMove = pos.GetSideToMove() == Color::Black;

        if (bishopOnLightSquare == promotionOnLightSquare &&
            Square::Distance(weakKing, pawnSquare) > 2 + blackToMove &&
            Square::Distance(strongKing, bishopSquare) > 1 && !bishopSquare.IsCorner())
        {
            outScore = KnownWinValue;
            outScore += 16 * pawnSquare.Rank();
            outScore += Square::Distance(weakKing, pawnSquare);
            outScore -= Square::Distance(strongKing, pawnSquare);
            return true;
        }
    }

    return false;
}

// Queen vs. Pawn
static bool EvaluateEndgame_KQvKP(const Position& pos, int32_t& outScore)
{
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
            if (pos.GetSideToMove() == Color::White && !pos.IsInCheck(Color::White))
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

// Rook vs. Pawn
static bool EvaluateEndgame_KRvKP(const Position& pos, int32_t& outScore)
{
    ASSERT(pos.Whites().pawns == 0 && pos.Whites().bishops == 0 && pos.Whites().knights == 0 && pos.Whites().rooks != 0 && pos.Whites().queens == 0);
    ASSERT(pos.Blacks().pawns != 0 && pos.Blacks().bishops == 0 && pos.Blacks().knights == 0 && pos.Blacks().rooks == 0 && pos.Blacks().queens == 0);

    if (pos.Whites().rooks.Count() == 1 && pos.Blacks().pawns.Count() == 1)
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
            (Square::Distance(weakKing, weakPawn) >= 6u && weakPawn.Rank() > 1) ||
            (Square::Distance(weakKing, weakPawn) >= 4u && weakKing.Rank() >= weakPawn.Rank() && weakPawn.Rank() > 1) ||
            (Square::Distance(weakKing, weakPawn) >= 3u && weakKing.Rank() >= weakPawn.Rank() && weakPawn.Rank() > 3))
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
static bool EvaluateEndgame_KRvKN(const Position& pos, int32_t& outScore)
{
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
static bool EvaluateEndgame_KRvKB(const Position& pos, int32_t& outScore)
{
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
static bool EvaluateEndgame_KQvKR(const Position& pos, int32_t& outScore)
{
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
static bool EvaluateEndgame_KQvKN(const Position& pos, int32_t& outScore)
{
    ASSERT(pos.Whites().pawns == 0 && pos.Whites().bishops == 0 && pos.Whites().knights == 0 && pos.Whites().rooks == 0 && pos.Whites().queens != 0);
    ASSERT(pos.Blacks().pawns == 0 && pos.Blacks().bishops == 0 && pos.Blacks().knights != 0 && pos.Blacks().rooks == 0 && pos.Blacks().queens == 0);

    if (pos.Whites().queens.Count() == 1 && pos.Blacks().knights.Count() == 1)
    {
        const Square strongKing(FirstBitSet(pos.Whites().king));
        const Square weakKing(FirstBitSet(pos.Blacks().king));
        const Square queenSquare(FirstBitSet(pos.Whites().queens));
        const Square knightSquare(FirstBitSet(pos.Blacks().knights));

        if (pos.GetSideToMove() == Color::Black)
        {
            // detect fork
            if (Bitboard::GetKnightAttacks(knightSquare) & Bitboard::GetKnightAttacks(queenSquare) & Bitboard::GetKnightAttacks(strongKing))
            {
                outScore = 0;
                return true;
            }
        }

        outScore = KnownWinValue + (pos.GetSideToMove() == Color::White ? 100 : -100);
        outScore -= 8 * weakKing.EdgeDistance(); // push king to edge
        outScore -= 2 * Square::Distance(weakKing, strongKing); // push kings close
        outScore += Square::Distance(weakKing, knightSquare); // push losing king and knight close
        return true;
    }

    return false;
}

static bool EvaluateEndgame_KRvKR(const Position& pos, int32_t& outScore)
{
    ASSERT(pos.Whites().pawns == 0 && pos.Whites().bishops == 0 && pos.Whites().knights == 0 && pos.Whites().rooks > 0 && pos.Whites().queens == 0);
    ASSERT(pos.Blacks().pawns == 0 && pos.Blacks().bishops == 0 && pos.Blacks().knights == 0 && pos.Blacks().rooks > 0 && pos.Blacks().queens == 0);

    if (pos.Whites().rooks.Count() == 1 && pos.Blacks().rooks.Count() == 1)
    {
        if (pos.IsInCheck(pos.GetSideToMove()))
        {
            return false;
        }

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

static bool EvaluateEndgame_KQvKQ(const Position& pos, int32_t& outScore)
{
    ASSERT(pos.Whites().pawns == 0 && pos.Whites().bishops == 0 && pos.Whites().knights == 0 && pos.Whites().rooks == 0 && pos.Whites().queens > 0);
    ASSERT(pos.Blacks().pawns == 0 && pos.Blacks().bishops == 0 && pos.Blacks().knights == 0 && pos.Blacks().rooks == 0 && pos.Blacks().queens > 0);

    if (pos.Whites().queens.Count() == 1 && pos.Blacks().queens.Count() == 1)
    {
        outScore = 0;
        return true;
    }

    return false;
}

// Rook+Pawn vs. Rook
static bool EvaluateEndgame_KRPvKR(const Position& pos, int32_t& outScore)
{
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

        static const int16_t strongPawnPsqt[] =
        {
              0,      0,      0,      0,      0,      0,      0,      0,
             84,    151,    136,    105,    105,    136,    151,     84,
             91,    157,    143,    115,    115,    143,    157,     91,
            141,    189,    179,    168,    168,    179,    189,    141,
            190,    222,    217,    216,    216,    217,    222,    190,
            290,    304,    303,    310,    310,    303,    304,    290,
            247,    260,    263,    266,    266,    263,    260,    247,
              0,      0,      0,      0,      0,      0,      0,      0,
        };
        static const int16_t strongRookPsqt[] =
        {
            151,    156,    161,    166,    166,    161,    156,    151,
            158,    160,    166,    173,    173,    166,    160,    158,
            162,    165,    170,    177,    177,    170,    165,    162,
            176,    180,    185,    190,    190,    185,    180,    176,
            181,    187,    191,    195,    195,    191,    187,    181,
            178,    184,    188,    191,    191,    188,    184,    178,
            176,    184,    189,    192,    192,    189,    184,    176,
            200,    208,    213,    217,    217,    213,    208,    200,
        };
        static const int16_t strongKingPsqt[] =
        {
             76,     87,    100,    106,    106,    100,     87,     76,
            101,    114,    137,    148,    148,    137,    114,    101,
            131,    156,    187,    200,    200,    187,    156,    131,
            151,    189,    228,    240,    240,    228,    189,    151,
            168,    216,    262,    278,    278,    262,    216,    168,
            177,    234,    280,    302,    302,    280,    234,    177,
            167,    216,    261,    281,    281,    261,    216,    167,
            150,    188,    219,    234,    234,    219,    188,    150,
        };
        static const int16_t weakRookPsqt[] =
        {
            211,    213,    205,    198,    198,    205,    213,    211,
            207,    209,    201,    197,    197,    201,    209,    207,
            197,    198,    189,    184,    184,    189,    198,    197,
            181,    179,    168,    163,    163,    168,    179,    181,
            180,    175,    163,    156,    156,    163,    175,    180,
            182,    176,    163,    154,    154,    163,    176,    182,
            183,    172,    158,    149,    149,    158,    172,    183,
            174,    169,    159,    154,    154,    159,    169,    174,
        };
        static const int16_t weakKingPsqt[] =
        {
            384,    350,    320,    308,    308,    320,    350,    384,
            324,    281,    246,    232,    232,    246,    281,    324,
            272,    232,    196,    178,    178,    196,    232,    272,
            228,    186,    152,    133,    133,    152,    186,    228,
            196,    151,    116,     97,     97,    116,    151,    196,
            176,    128,     92,     71,     71,     92,    128,    176,
            170,    123,     87,     67,     67,     87,    123,    170,
            187,    135,    107,     93,     93,    107,    135,    187,
        };

        outScore = (
            strongPawnPsqt[strongPawn.mIndex] +
            strongRookPsqt[strongRook.mIndex] +
            strongKingPsqt[strongKing.mIndex] +
            weakRookPsqt[weakRook.mIndex] +
            weakKingPsqt[weakKing.mIndex]) / 5;

        outScore += 16 * Square::Distance(weakKing, queeningSquare);
        outScore -= 16 * Square::Distance(strongKing, queeningSquare);

        return true;
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
    RegisterEndgame(MaterialMask_WhiteBishop|MaterialMask_WhitePawn, EvaluateEndgame_KBPvK);
    RegisterEndgame(MaterialMask_WhitePawn, EvaluateEndgame_KPvK);
    RegisterEndgame(MaterialMask_WhitePawn|MaterialMask_BlackPawn, EvaluateEndgame_KPvKP);
    RegisterEndgame(MaterialMask_WhiteQueen|MaterialMask_BlackPawn, EvaluateEndgame_KQvKP);
    RegisterEndgame(MaterialMask_WhiteRook|MaterialMask_BlackPawn, EvaluateEndgame_KRvKP);
    RegisterEndgame(MaterialMask_WhiteRook|MaterialMask_BlackKnight, EvaluateEndgame_KRvKN);
    RegisterEndgame(MaterialMask_WhiteRook|MaterialMask_BlackBishop, EvaluateEndgame_KRvKB);
    RegisterEndgame(MaterialMask_WhiteQueen|MaterialMask_BlackRook, EvaluateEndgame_KQvKR);
    RegisterEndgame(MaterialMask_WhiteQueen|MaterialMask_BlackKnight, EvaluateEndgame_KQvKN);
    RegisterEndgame(MaterialMask_WhiteRook|MaterialMask_BlackRook, EvaluateEndgame_KRvKR);
    RegisterEndgame(MaterialMask_WhiteQueen|MaterialMask_BlackQueen, EvaluateEndgame_KQvKQ);
    RegisterEndgame(MaterialMask_WhiteRook|MaterialMask_WhitePawn|MaterialMask_BlackRook, EvaluateEndgame_KRPvKR);
}

#ifdef COLLECT_ENDGAME_STATISTICS
#include <unordered_map>
static std::mutex s_matKeyOccurencesMutex;
static std::unordered_map<MaterialKey, uint64_t> s_matKeyOccurences;
#endif // COLLECT_ENDGAME_STATISTICS

bool EvaluateEndgame(const Position& pos, int32_t& outScore)
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
        const bool result = func(pos, outScore);
        if (result) { ASSERT(outScore != InvalidValue); }
        return result;
    }

    // find registered function (flipped color)
    evaluationFuncIndex = s_endgameEvaluationMap[FlipColor(materialMask)];
    if (evaluationFuncIndex != UINT8_MAX)
    {
        const EndgameEvaluationFunc& func = s_endgameEvaluationFunctions[evaluationFuncIndex];
        int32_t score = InvalidValue;
        const bool result = func(pos.SwappedColors(), score);
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
