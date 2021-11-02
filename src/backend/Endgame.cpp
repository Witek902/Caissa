#include "Endgame.hpp"
#include "Evaluate.hpp"
#include "Position.hpp"
#include "Move.hpp"
#include "MoveList.hpp"
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

static void RegisterEndgameFunction(MaterialMask materialMask, const EndgameEvaluationFunc& func)
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

static void RegisterEndgameFunction(MaterialMask materialMask, const uint8_t functionIndex)
{
    ASSERT(materialMask < MaterialMask_MAX);

    // function already registered
    ASSERT(s_endgameEvaluationMap[materialMask] == UINT8_MAX);
    ASSERT(s_endgameEvaluationMap[FlipColor(materialMask)] == UINT8_MAX);

    s_endgameEvaluationMap[materialMask] = functionIndex;
}

// Rook(s) and/or Queen(s) vs. lone king
static bool EvaluateEndgame_KXvK(const Position& pos, int32_t& outScore)
{
    ASSERT(pos.Blacks().OccupiedExcludingKing().Count() == 0);

    const Square whiteKing(FirstBitSet(pos.Whites().king));
    const Square blackKing(FirstBitSet(pos.Blacks().king));

    if (pos.GetSideToMove() == Color::Black)
    {
        MoveList moves;
        pos.GenerateKingMoveList(moves);

        // detect stalemate
        if (moves.Size() == 0)
        {
            outScore = 0;
            return true;
        }

        // check if a piece can be captured immediately
        if (pos.Whites().Occupied().Count() == 1)
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
    outScore += 900 * pos.Whites().queens.Count();
    outScore += 500 * pos.Whites().rooks.Count();
    outScore += 300 * (pos.Whites().bishops | pos.Whites().knights).Count();
    outScore += 100 * pos.Whites().pawns.Count();
    outScore += 8 * (3 - blackKing.EdgeDistance()); // push king to edge
    outScore += (7 - Square::Distance(blackKing, whiteKing)); // push kings close
    outScore = std::clamp(outScore, -TablebaseWinValue + 1, TablebaseWinValue - 1);

    // TODO put rook on a rank/file that limits king movement

    return true;
}

// knight(s) vs. lone king
static bool EvaluateEndgame_KNvK(const Position& pos, int32_t& outScore)
{
    ASSERT(pos.Blacks().OccupiedExcludingKing().Count() == 0);

    const Square whiteKing(FirstBitSet(pos.Whites().king));
    const Square blackKing(FirstBitSet(pos.Blacks().king));
    const int32_t numKnights = pos.Whites().knights.Count();

    if (numKnights <= 2)
    {
        // NOTE: there are checkmates with two knights, but they cannot be forced from all positions
        outScore = 0;
    }
    else // whiteKnights >= 3
    {
        outScore = numKnights > 3 ? KnownWinValue : 0;
        outScore += 300 * (numKnights - 3); // prefer keeping the knights
        outScore += 8 * (3 - blackKing.AnyCornerDistance()); // push king to corner
        outScore += (7 - Square::Distance(blackKing, whiteKing)); // push kings close
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

    const Square whiteKing(FirstBitSet(pos.Whites().king));
    const Square blackKing(FirstBitSet(pos.Blacks().king));
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
        outScore += 8 * (3 - blackKing.AnyCornerDistance()); // push king to corner
        outScore += (7 - Square::Distance(blackKing, whiteKing)); // push kings close
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

    const Square whiteKing(FirstBitSet(pos.Whites().king));
    const Square blackKing(FirstBitSet(pos.Blacks().king));
    const int32_t whiteBishops = pos.Whites().bishops.Count();
    const int32_t whiteKnights = pos.Whites().knights.Count();

    // push king to 'right' board corner
    const Square kingSquare = (pos.Whites().bishops & Bitboard::DarkSquares()) ? blackKing : blackKing.FlippedFile();

    outScore = KnownWinValue;
    outScore += 300 * (whiteBishops - 1); // prefer keeping the knights
    outScore += 300 * (whiteKnights - 1); // prefer keeping the knights
    outScore += 8 * (7 - kingSquare.DarkCornerDistance()); // push king to corner
    outScore += (7 - Square::Distance(blackKing, whiteKing)); // push kings close

    return true;
}

// pawn(s) vs. lone king
static bool EvaluateEndgame_KPvK(const Position& pos, int32_t& outScore)
{
    ASSERT(pos.Whites().pawns.Count() > 0);
    ASSERT(pos.Blacks().OccupiedExcludingKing().Count() == 0);

    const Square whiteKing(FirstBitSet(pos.Whites().king));
    const Square blackKing(FirstBitSet(pos.Blacks().king));
    const int32_t numPawns = pos.Whites().pawns.Count();

    Square strongKingSq = whiteKing;
    Square weakKingSq = blackKing;
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

        outScore = KnownWinValue;
        outScore += 8 * pawnSquare.Rank();
        outScore += 7 - std::max(0, (int32_t)Square::Distance(pawnSquare, strongKingSq) - 1); // push kings close to pawn
        outScore += std::max(0, (int32_t)Square::Distance(pawnSquare, weakKingSq) - 1); // push kings close to pawn
        return true;
    }
    else if (numPawns == 2)
    {
        // connected passed pawns
        if (Bitboard::GetKingAttacks(pawnSquare) & pos.Whites().pawns)
        {
            outScore = KnownWinValue;
            outScore += 8 * pawnSquare.Rank();
            outScore += 7 - std::max(0, (int32_t)Square::Distance(pawnSquare, strongKingSq) - 1); // push kings close to pawn
            outScore += std::max(0, (int32_t)Square::Distance(pawnSquare, weakKingSq) - 1); // push kings close to pawn
            return true;
        }
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

    const Square blackKing(FirstBitSet(pos.Blacks().king));

    // if all pawns are on A/H file and we have a wrong bishop, then it's a draw
    {
        if (((pos.Whites().pawns & ~Bitboard::FileBitboard<0>()) == 0) &&
            ((pos.Whites().bishops & Bitboard::LightSquares()) == 0) &&
            (Square::Distance(blackKing, Square_a8) <= 1))
        {
            outScore = 0;
            return true;
        }
        if (((pos.Whites().pawns & ~Bitboard::FileBitboard<7>()) == 0) &&
            ((pos.Whites().bishops & Bitboard::DarkSquares()) == 0) &&
            (Square::Distance(blackKing, Square_h8) <= 1))
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
                RegisterEndgameFunction((MaterialMask)mask, functionIndex);
            }
        }
    }

    RegisterEndgameFunction(MaterialMask_WhiteKnight, EvaluateEndgame_KNvK);

    RegisterEndgameFunction(MaterialMask_WhiteBishop, EvaluateEndgame_KBvK);
    RegisterEndgameFunction(MaterialMask_WhiteBishop|MaterialMask_BlackKnight, EvaluateEndgame_KBvK);

    RegisterEndgameFunction(MaterialMask_WhiteBishop|MaterialMask_WhiteKnight, EvaluateEndgame_KNBvK);

    RegisterEndgameFunction(MaterialMask_WhiteBishop|MaterialMask_WhitePawn, EvaluateEndgame_KBPvK);

    RegisterEndgameFunction(MaterialMask_WhitePawn, EvaluateEndgame_KPvK);
}

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

/*
    // Pawns vs Pawns
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
*/