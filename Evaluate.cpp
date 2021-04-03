#include "Evaluate.hpp"
#include "Move.hpp"

static constexpr int32_t c_kingValue            = 1000;
static constexpr int32_t c_queenValue           = 900;
static constexpr int32_t c_rookValue            = 500;
static constexpr int32_t c_bishopValue          = 330;
static constexpr int32_t c_knightValue          = 320;
static constexpr int32_t c_pawnValue            = 100;

static constexpr int32_t c_castlingRightsBonus  = 5;
static constexpr int32_t c_mobilityBonus        = 20;
static constexpr int32_t c_guardBonus           = 10;

static const int8_t c_PawnTable[] =
{
    55, 60, 65, 70, 70, 65, 60, 55,
    50, 50, 50, 50, 50, 50, 50, 50,
    20, 20, 30, 40, 40, 30, 20, 20,
    5,  5, 10, 30, 30, 10,  5,  5,
    0,  0,  0, 25, 25,  0,  0,  0,
    5, -5,-10,  0,  0,-10, -5,  5,
    5, 10, 10,-30,-30, 10, 10,  5,
    0,  0,  0,  0,  0,  0,  0,  0,
};

static const int8_t c_KnightTable[] =
{
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-30,-20,-30,-30,-20,-30,-50,
};

static const int8_t c_BishopTable[] =
{
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -20,-10,-40,-10,-10,-40,-10,-20,
};

static const int8_t c_KingTable_MiddleGame[] =
{
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -10, -20, -20, -20, -20, -20, -20, -10, 
     20,  20,   0,   0,   0,   0,  20,  20,
     20,  30,  10,   0,   0,  10,  30,  20,
};

static const int8_t c_KingTable_EndGame[] =
{
    -50,-40,-30,-20,-20,-30,-40,-50,
    -30,-20,-10,  0,  0,-10,-20,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-30,  0,  0,  0,  0,-30,-30,
    -50,-30,-30,-30,-30,-30,-30,-50,
};

static uint32_t FlipRank(uint32_t square)
{
    uint32_t rank = square / 8;
    uint32_t file = square % 8;
    square = 8u * (7u - rank) + file;
    return square;
}

int32_t ScoreQuietMove(const Move& move, const Color color)
{
    ASSERT(move.IsValid());
    ASSERT(!move.isCapture);
    ASSERT(!move.isEnPassant);

    uint32_t fromSquare = move.fromSquare.Index();
    uint32_t toSquare = move.toSquare.Index();

    if (color == Color::White)
    {
        fromSquare = FlipRank(fromSquare);
        toSquare = FlipRank(toSquare);
    }

    int32_t score = 0;

    switch (move.piece)
    {
    case Piece::Pawn:   score = c_PawnTable[toSquare] - c_PawnTable[fromSquare]; break;
    case Piece::Knight: score = c_KnightTable[toSquare] - c_KnightTable[fromSquare]; break;
    case Piece::Bishop: score = c_BishopTable[toSquare] - c_BishopTable[fromSquare]; break;
    }

    return std::max(0, score);
}

int32_t Evaluate(const Position& position)
{
    int32_t value = 0;

    value += c_queenValue * ((int32_t)position.Whites().queens.Count() - (int32_t)position.Blacks().queens.Count());
    value += c_rookValue * ((int32_t)position.Whites().rooks.Count() - (int32_t)position.Blacks().rooks.Count());
    value += c_bishopValue * ((int32_t)position.Whites().bishops.Count() - (int32_t)position.Blacks().bishops.Count());
    value += c_knightValue * ((int32_t)position.Whites().knights.Count() - (int32_t)position.Blacks().knights.Count());
    value += c_pawnValue * ((int32_t)position.Whites().pawns.Count() - (int32_t)position.Blacks().pawns.Count());

    const Bitboard whiteAttackedSquares = position.GetAttackedSquares(Color::White);
    const Bitboard blackAttackedSquares = position.GetAttackedSquares(Color::Black);
    const Bitboard whiteOccupiedSquares = position.Whites().Occupied();
    const Bitboard blackOccupiedSquares = position.Blacks().Occupied();

    const Bitboard whitesMobility = whiteAttackedSquares & ~whiteOccupiedSquares;
    const Bitboard blacksMobility = blackAttackedSquares & ~blackOccupiedSquares;
    value += c_mobilityBonus * ((int32_t)whitesMobility.Count() - (int32_t)blacksMobility.Count());

    const Bitboard whitesGuardedPieces = whiteAttackedSquares & whiteOccupiedSquares;
    const Bitboard blacksGuardedPieces = blackAttackedSquares & blackOccupiedSquares;
    value += c_guardBonus * ((int32_t)whitesGuardedPieces.Count() - (int32_t)blacksGuardedPieces.Count());

    value += c_castlingRightsBonus * ((int32_t)__popcnt16(position.GetWhitesCastlingRights()) - (int32_t)__popcnt16(position.GetBlacksCastlingRights()));

    if (whiteAttackedSquares & position.Blacks().king)
    {
        value += 80;
    }
    if (blackAttackedSquares & position.Whites().king)
    {
        value -= 80;
    }

    // piece square tables
    {
        int32_t pieceSquareValue = 0;

        position.Whites().pawns.Iterate([&](uint32_t square)
        {
            square = FlipRank(square);
            ASSERT(square < 64);
            pieceSquareValue += c_PawnTable[square];
        });
        position.Blacks().pawns.Iterate([&](uint32_t square)
        {
            ASSERT(square < 64);
            pieceSquareValue -= c_PawnTable[square];
        });

        position.Whites().knights.Iterate([&](uint32_t square)
        {
            square = FlipRank(square);
            ASSERT(square < 64);
            pieceSquareValue += c_KnightTable[square];
        });
        position.Blacks().knights.Iterate([&](uint32_t square)
        {
            ASSERT(square < 64);
            pieceSquareValue -= c_KnightTable[square];
        });

        position.Whites().bishops.Iterate([&](uint32_t square)
        {
            square = FlipRank(square);
            ASSERT(square < 64);
            pieceSquareValue += c_BishopTable[square];
        });
        position.Blacks().bishops.Iterate([&](uint32_t square)
        {
            ASSERT(square < 64);
            pieceSquareValue -= c_BishopTable[square];
        });

        value += pieceSquareValue / 5;
    }

    return value;
}
