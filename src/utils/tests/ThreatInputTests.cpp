#include "TestCommon.hpp"
#include "../TrainerCommon.hpp"
#include "../../backend/MoveList.hpp"
#include "../../backend/NeuralNetworkEvaluator.hpp"
#include "../../backend/Position.hpp"
#include "../../backend/ThreatInputs.hpp"

#include <algorithm>
#include <random>
#include <vector>

namespace {

std::vector<uint16_t> GetSortedThreatFeatures(const Position& pos, const Color perspective)
{
    uint16_t features[nn::MaxThreatRecords];
    const uint32_t numFeatures = PositionToThreatFeatures(pos, perspective, features);
    std::vector<uint16_t> result(features, features + numFeatures);
    std::sort(result.begin(), result.end());
    return result;
}

bool IsMutualPair(const Piece attacker, const Color attackerColor, const Piece victim, const Color victimColor)
{
    return attacker == victim && (attackerColor != victimColor || attacker != Piece::Pawn);
}

bool IsFeaturePair(const Piece attacker, const Piece victim)
{
    if (victim == Piece::King) return false;
    if (attacker == Piece::Pawn) return victim != Piece::Bishop && victim != Piece::Queen;
    if (attacker == Piece::Bishop || attacker == Piece::Rook) return victim != Piece::Queen;
    return true;
}

// every threat that fits on an empty board maps to its own feature, or to none if it is excluded
void TestFeatureLayout()
{
    const ThreatOrientation identity = { 0, 0 };

    std::vector<bool> used(nn::NumThreatInputs, false);
    uint32_t numUsed = 0;

    for (Color attackerColor = 0; attackerColor < 2; ++attackerColor)
    {
        for (uint32_t a = (uint32_t)Piece::Pawn; a <= (uint32_t)Piece::Queen; ++a)
        {
            const Piece attacker = (Piece)a;
            for (uint32_t from = 0; from < 64; ++from)
            {
                Bitboard attacks = 0;
                switch (attacker)
                {
                case Piece::Pawn:
                    if (Square(from).Rank() != 0 && Square(from).Rank() != 7)
                        attacks = Bitboard::GetPawnAttacks(Square(from), attackerColor);
                    break;
                case Piece::Knight: attacks = Bitboard::GetKnightAttacks(Square(from)); break;
                case Piece::Bishop: attacks = Bitboard::GenerateBishopAttacks(Square(from), 0); break;
                case Piece::Rook:   attacks = Bitboard::GenerateRookAttacks(Square(from), 0); break;
                default:            attacks = Bitboard::GenerateQueenAttacks(Square(from), 0); break;
                }

                attacks.Iterate([&](const uint32_t to)
                {
                    for (Color victimColor = 0; victimColor < 2; ++victimColor)
                    {
                        for (uint32_t v = (uint32_t)Piece::Pawn; v <= (uint32_t)Piece::King; ++v)
                        {
                            const Piece victim = (Piece)v;
                            const ThreatRecord record = { ThreatPieceCode(attacker, attackerColor), (uint8_t)from, ThreatPieceCode(victim, victimColor), (uint8_t)to };
                            const uint32_t index = GetThreatFeatureIndex(record, identity);

                            bool expectFeature = IsFeaturePair(attacker, victim);
                            if (IsMutualPair(attacker, attackerColor, victim, victimColor))
                                expectFeature &= from > to;

                            TEST_EXPECT_EQ(index < nn::NumThreatInputs, expectFeature);
                            if (index < nn::NumThreatInputs)
                            {
                                TEST_EXPECT(!used[index]);
                                used[index] = true;
                                numUsed++;
                            }
                        }
                    }
                });
            }
        }
    }

    // the rest are the unused halves of the same-type blocks
    TEST_EXPECT_EQ(numUsed, 53564u);
}

// index lists from another engines that uses the same threats layout
void TestGoldenFeatures()
{
    {
        const std::vector<uint16_t> expected =
        {
            506,   525,   3878,  3879,  3899,  3900,  8351,  8449,
            9240,  9344,  15603, 15604, 15605, 18512, 32570, 32589,
            36699, 36700, 36720, 36721, 42790, 42888, 43687, 43791,
            54247, 54248, 54249, 57166,
        };

        const Position pos(Position::InitPositionFEN);
        TEST_EXPECT(GetSortedThreatFeatures(pos, White) == expected);
        TEST_EXPECT(GetSortedThreatFeatures(pos, Black) == expected);
    }

    {
        const std::vector<uint16_t> whiteExpected =
        {
            34,    95,    605,   606,   608,   1276,  2034,  2374,
            2376,  2377,  4517,  8351,  8449,  15907, 15908, 15919,
            17370, 18821, 23190, 24659, 30086, 30134, 30195, 30397,
            30398, 30401, 30488, 30491, 30807, 30809, 30840, 32491,
            32521, 33531, 35486, 37185, 38306, 42786, 42888, 54045,
            54050, 54054, 54055, 55505,
        };
        const std::vector<uint16_t> blackExpected =
        {
            3,     4,     7,     94,    97,    389,   581,   612,
            1619,  2263,  2265,  2293,  4490,  5607,  8355,  8449,
            15752, 15753, 15758, 15765, 17213, 30107, 30143, 30372,
            30489, 30710, 30711, 30712, 32510, 32512, 32515, 33186,
            33740, 35517, 37212, 42790, 42888, 46560, 48008, 53839,
            53847, 53848, 55300, 56761,
        };

        const Position pos("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1");
        TEST_EXPECT(GetSortedThreatFeatures(pos, White) == whiteExpected);
        TEST_EXPECT(GetSortedThreatFeatures(pos, Black) == blackExpected);
    }
}

static const char* c_testPositions[] =
{
    Position::InitPositionFEN,
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
    "rnbqkb1r/pp1p1ppp/5n2/2pPp3/8/8/PPP1PPPP/RNBQKBNR w KQkq c6 0 4",
    "k7/2n1n3/1nbNbn2/2NbRBn1/1nbRQR2/2NBRBN1/3N1N2/7K w - - 0 1",
    "K7/8/8/BNQNQNB1/N5N1/R1Q1q2r/n5n1/bnqnqnbk w - - 0 1",
    "1r6/1P4bk/3qr1p1/N6p/3pp2P/6R1/3Q1PP1/1R4K1 w - - 1 42",
    "3NkN2/BqnQQR2/r1qnB1Qn/1Pnqnp2/b1qNpQP1/1RNr1b2/3N4/4K3 b - - 0 1",
    "1k1bnrnb/3n3n/3r1q1r/BNRN3n/N3nrnb/R1Q1R3/N3N3/BNRNB1K1 w - - 0 1",
    "4k3/1P6/8/8/8/8/6p1/4K3 w - - 0 1",
    "bqnb1rkr/pp3ppp/3ppn2/2p5/5P2/P2P4/NPP1P1PP/BQ1BNRKR w HFhf - 2 9",
    "nrkb1qbr/pp1pppp1/5n2/7p/2p5/1N1NPP2/PPPP2PP/1RKB1QBR w HBhb - 0 9",
    "rnkrbbq1/pppppnp1/7p/8/1B1Q1p2/3P1P2/PPP1P1PP/RNKR1B1N w DAda - 2 9",
    "1r2k1r1/pppppppp/8/8/8/8/PPPPPPPP/1R2K1R1 w GBgb - 0 1",
    "qQqQqQqk/QqQqQqbq/qQqQqQqQ/QqQqQqQq/qQqQqQqQ/QqQqQqQq/qQqQqQBQ/QqQqQqQK w - - 0 1",
};

// the features of a position and of its color-flipped twin are the same
void TestSymmetry()
{
    for (const char* fen : c_testPositions)
    {
        Position pos;
        TEST_EXPECT(pos.FromFEN(fen));
        const Position swapped = pos.SwappedColors();
        TEST_EXPECT(GetSortedThreatFeatures(pos, White) == GetSortedThreatFeatures(swapped, Black));
        TEST_EXPECT(GetSortedThreatFeatures(pos, Black) == GetSortedThreatFeatures(swapped, White));
        TEST_EXPECT(GetSortedThreatFeatures(pos, White).size() == GetSortedThreatFeatures(pos, Black).size());

        if (pos.GetWhitesCastlingRights() == 0 && pos.GetBlacksCastlingRights() == 0)
        {
            const Position mirrored = pos.MirroredHorizontally();
            TEST_EXPECT(GetSortedThreatFeatures(pos, White) == GetSortedThreatFeatures(mirrored, White));
            TEST_EXPECT(GetSortedThreatFeatures(pos, Black) == GetSortedThreatFeatures(mirrored, Black));
        }
    }
}

void ValidateMove(const Position& pos, const Move move)
{
    Position child = pos;
    if (!child.DoMove(move))
        return;

    TEST_EXPECT(ValidateThreatDeltas(pos, child, White));
    TEST_EXPECT(ValidateThreatDeltas(pos, child, Black));
}

// the threat deltas of every legal move match the difference of the full feature sets
void TestDeltasOfAllMoves()
{
    for (const char* fen : c_testPositions)
    {
        Position pos;
        TEST_EXPECT(pos.FromFEN(fen));

        MoveList moves;
        pos.GetNumLegalMoves(&moves);
        for (uint32_t i = 0; i < moves.Size(); ++i)
            ValidateMove(pos, moves.GetMove(i));

        // a null move changes no threats
        Position nullMovePos = pos;
        if (!pos.IsInCheck() && nullMovePos.DoNullMove())
        {
            ThreatDeltas deltas;
            GetThreatDeltas(pos, nullMovePos, deltas);
            TEST_EXPECT_EQ(deltas.numAdded, 0u);
            TEST_EXPECT_EQ(deltas.numRemoved, 0u);
        }
    }
}

void TestDeltasInRandomGames()
{
    std::mt19937 gen(12345);

    for (const char* fen : c_testPositions)
    {
        for (uint32_t game = 0; game < 20; ++game)
        {
            Position pos;
            TEST_EXPECT(pos.FromFEN(fen));
            for (uint32_t ply = 0; ply < 200; ++ply)
            {
                MoveList moves;
                if (pos.GetNumLegalMoves(&moves) == 0)
                    break;

                const Move move = moves.GetMove(std::uniform_int_distribution<uint32_t>(0, moves.Size() - 1)(gen));
                ValidateMove(pos, move);

                TEST_EXPECT_EQ(GetSortedThreatFeatures(pos, White).size(), GetSortedThreatFeatures(pos, Black).size());

                VERIFY(pos.DoMove(move));
            }
        }
    }
}

// random position the search could evaluate: no pawns on the first or last rank, neither side in check
bool GenerateEvaluablePosition(std::mt19937& gen, Position& outPos)
{
    static const Piece pieces[] = { Piece::Pawn, Piece::Knight, Piece::Bishop, Piece::Rook, Piece::Queen };
    const Bitboard backRanks = Bitboard::RankBitboard<0>() | Bitboard::RankBitboard<7>();

    Position pos;
    Bitboard occupied = 0;
    const auto randomFreeSquare = [&](const Bitboard excluded)
    {
        for (;;)
        {
            const Square square(static_cast<uint32_t>(gen() % 64));
            if (!((occupied | excluded) & square))
                return square;
        }
    };

    const Square whiteKing = randomFreeSquare(0);
    occupied |= whiteKing.GetBitboard();
    const Square blackKing = randomFreeSquare(Bitboard::GetKingAttacks(whiteKing));
    occupied |= blackKing.GetBitboard();
    pos.SetPiece(whiteKing, Piece::King, White);
    pos.SetPiece(blackKing, Piece::King, Black);

    const uint32_t numPieces = static_cast<uint32_t>(gen() % 31);
    for (uint32_t i = 0; i < numPieces; ++i)
    {
        const Piece piece = pieces[gen() % 5];
        const Square square = randomFreeSquare(piece == Piece::Pawn ? backRanks : Bitboard(0));
        occupied |= square.GetBitboard();
        pos.SetPiece(square, piece, static_cast<Color>(gen() % 2));
    }

    pos.SetSideToMove(static_cast<Color>(gen() % 2));

    if (!pos.IsValid() || pos.IsInCheck(White) || pos.IsInCheck(Black))
        return false;

    outPos = pos;
    return true;
}

// the inputs zeroed in saved nets are never active in a position the search evaluates
void TestUnreachableInputs()
{
    const std::vector<bool> unreachablePieceInputs = GetUnreachablePieceInputs();
    const std::vector<bool> unreachableThreatInputs = GetUnreachableThreatInputs();

    std::mt19937 gen(54321);
    for (uint32_t numPositions = 0; numPositions < 100000; )
    {
        Position pos;
        if (!GenerateEvaluablePosition(gen, pos))
            continue;

        for (Color perspective = 0; perspective < 2; ++perspective)
        {
            uint16_t features[64];
            const uint32_t numFeatures = PositionToFeaturesVector(pos, features, perspective);
            for (uint32_t i = 0; i < numFeatures; ++i)
                TEST_EXPECT(!unreachablePieceInputs[features[i]]);

            uint16_t threatFeatures[nn::MaxThreatRecords];
            const uint32_t numThreatFeatures = PositionToThreatFeatures(pos, perspective, threatFeatures);
            for (uint32_t i = 0; i < numThreatFeatures; ++i)
                TEST_EXPECT(!unreachableThreatInputs[threatFeatures[i]]);
        }

        numPositions++;
    }
}

// Threats found by stepping each ray one square at a time, independent of the bitboard attack code.
// Records packed as (attacker, from, victim, to), sorted.
std::vector<uint32_t> GetNaiveThreatRecords(const Position& pos)
{
    struct Step { int32_t file, rank; };
    static const Step knightSteps[] = { { 1, 2 }, { 2, 1 }, { 2, -1 }, { 1, -2 }, { -1, -2 }, { -2, -1 }, { -2, 1 }, { -1, 2 } };
    static const Step bishopSteps[] = { { 1, 1 }, { 1, -1 }, { -1, 1 }, { -1, -1 } };
    static const Step rookSteps[] = { { 1, 0 }, { -1, 0 }, { 0, 1 }, { 0, -1 } };

    const auto pieceAt = [&pos](const uint32_t square, Piece& outPiece, Color& outColor)
    {
        for (Color color = 0; color < 2; ++color)
        {
            outPiece = pos.GetSide(color).GetPieceAtSquare(Square(square));
            outColor = color;
            if (outPiece != Piece::None)
                return true;
        }
        return false;
    };

    std::vector<uint32_t> records;
    for (uint32_t from = 0; from < 64; ++from)
    {
        Piece piece;
        Color color;
        if (!pieceAt(from, piece, color) || piece == Piece::King)
            continue;

        const int32_t file = from % 8;
        const int32_t rank = from / 8;

        // false when the ray stops: off the board or at any piece, kings included
        const auto visit = [&](const int32_t targetFile, const int32_t targetRank)
        {
            if (targetFile < 0 || targetFile > 7 || targetRank < 0 || targetRank > 7)
                return false;
            const uint32_t to = targetRank * 8 + targetFile;
            Piece victim;
            Color victimColor;
            if (!pieceAt(to, victim, victimColor))
                return true;
            if (IsFeaturePair(piece, victim))
                records.push_back((uint32_t)ThreatPieceCode(piece, color) << 24 | from << 16 | (uint32_t)ThreatPieceCode(victim, victimColor) << 8 | to);
            return false;
        };

        const auto slide = [&](const Step* steps, const uint32_t numSteps)
        {
            for (uint32_t i = 0; i < numSteps; ++i)
                for (int32_t k = 1; visit(file + k * steps[i].file, rank + k * steps[i].rank); ++k) {}
        };

        switch (piece)
        {
        case Piece::Pawn:
            visit(file - 1, rank + (color == White ? 1 : -1));
            visit(file + 1, rank + (color == White ? 1 : -1));
            break;
        case Piece::Knight:
            for (const Step& step : knightSteps)
                visit(file + step.file, rank + step.rank);
            break;
        case Piece::Bishop: slide(bishopSteps, 4); break;
        case Piece::Rook: slide(rookSteps, 4); break;
        case Piece::Queen: slide(bishopSteps, 4); slide(rookSteps, 4); break;
        default: break;
        }
    }

    std::sort(records.begin(), records.end());
    return records;
}

void CheckAgainstNaiveReference(const Position& pos)
{
    const std::vector<uint32_t> expected = GetNaiveThreatRecords(pos);

    ThreatRecord records[nn::MaxThreatRecords];
    const uint32_t numRecords = GetThreatRecords(pos, records);
    std::vector<uint32_t> actual;
    for (uint32_t i = 0; i < numRecords; ++i)
        actual.push_back((uint32_t)records[i].attacker << 24 | (uint32_t)records[i].from << 16 | (uint32_t)records[i].victim << 8 | records[i].to);
    std::sort(actual.begin(), actual.end());
    TEST_EXPECT(actual == expected);

    // of two same-type pieces threatening each other only the threat from the higher oriented square is a feature
    for (Color perspective = 0; perspective < 2; ++perspective)
    {
        const uint32_t squareFlip = (perspective == Black ? 56u : 0u) | (pos.GetSide(perspective).GetKingSquare().File() >= 4 ? 7u : 0u);
        uint32_t expectedFeatures = 0;
        for (const uint32_t record : expected)
        {
            const uint8_t attacker = (uint8_t)(record >> 24);
            const uint8_t victim = (uint8_t)(record >> 8);
            const uint32_t from = ((record >> 16) & 0xFF) ^ squareFlip;
            const uint32_t to = (record & 0xFF) ^ squareFlip;
            const bool mutual = IsMutualPair((Piece)(attacker & 7), (Color)(attacker >> 3), (Piece)(victim & 7), (Color)(victim >> 3));
            expectedFeatures += !(mutual && from < to);
        }
        TEST_EXPECT_EQ(GetSortedThreatFeatures(pos, perspective).size(), (size_t)expectedFeatures);
    }
}

void TestNaiveReference()
{
    for (const char* fen : c_testPositions)
    {
        Position pos;
        TEST_EXPECT(pos.FromFEN(fen));
        CheckAgainstNaiveReference(pos);
    }

    std::mt19937 gen(777);
    for (const char* fen : c_testPositions)
    {
        for (uint32_t game = 0; game < 5; ++game)
        {
            Position pos;
            TEST_EXPECT(pos.FromFEN(fen));
            for (uint32_t ply = 0; ply < 200; ++ply)
            {
                MoveList moves;
                if (pos.GetNumLegalMoves(&moves) == 0)
                    break;
                VERIFY(pos.DoMove(moves.GetMove(std::uniform_int_distribution<uint32_t>(0, moves.Size() - 1)(gen))));
                CheckAgainstNaiveReference(pos);
            }
        }
    }

    for (uint32_t numPositions = 0; numPositions < 100000; )
    {
        Position pos;
        if (!GenerateEvaluablePosition(gen, pos))
            continue;
        CheckAgainstNaiveReference(pos);
        numPositions++;
    }
}

} // namespace

void RunThreatInputTests()
{
    TestFeatureLayout();
    TestGoldenFeatures();
    TestSymmetry();
    TestNaiveReference();
    TestDeltasOfAllMoves();
    TestDeltasInRandomGames();
    TestUnreachableInputs();
}
