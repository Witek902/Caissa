#include "PgnParser.hpp"

#include "../backend/Position.hpp"
#include "../backend/Move.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

#define TEST_EXPECT(x) \
    if (!(x)) { std::cout << "Test failed: " << #x << " at " << __FILE__ << ":" << __LINE__ << std::endl; DEBUG_BREAK(); }

#define TEST_EXPECT_EQ(a, b) \
    if ((a) != (b)) { std::cout << "Test failed: " << #a << " == " << #b << " (got " << (a) << " vs " << (b) << ") at " << __FILE__ << ":" << __LINE__ << std::endl; DEBUG_BREAK(); }

static std::vector<Game> ParseAll(const char* pgn)
{
    std::istringstream ss(pgn);
    std::vector<Game> games;
    ParsePgn(ss, [&](Game& g) { games.push_back(g); return true; });
    return games;
}

static void TestBasicGame()
{
    // Fool's mate: 1. f3 e5 2. g4 Qh4#
    const char* pgn =
        "[Event \"?\"]\n"
        "[Result \"0-1\"]\n"
        "\n"
        "1. f3 e5 2. g4 Qh4# 0-1\n";

    auto games = ParseAll(pgn);
    TEST_EXPECT(games.size() == 1);
    TEST_EXPECT(games[0].GetMoves().size() == 4);
    TEST_EXPECT(games[0].GetScore() == Game::Score::BlackWins);
}

static void TestNonStandardStart()
{
    // Position after 1.e4: it's Black to move
    const char* pgn =
        "[Event \"?\"]\n"
        "[FEN \"rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1\"]\n"
        "[SetUp \"1\"]\n"
        "[Result \"*\"]\n"
        "\n"
        "1... e5 2. Nf3 Nc6 *\n";

    auto games = ParseAll(pgn);
    TEST_EXPECT(games.size() == 1);
    TEST_EXPECT(games[0].GetMoves().size() == 3);
    TEST_EXPECT(games[0].GetInitialPosition().GetSideToMove() == Black);
}

static void TestMoveScores()
{
    // Caissa self-play format: {score/0}
    const char* pgn =
        "[Event \"?\"]\n"
        "[Result \"*\"]\n"
        "\n"
        "1. e4 {+0.25/0} e5 {-0.30/0} 2. Nf3 {+0.20/0} *\n";

    auto games = ParseAll(pgn);
    TEST_EXPECT(games.size() == 1);

    const auto& scores = games[0].GetMoveScores();
    TEST_EXPECT(scores.size() == 3);
    // White's move e4: +0.25 from White's POV → stored as +25
    TEST_EXPECT_EQ(scores[0], 25);
    // Black's move e5: -0.30 from Black's POV → stored as +30 (White's POV = negated)
    TEST_EXPECT_EQ(scores[1], 30);
    // White's move Nf3: +0.20 from White's POV → stored as +20
    TEST_EXPECT_EQ(scores[2], 20);
}

static void TestOpenBenchFormat()
{
    // OpenBench comment format: {score depth/seldepth time nodes}
    // For this we need a valid starting FEN where Nge2 and Nxe4 are legal.
    // Use a custom FEN from the sample file:
    const char* pgnWithFen =
        "[Event \"?\"]\n"
        "[FEN \"r1b1k2r/ppp1qppp/2np1n2/8/3PP3/2N4P/PPP3P1/R2QKBNR w KQkq - 0 1\"]\n"
        "[SetUp \"1\"]\n"
        "[Result \"1/2-1/2\"]\n"
        "\n"
        "1. Nge2 {-1.08 18/37 477 637018} Nxe4 {+1.24 19/33 228 294164} 1/2-1/2\n";

    auto games = ParseAll(pgnWithFen);
    TEST_EXPECT(games.size() == 1);
    TEST_EXPECT(games[0].GetMoves().size() == 2);

    const auto& scores = games[0].GetMoveScores();
    TEST_EXPECT(scores.size() == 2);
    // Nge2 is White's move: -1.08 from White's POV → -108
    TEST_EXPECT_EQ(scores[0], -108);
    // Nxe4 is Black's move: +1.24 from Black's POV → negated to -124 (White's POV)
    TEST_EXPECT_EQ(scores[1], -124);
}

static void TestMateScore()
{
    const char* pgn =
        "[Event \"?\"]\n"
        "[Result \"1-0\"]\n"
        "\n"
        "1. f3 {+0.00/0} e5 {-0.00/0} 2. g4 {+M2/0} Qh4# {-M1/0} 1-0\n";

    auto games = ParseAll(pgn);
    TEST_EXPECT(games.size() == 1);

    const auto& scores = games[0].GetMoveScores();
    TEST_EXPECT(scores.size() == 4);
    // g4 is White's move: +M2 from White's POV
    TEST_EXPECT_EQ(scores[2], CheckmateValue - 2 * 2 + 1);
    // Qh4# is Black's move: -M1 from Black's POV → negated to +M1 in White's POV
    // -M1 parsed: score = -CheckmateValue + 2*1 - 1 = -31999
    // stored (Black's move): negated → +31999 = CheckmateValue - 2*1 + 1
    TEST_EXPECT_EQ(scores[3], CheckmateValue - 2 * 1 + 1);
}

static void TestMultipleGames()
{
    const char* pgn =
        "[Event \"?\"]\n"
        "[Result \"0-1\"]\n"
        "\n"
        "1. f3 e5 2. g4 Qh4# 0-1\n"
        "\n"
        "[Event \"?\"]\n"
        "[Result \"1-0\"]\n"
        "\n"
        "1. e4 e5 2. Qh5 Nc6 3. Bc4 Nf6 4. Qxf7# 1-0\n";

    auto games = ParseAll(pgn);
    TEST_EXPECT_EQ(games.size(), 2u);
    TEST_EXPECT(games[0].GetMoves().size() == 4);
    TEST_EXPECT(games[0].GetScore() == Game::Score::BlackWins);
    TEST_EXPECT(games[1].GetMoves().size() == 7);
    TEST_EXPECT(games[1].GetScore() == Game::Score::WhiteWins);
}

static void TestSkipBadGame()
{
    // Second game has an illegal/invalid move — should be skipped
    const char* pgn =
        "[Event \"?\"]\n"
        "[Result \"0-1\"]\n"
        "\n"
        "1. f3 e5 2. g4 Qh4# 0-1\n"
        "\n"
        "[Event \"?\"]\n"
        "[Result \"*\"]\n"
        "\n"
        "1. e4 INVALID *\n"
        "\n"
        "[Event \"?\"]\n"
        "[Result \"0-1\"]\n"
        "\n"
        "1. f3 e5 2. g4 Qh4# 0-1\n";

    auto games = ParseAll(pgn);
    TEST_EXPECT_EQ(games.size(), 2u);
    TEST_EXPECT(games[0].GetScore() == Game::Score::BlackWins);
    TEST_EXPECT(games[1].GetScore() == Game::Score::BlackWins);
}

static void TestAdjudicatedResult()
{
    // Game with no natural result — Result header provides forced score
    const char* pgn =
        "[Event \"?\"]\n"
        "[Result \"1-0\"]\n"
        "\n"
        "1. e4 e5 2. Nf3 Nc6 1-0\n";

    auto games = ParseAll(pgn);
    TEST_EXPECT(games.size() == 1);
    TEST_EXPECT(games[0].GetScore() == Game::Score::WhiteWins);
}

static void TestCastling()
{
    // Both sides castle; O-O must not be confused with a termination token
    const char* pgn =
        "[Event \"?\"]\n"
        "[Result \"*\"]\n"
        "\n"
        "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 O-O *\n";

    auto games = ParseAll(pgn);
    TEST_EXPECT(games.size() == 1);
    TEST_EXPECT_EQ(games[0].GetMoves().size(), 14u);
}

static void TestEarlyStop()
{
    // Callback returns false after first game → only first game counted
    const char* pgn =
        "[Result \"0-1\"]\n\n1. f3 e5 2. g4 Qh4# 0-1\n"
        "[Result \"1-0\"]\n\n1. e4 e5 2. Qh5 Nc6 3. Bc4 Nf6 4. Qxf7# 1-0\n";

    std::istringstream ss(pgn);
    uint64_t n = ParsePgn(ss, [](Game&) { return false; });
    // Callback returned false on first game, so count is 1 (it was delivered before stop)
    TEST_EXPECT_EQ(n, 1u);
}

static void TestRoundTrip()
{
    // Generate a PGN from ToPGN(), re-parse it, verify game is identical
    Game original;
    original.Reset(Position(Position::InitPositionFEN));
    original.DoMove(Move::Make(Square_e2, Square_e4, Piece::Pawn), 25);
    original.DoMove(Move::Make(Square_e7, Square_e5, Piece::Pawn), -30);
    original.DoMove(Move::Make(Square_g1, Square_f3, Piece::Knight), 20);

    const std::string pgnStr = original.ToPGN(true);

    std::istringstream ss(pgnStr);
    std::vector<Game> games;
    ParsePgn(ss, [&](Game& g) { games.push_back(g); return true; });

    TEST_EXPECT(games.size() == 1);
    TEST_EXPECT_EQ(games[0].GetMoves().size(), original.GetMoves().size());

    const auto& origScores = original.GetMoveScores();
    const auto& parsedScores = games[0].GetMoveScores();
    TEST_EXPECT_EQ(origScores.size(), parsedScores.size());
    for (size_t i = 0; i < origScores.size(); ++i)
        TEST_EXPECT_EQ(origScores[i], parsedScores[i]);
}

void RunPgnParserTests()
{
    std::cout << "Running PGN parser tests..." << std::endl;

    TestBasicGame();
    TestNonStandardStart();
    TestMoveScores();
    TestOpenBenchFormat();
    TestMateScore();
    TestMultipleGames();
    TestSkipBadGame();
    TestAdjudicatedResult();
    TestCastling();
    TestEarlyStop();
    TestRoundTrip();

    std::cout << "PGN parser tests done." << std::endl;
}
