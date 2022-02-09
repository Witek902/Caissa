#include "GameCollection.hpp"
#include "../backend/Search.hpp"
#include "../backend/TranspositionTable.hpp"

#include <iostream>

#define TEST_EXPECT(x) \
    if (!(x)) { std::cout << "Test failed: " << #x << std::endl; DEBUG_BREAK(); }

static void TestGameSerialization(const Game& originalGame)
{
    std::vector<uint8_t> buffer;

    {
        MemoryOutputStream stream(buffer);
        GameCollection::Writer writer(stream);
        TEST_EXPECT(writer.WriteGame(originalGame));
    }
    TEST_EXPECT(buffer.size() > 0);

    Game readGame;
    {
        MemoryInputStream stream(buffer);
        GameCollection::Reader reader(stream);
        TEST_EXPECT(reader.ReadGame(readGame));
    }

    TEST_EXPECT(readGame == originalGame);
}

void RunGameTests()
{
    std::cout << "Running Game tests..." << std::endl;

    {
        Game game;
        TEST_EXPECT(game.GetScore() == Game::Score::Unknown);
        TEST_EXPECT(game == game);

        TestGameSerialization(game);
    }

    // game ended in checkmate
    {
        Game game;
        game.Reset(Position(Position::InitPositionFEN));
        TEST_EXPECT(game.DoMove(Move::Make(Square_f2, Square_f3, Piece::Pawn)));
        TEST_EXPECT(game.DoMove(Move::Make(Square_e7, Square_e5, Piece::Pawn)));
        TEST_EXPECT(game.DoMove(Move::Make(Square_g2, Square_g4, Piece::Pawn)));
        TEST_EXPECT(game.DoMove(Move::Make(Square_d8, Square_h4, Piece::Queen)));
        TEST_EXPECT(game.GetScore() == Game::Score::BlackWins);
        TEST_EXPECT(game.GetMoves().size() == 4);

        TestGameSerialization(game);
    }

    {
        Search search;
        TranspositionTable tt{ 16 * 1024 };

        SearchParam param{ tt };
        param.debugLog = false;
        param.numPvLines = UINT32_MAX;
        param.limits.maxDepth = 6;
        param.numPvLines = 1;

        Game game;
        game.Reset(Position(Position::InitPositionFEN));
        TEST_EXPECT(game.DoMove(Move::Make(Square_d2, Square_d4, Piece::Pawn)));
        TEST_EXPECT(game.DoMove(Move::Make(Square_e7, Square_e5, Piece::Pawn)));
        TEST_EXPECT(game.GetScore() == Game::Score::Unknown);

        SearchResult result;
        search.DoSearch(game, param, result);

        TEST_EXPECT(result.size() == 1);
        TEST_EXPECT(result[0].moves.front() == Move::Make(Square_d4, Square_e5, Piece::Pawn, Piece::None, true));
        TEST_EXPECT(result[0].score > 0);
    }
}