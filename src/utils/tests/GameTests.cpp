#include "TestCommon.hpp"
#include "../GameCollection.hpp"
#include "../../backend/Search.hpp"

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

// plays space-separated moves in LAN notation
static void PlayMoves(Game& game, const char* moves)
{
    std::istringstream stream(moves);
    std::string moveStr;
    while (stream >> moveStr)
    {
        const Move move = game.GetPosition().MoveFromString(moveStr.c_str());
        TEST_EXPECT(move.IsValid());
        TEST_EXPECT(game.DoMove(move));
    }
}

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
        std::vector<Move> moves;
        TEST_EXPECT(GameCollection::ReadGame(stream, readGame, moves));
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
        TEST_EXPECT(game.GetMoves().Size() == 4);

        TestGameSerialization(game);
    }

    // game ended in stalemate
    {
        Game game;
        game.Reset(Position("4K2k/8/6q1/4P3/6Q1/8/8/8 w - - 27 74"));

        const Move move = game.GetPosition().MoveFromString("g4g6");
        TEST_EXPECT(move.IsValid());

        TEST_EXPECT(game.DoMove(move));
        TEST_EXPECT(game.GetScore() == Game::Score::Draw);
        TEST_EXPECT(game.GetPosition().IsStalemate());

        TestGameSerialization(game);
    }

    {
        Search search;
        TranspositionTable tt{ 16 * 1024 };

        SearchParam param{ tt };
        param.debugLog = false;
        param.numPvLines = UINT32_MAX;
        param.limits.maxDepth = 10;
        param.numPvLines = 1;

        Game game;
        game.Reset(Position(Position::InitPositionFEN));
        TEST_EXPECT(game.DoMove(Move::Make(Square_d2, Square_d4, Piece::Pawn)));
        TEST_EXPECT(game.DoMove(Move::Make(Square_e7, Square_e5, Piece::Pawn)));
        TEST_EXPECT(game.GetScore() == Game::Score::Unknown);

        SearchResult result;
        search.DoSearch(game, param, result);

        TEST_EXPECT(result.Size() == 1);
        TEST_EXPECT(result[0].moves.Front() == Move::Make(Square_d4, Square_e5, Piece::Pawn, Piece::None, true));
        TEST_EXPECT(result[0].score > 0);
    }

    // threefold repetition
    {
        Game game;
        game.Reset(Position(Position::InitPositionFEN));
        const Position start = game.GetPosition();
        TEST_EXPECT(game.GetRepetitionCount(start) == 1);

        PlayMoves(game, "g1f3 g8f6 f3g1 f6g8");
        TEST_EXPECT(game.GetRepetitionCount(start) == 2);
        TEST_EXPECT(!game.IsDrawn());

        PlayMoves(game, "g1f3 g8f6 f3g1");
        TEST_EXPECT(!game.IsDrawn());

        PlayMoves(game, "f6g8");
        TEST_EXPECT(game.GetRepetitionCount(start) == 3);
        TEST_EXPECT(game.IsDrawn());
        TEST_EXPECT(game.GetScore() == Game::Score::Draw);
    }

    // lost castling rights make it a different position
    {
        Game game;
        game.Reset(Position("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"));
        const Position start = game.GetPosition();

        PlayMoves(game, "a1a2 a8a7 a2a1 a7a8");
        TEST_EXPECT(game.GetRepetitionCount(start) == 1);
        TEST_EXPECT(game.GetRepetitionCount(game.GetPosition()) == 1);

        PlayMoves(game, "a1a2 a8a7 a2a1 a7a8 a1a2 a8a7 a2a1 a7a8");
        TEST_EXPECT(game.GetRepetitionCount(game.GetPosition()) == 3);
        TEST_EXPECT(game.IsDrawn());
    }

    // an en passant right makes it a different position
    {
        Game game;
        game.Reset(Position("4k3/3p4/8/4P3/8/8/8/4K3 b - - 0 1"));
        PlayMoves(game, "d7d5");
        const Position afterPush = game.GetPosition();
        TEST_EXPECT(afterPush.GetEnPassantSquare() == Square_d6);

        PlayMoves(game, "e1d1 e8d8 d1e1 d8e8");
        TEST_EXPECT(game.GetRepetitionCount(afterPush) == 1);
        TEST_EXPECT(game.GetRepetitionCount(game.GetPosition()) == 1);
    }

    // a double push that no pawn can capture gives no en passant right
    {
        Game game;
        game.Reset(Position("4k3/3p4/8/8/8/8/8/4K3 b - - 0 1"));
        PlayMoves(game, "d7d5");
        const Position afterPush = game.GetPosition();

        PlayMoves(game, "e1d1 e8d8 d1e1 d8e8");
        TEST_EXPECT(game.GetRepetitionCount(afterPush) == 2);
    }

    // 50-move rule
    {
        Game game;
        game.Reset(Position("4k3/8/8/8/8/8/8/R3K3 w - - 99 80"));
        TEST_EXPECT(!game.IsDrawn());
        PlayMoves(game, "e1d1");
        TEST_EXPECT(game.IsDrawn());
        TEST_EXPECT(game.GetScore() == Game::Score::Draw);
    }

    // a pawn move or a capture resets the 50-move counter
    {
        Game game;
        game.Reset(Position("4k3/8/8/8/8/8/P7/R3K3 w - - 99 80"));
        PlayMoves(game, "a2a3");
        TEST_EXPECT(game.GetPosition().GetHalfMoveCount() == 0);
        TEST_EXPECT(!game.IsDrawn());

        game.Reset(Position("4k3/8/8/8/8/8/p7/R3K3 w - - 99 80"));
        PlayMoves(game, "a1a2");
        TEST_EXPECT(game.GetPosition().GetHalfMoveCount() == 0);
        TEST_EXPECT(!game.IsDrawn());
    }

    // checkmate on the 100th half-move wins
    {
        Game game;
        game.Reset(Position("7k/8/6K1/8/8/8/8/R7 w - - 99 80"));
        PlayMoves(game, "a1a8");
        TEST_EXPECT(game.GetPosition().IsMate());
        TEST_EXPECT(!game.IsDrawn());
        TEST_EXPECT(game.GetScore() == Game::Score::WhiteWins);
    }

    // a check that is not mate on the 100th half-move is a draw
    {
        Game game;
        game.Reset(Position("7k/8/5K2/8/8/8/8/R7 w - - 99 80"));
        PlayMoves(game, "a1a8");
        TEST_EXPECT(game.GetPosition().IsInCheck());
        TEST_EXPECT(game.IsDrawn());
        TEST_EXPECT(game.GetScore() == Game::Score::Draw);
    }
}