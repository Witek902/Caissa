#include "TestCommon.hpp"
#include "../../backend/Search.hpp"
#include "../../backend/Game.hpp"
#include "../../backend/TranspositionTable.hpp"
#include "../../backend/Score.hpp"

// this test suite runs full search on well known/easy positions
void RunSearchTests()
{
    std::cout << "Running Search tests..." << std::endl;

    Search search;
    TranspositionTable tt{ 16 * 1024 * 1024 };
    SearchResult result;
    Game game;

    SearchParam param{ tt };
    param.debugLog = false;
    param.numPvLines = UINT32_MAX;

    // insufficient material draw
    {
        param.limits.maxDepth = 4;
        param.numPvLines = UINT32_MAX;

        game.Reset(Position("4k2K/8/8/8/8/8/8/8 w - - 0 1"));
        search.DoSearch(game, param, result);

        TEST_EXPECT(result.Size() == 3);
        TEST_EXPECT(std::abs(result[0].score) <= DrawScoreRandomness);
        TEST_EXPECT(std::abs(result[1].score) <= DrawScoreRandomness);
        TEST_EXPECT(std::abs(result[2].score) <= DrawScoreRandomness);
    }

    // stalemate (no legal move)
    {
        param.limits.maxDepth = 10;
        param.numPvLines = UINT32_MAX;

        game.Reset(Position("k7/2Q5/1K6/8/8/8/8/8 b - - 0 1"));
        search.DoSearch(game, param, result);

        TEST_EXPECT(result.Size() == 0);
    }

    // checkmate (no legal move)
    {
        param.limits.maxDepth = 10;
        param.numPvLines = UINT32_MAX;

        game.Reset(Position("5k2/5Q2/1p4Bp/2b4P/5P2/5K2/P5P1/8 b - - 0 54"));
        search.DoSearch(game, param, result);

        TEST_EXPECT(result.Size() == 0);
    }

    // mate in one
    {
        param.limits.maxDepth = 16;
        param.numPvLines = UINT32_MAX;

        game.Reset(Position("k7/7Q/1K6/8/8/8/8/8 w - - 0 1"));
        search.DoSearch(game, param, result);

        TEST_EXPECT(result.Size() == 27);
        TEST_EXPECT(result[0].score == CheckmateValue - 1);
        TEST_EXPECT(result[1].score == CheckmateValue - 1);
        TEST_EXPECT(result[2].score == CheckmateValue - 1);
        TEST_EXPECT(result[3].score == CheckmateValue - 1);
    }

    // mate in one
    {
        param.limits.maxDepth = 16;
        param.numPvLines = UINT32_MAX;

        game.Reset(Position("7k/7p/2Q5/8/2Br1PK1/6P1/4P3/5q2 w - - 99 100"));
        search.DoSearch(game, param, result);

        TEST_EXPECT(result.Size() == 36);
        TEST_EXPECT(result[0].score == CheckmateValue - 1);
        TEST_EXPECT(result[1].score == 0);
    }

    // mate in two
    {
        param.limits.maxDepth = 60;
        param.limits.mateSearch = true;
        param.numPvLines = 1;

        game.Reset(Position("K4BB1/1Q6/5p2/8/2R2r1r/N2N2q1/kp1p1p1p/b7 w - - 0 1"));
        search.DoSearch(game, param, result);

        TEST_EXPECT(result.Size() == 1);
        TEST_EXPECT(result[0].score == CheckmateValue - 3);
        TEST_EXPECT(result[0].moves.Front() == Move::Make(Square_b7, Square_f3, Piece::Queen));

        param.limits.mateSearch = false;
    }

    // perpetual check
    {
        param.limits.maxDepth = 12;
        param.limits.mateSearch = true;
        param.numPvLines = 1;

        game.Reset(Position("6k1/6p1/8/6KQ/1r6/q2b4/8/8 w - - 0 1"));
        search.DoSearch(game, param, result);

        TEST_EXPECT(result.Size() == 1);
        TEST_EXPECT(std::abs(result[0].score) <= DrawScoreRandomness);
        TEST_EXPECT(result[0].moves.Front() == Move::Make(Square_h5, Square_e8, Piece::Queen));

        param.limits.mateSearch = false;
    }

    // winning KPvK
    {
        param.limits.maxDepth = 8;
        param.numPvLines = UINT32_MAX;

        game.Reset(Position("4k3/8/8/8/8/8/5P2/5K2 w - - 0 1"));
        search.DoSearch(game, param, result);

        TEST_EXPECT(result.Size() == 6);
        TEST_EXPECT(result[0].score > 1000);
        TEST_EXPECT(result[1].score > 1000);
    }

    // drawing KPvK
    {
        param.limits.maxDepth = 15;
        param.numPvLines = UINT32_MAX;

        game.Reset(Position("4k3/8/8/8/8/8/7P/7K w - - 0 1"));
        search.DoSearch(game, param, result);

        TEST_EXPECT(result.Size() == 4);
        TEST_EXPECT(std::abs(result[0].score) < 150);
        TEST_EXPECT(std::abs(result[1].score) < 150);
        TEST_EXPECT(std::abs(result[2].score) < 150);
        TEST_EXPECT(std::abs(result[3].score) < 150);
    }

    // chess-rook skewer
    {
        param.limits.maxDepth = 4;
        param.numPvLines = UINT32_MAX;

        game.Reset(Position("3k3r/8/8/8/8/8/8/KR6 w - - 0 1"));
        search.DoSearch(game, param, result);

        TEST_EXPECT(result.Size() == 15);

        TEST_EXPECT(result[0].moves.Front() == Move::Make(Square_b1, Square_b8, Piece::Rook));
        TEST_EXPECT(result[0].score >= KnownWinValue);      // Rb8 is winning

        TEST_EXPECT(result[1].score < KnownWinValue);       // draw
        TEST_EXPECT(result[13].score < KnownWinValue);      // draw

        TEST_EXPECT(result[14].moves.Front() == Move::Make(Square_b1, Square_h1, Piece::Rook));
        TEST_EXPECT(result[14].score <= -KnownWinValue);    // Rh1 is loosing
    }

    // Lasker-Reichhelm (TT test)
    {
        param.limits.maxDepth = 18;
        param.numPvLines = 1;

        game.Reset(Position("8/k7/3p4/p2P1p2/P2P1P2/8/8/K7 w - - 0 1"));
        search.DoSearch(game, param, result);

        TEST_EXPECT(result.Size() == 1);
        TEST_EXPECT(result[0].score >= 100);
        TEST_EXPECT(result[0].moves.Front() == Move::Make(Square_a1, Square_b1, Piece::King));
    }

    // search explosion test 1
    {
        param.limits.maxDepth = 1;
        param.numPvLines = 1;

        game.Reset(Position("KNnNnNnk/NnNnNnNn/nNnNnNnN/NnNnNnNn/nNnNnNnN/NnNnNnNn/nNnNnNnN/NnNnNnNn w - - 0 1"));
        search.DoSearch(game, param, result);

        TEST_EXPECT(result.Size() == 1);
    }

    /*
    // search explosion test 2
    {
        param.limits.maxDepth = 1;
        param.numPvLines = 1;

        game.Reset(Position("qQqqkqqQ/Qqqqqqqq/qQqqqqqQ/QqQqQqQq/qQqQqQqQ/QqQQQQQq/qQQQQQQQ/QqQQKQQq w - - 0 1"));
        search.DoSearch(game, param, result);

        TEST_EXPECT(result.size() == 1);
    }
    */

    // search explosion test 3
    {
        param.limits.maxDepth = 1;
        param.numPvLines = 1;

        game.Reset(Position("q2k2q1/2nqn2b/1n1P1n1b/2rnr2Q/1NQ1QN1Q/3Q3B/2RQR2B/Q2K2Q1 w - - 0 1"));
        search.DoSearch(game, param, result);

        TEST_EXPECT(result.Size() == 1);
    }

    // mate in 1 with huge material disadvantage
    {
        param.limits.maxDepth = 5;
        param.numPvLines = 1;

        game.Reset(Position("qqqqqqqq/qkqqqqqq/qqNqqqqq/qqq1qqqq/qqqq1qqq/qqqqq1qq/qqqqqqBn/qqqqqqnK w - - 0 1"));
        search.DoSearch(game, param, result);

        TEST_EXPECT(result.Size() == 1);
        TEST_EXPECT(result[0].score == CheckmateValue - 1);
        TEST_EXPECT(result[0].moves.Front() == PackedMove(Square_c6, Square_a5) ||
                    result[0].moves.Front() == PackedMove(Square_c6, Square_d8));
    }

    // mate in 1, more than 218 moves possible
    {
        param.limits.maxDepth = 8;
        param.numPvLines = 1;

        game.Reset(Position("QQQQQQBk/Q6B/Q6Q/Q6Q/Q6Q/Q6Q/Q6Q/KQQQQQQQ w - - 0 1"));
        search.DoSearch(game, param, result);

        TEST_EXPECT(result.Size() == 1);
        TEST_EXPECT(result[0].score == CheckmateValue - 1);
    }

    // mate on 50th move is a draw
    {
        param.limits.maxDepth = 10;
        param.numPvLines = 1;

        game.Reset(Position("8/6B1/8/8/2K2n2/k7/1R6/8 b - - 98 2"));
        search.DoSearch(game, param, result);

        TEST_EXPECT(result.Size() == 1);
        TEST_EXPECT(result[0].score == 0);
    }
}

void RunTranspositionTableTests()
{
    std::cout << "Running TranspositionTable tests..." << std::endl;

    TranspositionTable tt(1024 * 1024); // 1MB
    
    TEST_EXPECT(tt.GetSize() > 0);
    
    Position pos1(Position::InitPositionFEN);
    Position pos2("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1");
    
    // Write and read
    {
        TTEntry entry;
        TEST_EXPECT(!tt.Read(pos1, entry));
        TEST_EXPECT(!entry.IsValid());
        
        tt.Write(pos1, 100, 95, 5, TTEntry::Bounds::Exact, PackedMove(Square_e2, Square_e4));
        
        TEST_EXPECT(tt.Read(pos1, entry));
        TEST_EXPECT(entry.IsValid());
        TEST_EXPECT(entry.score == 100);
        TEST_EXPECT(entry.staticEval == 95);
        TEST_EXPECT(entry.depth == 5);
        TEST_EXPECT(entry.bounds == TTEntry::Bounds::Exact);
        TEST_EXPECT(entry.move == PackedMove(Square_e2, Square_e4));
    }
    
    // Different bounds
    {
        tt.Write(pos2, 200, 190, 3, TTEntry::Bounds::Lower, PackedMove::Invalid());
        TTEntry entry;
        TEST_EXPECT(tt.Read(pos2, entry));
        TEST_EXPECT(entry.bounds == TTEntry::Bounds::Lower);
        
        tt.Write(pos2, 150, 145, 4, TTEntry::Bounds::Upper, PackedMove::Invalid());
        TEST_EXPECT(tt.Read(pos2, entry));
        TEST_EXPECT(entry.bounds == TTEntry::Bounds::Upper);
        TEST_EXPECT(entry.depth == 4);
    }
    
    // Generation
    {
        tt.NextGeneration();
        TTEntry entry;
        TEST_EXPECT(tt.Read(pos1, entry));
        TEST_EXPECT(entry.IsValid());
        
        // Write new entry with new generation
        tt.Write(pos1, 110, 105, 6, TTEntry::Bounds::Exact, PackedMove(Square_e2, Square_e4));
        TEST_EXPECT(tt.Read(pos1, entry));
        TEST_EXPECT(entry.depth == 6);
    }
    
    // Clear
    {
        tt.Clear();
        TTEntry entry;
        TEST_EXPECT(!tt.Read(pos1, entry));
    }
    
    // Resize
    {
        size_t oldSize = tt.GetSize();
        tt.Resize(2 * 1024 * 1024);
        TEST_EXPECT(tt.GetSize() >= oldSize);
        
        // After resize, entries should be cleared
        TTEntry entry;
        TEST_EXPECT(!tt.Read(pos1, entry));
    }
    
    // Prefetch (should not crash)
    {
        tt.Prefetch(pos1.GetHash());
        tt.Prefetch(pos2.GetHash());
    }

    // mate and tablebase scores keep their distance to mate when written at one height and read at another
    {
        const auto writeAndRead = [&](int32_t score, int32_t writeHeight, int32_t readHeight, int32_t halfMoveCount) -> int32_t
        {
            tt.Write(pos1, ScoreToTT(ScoreType(score), writeHeight), 0, 1, TTEntry::Bounds::Exact);
            TTEntry entry;
            TEST_EXPECT(tt.Read(pos1, entry));
            return ScoreFromTT(entry.score, readHeight, halfMoveCount);
        };

        const int32_t heights[] = { 0, 1, 7, 40, 100, 150 };
        const int32_t distances[] = { 1, 2, 15, 60, 99 }; // plies from the written node to mate

        for (const int32_t writeHeight : heights)
        {
            for (const int32_t readHeight : heights)
            {
                for (const int32_t distance : distances)
                {
                    TEST_EXPECT_EQ(writeAndRead(CheckmateValue - writeHeight - distance, writeHeight, readHeight, 0), CheckmateValue - readHeight - distance);
                    TEST_EXPECT_EQ(writeAndRead(-CheckmateValue + writeHeight + distance, writeHeight, readHeight, 0), -CheckmateValue + readHeight + distance);
                    TEST_EXPECT_EQ(writeAndRead(TablebaseWinValue - writeHeight - distance, writeHeight, readHeight, 0), TablebaseWinValue - readHeight - distance);
                    TEST_EXPECT_EQ(writeAndRead(-TablebaseWinValue + writeHeight + distance, writeHeight, readHeight, 0), -TablebaseWinValue + readHeight + distance);
                }

                // mate further away than the 50-move rule allows becomes a non-mate win or loss
                TEST_EXPECT_EQ(writeAndRead(CheckmateValue - writeHeight - 9, writeHeight, readHeight, 90), CheckmateValue - readHeight - 9);
                TEST_EXPECT_EQ(writeAndRead(-CheckmateValue + writeHeight + 9, writeHeight, readHeight, 90), -CheckmateValue + readHeight + 9);
                TEST_EXPECT_EQ(writeAndRead(CheckmateValue - writeHeight - 10, writeHeight, readHeight, 90), CheckmateValue - MaxSearchDepth - 1);
                TEST_EXPECT_EQ(writeAndRead(-CheckmateValue + writeHeight + 10, writeHeight, readHeight, 90), -CheckmateValue + MaxSearchDepth + 1);
                TEST_EXPECT(!IsMate(ScoreType(writeAndRead(CheckmateValue - writeHeight - 10, writeHeight, readHeight, 90))));

                // other scores do not depend on the height
                const int32_t plainScores[] = { 0, 1, -1, 250, -250, KnownWinValue, -KnownWinValue, TablebaseWinValue - MaxSearchDepth - 1, -TablebaseWinValue + MaxSearchDepth + 1 };
                for (const int32_t score : plainScores)
                {
                    TEST_EXPECT_EQ(writeAndRead(score, writeHeight, readHeight, 0), score);
                }
            }
        }
    }
}

void RunScoreTests()
{
    std::cout << "Running Score tests..." << std::endl;

    // ScoreToStr
    {
        TEST_EXPECT(ScoreToStr(CheckmateValue - 1) == "+M1");
        TEST_EXPECT(ScoreToStr(CheckmateValue - 3) == "+M2");
        TEST_EXPECT(ScoreToStr(-CheckmateValue + 1) == "-M1");
        TEST_EXPECT(ScoreToStr(-CheckmateValue + 5) == "-M3");
        TEST_EXPECT(ScoreToStr(150) == "+1.50");
        TEST_EXPECT(ScoreToStr(-250) == "-2.50");
        TEST_EXPECT(ScoreToStr(0) == "+0.00");
    }

    // IsMate
    {
        TEST_EXPECT(IsMate(CheckmateValue - 1));
        TEST_EXPECT(IsMate(CheckmateValue - 10));
        TEST_EXPECT(IsMate(-CheckmateValue + 1));
        TEST_EXPECT(IsMate(-CheckmateValue + 10));
        TEST_EXPECT(!IsMate(1000));
        TEST_EXPECT(!IsMate(-1000));
        TEST_EXPECT(!IsMate(0));
    }
}
