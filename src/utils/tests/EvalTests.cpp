#include "TestCommon.hpp"
#include "../../backend/Evaluate.hpp"
#include "../../backend/MoveGen.hpp"
#include "../../backend/Position.hpp"

void RunEvalTests()
{
    TEST_EXPECT(Evaluate(Position("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")) > 0);
    TEST_EXPECT(Evaluate(Position("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1")) > 0);
    TEST_EXPECT(Evaluate(Position("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")) == Evaluate(Position("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1")));
    TEST_EXPECT(Evaluate(Position("r6r/1p3p2/1n1p1kpp/pPpPp1nP/P1P1PqPR/4NP2/3NK2R/Q7 w - - 0 1")) == Evaluate(Position("q7/3nk2r/4np2/p1p1pQpr/PpPpP1Np/1N1P1KPP/1P3P2/R6R b - - 0 1")));

    // KvK
    TEST_EXPECT(0 == Evaluate(Position("K7/8/8/8/8/8/8/7k w - - 0 1")));
    TEST_EXPECT(0 == Evaluate(Position("K7/8/8/8/8/8/8/7k w - - 0 1")));

    // KvB
    TEST_EXPECT(0 == Evaluate(Position("K7/8/8/8/8/8/8/6bk w - - 0 1")));
    TEST_EXPECT(0 == Evaluate(Position("K7/8/8/8/8/8/8/6bk b - - 0 1")));
    TEST_EXPECT(0 == Evaluate(Position("K7/B7/8/8/8/8/8/7k w - - 0 1")));
    TEST_EXPECT(0 == Evaluate(Position("K7/B7/8/8/8/8/8/7k b - - 0 1")));

    // KvN
    TEST_EXPECT(0 == Evaluate(Position("K7/8/8/8/8/8/8/6nk w - - 0 1")));
    TEST_EXPECT(0 == Evaluate(Position("K7/8/8/8/8/8/8/6nk b - - 0 1")));
    TEST_EXPECT(0 == Evaluate(Position("K7/N7/8/8/8/8/8/7k w - - 0 1")));
    TEST_EXPECT(0 == Evaluate(Position("K7/N7/8/8/8/8/8/7k b - - 0 1")));

    // KvNN
    TEST_EXPECT(0 == Evaluate(Position("K7/N7/N7/8/8/8/8/7k w - - 0 1")));
    TEST_EXPECT(0 == Evaluate(Position("K7/N7/N7/8/8/8/8/7k b - - 0 1")));
    TEST_EXPECT(0 == Evaluate(Position("K7/8/8/8/8/8/8/5nnk w - - 0 1")));
    TEST_EXPECT(0 == Evaluate(Position("K7/8/8/8/8/8/8/5nnk b - - 0 1")));

    // KNvKN
    TEST_EXPECT(0 == Evaluate(Position("n6k/8/8/8/3NK3/8/8/8 w - - 0 1")));
    TEST_EXPECT(0 == Evaluate(Position("N6K/8/8/8/3nk3/8/8/8 w - - 0 1")));
    TEST_EXPECT(0 == Evaluate(Position("n6k/8/8/8/3NK3/8/8/8 b - - 0 1")));
    TEST_EXPECT(0 == Evaluate(Position("N6K/8/8/8/3nk3/8/8/8 b - - 0 1")));

    // KvBB (same color)
    TEST_EXPECT(0 == Evaluate(Position("KB6/B7/8/8/8/8/8/7k w - - 0 1")));
    TEST_EXPECT(0 == Evaluate(Position("KB6/B7/8/8/8/8/8/7k b - - 0 1")));
    TEST_EXPECT(0 == Evaluate(Position("K7/8/8/8/8/8/7b/6bk w - - 0 1")));
    TEST_EXPECT(0 == Evaluate(Position("K7/8/8/8/8/8/7b/6bk b - - 0 1")));

    // KvBB (opposite colors)
    TEST_EXPECT(KnownWinValue <= Evaluate(Position("K7/B7/B7/8/8/8/8/7k w - - 0 1")));
    TEST_EXPECT(-KnownWinValue >= Evaluate(Position("K7/B7/B7/8/8/8/8/7k b - - 0 1")));
    TEST_EXPECT(-KnownWinValue >= Evaluate(Position("K7/8/8/8/8/7b/7b/7k w - - 0 1")));
    TEST_EXPECT(-KnownWinValue >= Evaluate(Position("K7/8/8/8/8/7b/7b/7k w - - 0 1")));

    // KvR
    TEST_EXPECT(KnownWinValue <= Evaluate(Position("K7/R7/8/8/8/8/8/7k w - - 0 1")));
    TEST_EXPECT(KnownWinValue <= Evaluate(Position("K7/R7/8/8/8/8/8/7k w - - 0 1")));
    TEST_EXPECT(-KnownWinValue >= Evaluate(Position("K7/8/8/8/8/8/8/6rk w - - 0 1")));
    TEST_EXPECT(-KnownWinValue >= Evaluate(Position("K7/8/8/8/8/8/8/6rk w - - 0 1")));
    TEST_EXPECT(-KnownWinValue >= Evaluate(Position("8/8/8/8/8/8/6k1/KRR5 b - - 0 1")));

    // KvQ
    TEST_EXPECT(KnownWinValue <= Evaluate(Position("K7/Q7/8/8/8/8/8/7k w - - 0 1")));
    TEST_EXPECT(KnownWinValue <= Evaluate(Position("K7/Q7/8/8/8/8/8/7k w - - 0 1")));
    TEST_EXPECT(-KnownWinValue >= Evaluate(Position("K7/8/8/8/8/8/8/6qk w - - 0 1")));
    TEST_EXPECT(-KnownWinValue >= Evaluate(Position("K7/8/8/8/8/8/8/6qk w - - 0 1")));

    // KQvKQ
    TEST_EXPECT(Evaluate(Position("q5k1/8/8/8/8/8/7K/QQ6 w - - 0 1")) > Evaluate(Position("q5k1/8/8/8/8/8/7K/Q7 w - - 0 1")));

    // KRvKR
    TEST_EXPECT(Evaluate(Position("r5k1/8/8/8/8/8/7K/RR6 w - - 0 1")) > Evaluate(Position("r5k1/8/8/8/8/8/7K/R7 w - - 0 1")));

    // KvP (white winning)
    TEST_EXPECT(KnownWinValue <= Evaluate(Position("7k/8/8/8/8/8/P7/K7 w - - 0 1")));
    TEST_EXPECT(-KnownWinValue >= Evaluate(Position("7k/8/8/8/8/8/P7/K7 b - - 0 1")));
    TEST_EXPECT(KnownWinValue <= Evaluate(Position("8/8/1k6/8/8/1K6/1P6/8 w - - 0 1")));
    TEST_EXPECT(0 == Evaluate(Position("8/8/1k6/8/8/1K6/1P6/8 b - - 0 1")));
    TEST_EXPECT(0 == Evaluate(Position("5k2/8/8/8/8/8/P7/K7 w - - 0 1")));
    TEST_EXPECT(0 == Evaluate(Position("5k2/8/8/8/8/8/P7/K7 w - - 0 1")));

    // KvP (black winning)
    TEST_EXPECT(-KnownWinValue >= Evaluate(Position("7k/7p/8/8/8/8/8/K7 w - - 0 1")));
    TEST_EXPECT(KnownWinValue <= Evaluate(Position("7k/7p/8/8/8/8/8/K7 b - - 0 1")));
    TEST_EXPECT(KnownWinValue <= Evaluate(Position("8/6p1/6k1/8/8/6K1/8/8 b - - 0 1")));
    TEST_EXPECT(0 == Evaluate(Position("8/6p1/6k1/8/8/6K1/8/8 w - - 0 1")));
    TEST_EXPECT(0 == Evaluate(Position("7k/7p/8/8/8/8/8/2K5 w - - 0 1")));
    TEST_EXPECT(0 == Evaluate(Position("7k/7p/8/8/8/8/8/2K5 b - - 0 1")));

    // KvPs (white winning)
    TEST_EXPECT(KnownWinValue < Evaluate(Position("8/5k1P/7P/8/8/8/8/K7 w - - 0 1")));
    TEST_EXPECT(KnownWinValue < Evaluate(Position("7K/8/5k1P/8/8/7P/8/8 w - - 0 1")));
    TEST_EXPECT(0 < Evaluate(Position("4k3/8/7P/6KP/7P/7P/7P/8 w - - 0 1")));
    TEST_EXPECT(KnownWinValue < Evaluate(Position("1k6/1P6/P7/8/8/8/8/K7 w - - 0 1")));

    // KvPs (draw)
    TEST_EXPECT(0 == Evaluate(Position("8/8/5k2/7P/1K6/7P/8/8 w - - 0 1")));
    TEST_EXPECT(0 == Evaluate(Position("8/6k1/8/6KP/7P/7P/7P/8 w - - 0 1")));
    TEST_EXPECT(0 == Evaluate(Position("8/6k1/8/6KP/7P/7P/7P/8 w - - 0 1")));
    TEST_EXPECT(0 == Evaluate(Position("6k1/8/7P/6KP/7P/7P/7P/8 w - - 0 1")));

    // KBPvK (drawn)
    TEST_EXPECT(0 == Evaluate(Position("k7/P7/8/K7/3B4/8/P7/B7 w - - 0 1")));
    TEST_EXPECT(0 == Evaluate(Position("7k/7P/8/8/2B5/3B4/7P/6K1 w - - 0 1")));
    TEST_EXPECT(0 == Evaluate(Position("b7/p7/8/3b4/k7/8/p7/K7 b - - 0 1")));
    TEST_EXPECT(0 == Evaluate(Position("6k1/7p/3b4/2b5/8/8/7p/7K b - - 0 1")));
    //TEST_EXPECT(0 == Evaluate(Position("8/8/3k4/8/8/P7/7B/7K b - - 0 1")));
    //TEST_EXPECT(0 == Evaluate(Position("8/8/5k2/8/8/7P/B7/K7 b - - 0 1")));
    //TEST_EXPECT(0 == Evaluate(Position("8/8/5k2/8/8/7P/B6P/K7 b - - 0 1")));
    //TEST_EXPECT(0 == Evaluate(Position("2k5/8/8/8/8/8/B6P/K7 w - - 0 1")));
    //TEST_EXPECT(0 == Evaluate(Position("2k5/8/8/8/8/8/B6P/K7 b - - 0 1")));
    //TEST_EXPECT(0 == Evaluate(Position("1k6/8/8/8/8/8/B6P/K7 b - - 0 1")));
    //TEST_EXPECT(0 == Evaluate(Position("5k2/8/8/8/8/8/P6B/7K w - - 0 1")));
    //TEST_EXPECT(0 == Evaluate(Position("4k3/8/8/8/8/7K/B6P/8 w - - 0 1")));
    //TEST_EXPECT(0 == Evaluate(Position("4k3/8/8/8/8/7K/B6P/8 b - - 0 1")));
    //TEST_EXPECT(0 == Evaluate(Position("4k3/8/8/8/8/7K/B6P/8 b - - 0 1")));
    //TEST_EXPECT(0 == Evaluate(Position("4k3/8/8/8/7K/8/B6P/8 b - - 0 1")));
    //TEST_EXPECT(0 == Evaluate(Position("7k/8/6K1/8/8/7P/2B4P/8 w - - 0 1")));

    // KBPvK (winning)
    TEST_EXPECT(0 < Evaluate(Position("7k/7P/8/8/2B5/3B4/6P1/6K1 w - - 0 1")));
    TEST_EXPECT(0 < Evaluate(Position("7k/7P/8/8/2B5/8/3B3P/6K1 w - - 0 1")));
    TEST_EXPECT(0 < Evaluate(Position("k7/P7/8/8/5B2/4B3/1P6/1K6 w - - 0 1")));
    TEST_EXPECT(0 < Evaluate(Position("k7/P7/8/8/5B2/8/P3B3/1K6 w - - 0 1")));
    TEST_EXPECT(0 > Evaluate(Position("8/8/4k2P/8/8/8/B7/K7 b - - 0 1")));
    TEST_EXPECT(0 < Evaluate(Position("1k6/8/8/8/8/8/B6P/K7 w - - 0 1")));
    TEST_EXPECT(0 < Evaluate(Position("6k1/8/8/8/8/8/P6B/7K w - - 0 1")));
    TEST_EXPECT(0 < Evaluate(Position("4k3/8/8/7K/8/8/B6P/8 w - - 0 1")));
    TEST_EXPECT(0 < Evaluate(Position("4k3/8/8/8/7K/8/B6P/8 w - - 0 1")));
    TEST_EXPECT(0 > Evaluate(Position("4k3/8/8/7K/8/8/B6P/8 b - - 0 1")));

    // KNNNvK
    TEST_EXPECT(Evaluate(Position("3k4/8/8/8/8/8/8/2NKNN2 w - - 0 1")) >= KnownWinValue);
    TEST_EXPECT(Evaluate(Position("3k4/8/8/8/8/8/8/2NKNN2 b - - 0 1")) <= -KnownWinValue);

    // KBBvK
    TEST_EXPECT(Evaluate(Position("3k4/8/8/8/8/8/8/3KBB2 w - - 0 1")) >= KnownWinValue);
    TEST_EXPECT(Evaluate(Position("3k4/8/8/8/8/8/8/3KBB2 b - - 0 1")) <= -KnownWinValue);

    // KBBBvK
    TEST_EXPECT(Evaluate(Position("3k4/8/8/8/8/8/8/2BKBB2 w - - 0 1")) >= KnownWinValue);
    TEST_EXPECT(Evaluate(Position("3k4/8/8/8/8/8/8/2BKBB2 b - - 0 1")) <= -KnownWinValue);

    // KBBPvK
    TEST_EXPECT(Evaluate(Position("k7/8/8/8/8/8/P7/3KBB2 w - - 0 1")) >= KnownWinValue);
    TEST_EXPECT(Evaluate(Position("k7/8/8/8/8/8/P7/3KBB2 b - - 0 1")) <= -KnownWinValue);

    // KPPvK
    TEST_EXPECT(Evaluate(Position("K7/8/8/8/7k/7P/6P1/8 w - - 0 1")) >= KnownWinValue);
    TEST_EXPECT(Evaluate(Position("K7/8/8/3PP3/4k3/8/8/8 w - - 0 1")) >= KnownWinValue);
    TEST_EXPECT(Evaluate(Position("8/8/8/8/8/6P1/5Pk1/K7 b - - 0 1")) <= -KnownWinValue);

    // extreme imbalance
    {
        {
            const ScoreType score = Evaluate(Position("QQQQQQpk/QQQQQQpp/QQQQQQQQ/QQQQQQQQ/QQQQQQQQ/QQQQQQQQ/QQQQQQQQ/KQQQQQQQ w - - 0 1"));
            TEST_EXPECT(score > 1000);
            TEST_EXPECT(score < KnownWinValue);
        }
        {
            const ScoreType score = Evaluate(Position("qqqqkqqq/qqqqqqqq/qqqqqqqq/qqqqqqqq/pppppppp/8/PPPPPPPP/4K3 w - - 0 1"));
            TEST_EXPECT(score < -1000);
            TEST_EXPECT(score > -KnownWinValue);
        }
        {
            const ScoreType score = Evaluate(Position("RRRRRRpk/RRRRRRpp/RRRRRRRR/RRRRRRRR/RRRRRRRR/RRRRRRRR/RRRRRRRR/KRRRRRRR w - - 0 1"));
            TEST_EXPECT(score > 1000);
            TEST_EXPECT(score < KnownWinValue);
        }
        {
            const ScoreType score = Evaluate(Position("rrrrkrrr/rrrrrrrr/rrrrrrrr/rrrrrrrr/pppppppp/8/PPPPPPPP/4K3 w - - 0 1"));
            TEST_EXPECT(score < -1000);
            TEST_EXPECT(score > -KnownWinValue);
        }
    }

    // pawns endgame
    TEST_EXPECT(Evaluate(Position("k7/p7/8/8/8/8/PP6/K7 w - - 0 1")) >= 0);
    TEST_EXPECT(Evaluate(Position("k7/p7/8/8/8/8/PPP5/K7 w - - 0 1")) > 0);
    TEST_EXPECT(Evaluate(Position("k7/p7/8/8/8/8/PPP5/K7 b - - 0 1")) < 0);
    TEST_EXPECT(Evaluate(Position("k7/pp6/8/8/8/8/PPP5/K7 w - - 0 1")) >= 0);
    TEST_EXPECT(Evaluate(Position("k7/pp6/8/8/8/8/PPP5/K7 w - - 0 1")) >= 0);
    TEST_EXPECT(Evaluate(Position("k7/p7/8/8/8/8/PPPP4/K7 w - - 0 1")) > 0);
    TEST_EXPECT(Evaluate(Position("k7/p7/8/8/8/8/PPPP4/K7 b - - 0 1")) < 0);

    // queen vs. weaker piece
    TEST_EXPECT(Evaluate(Position("3rk3/8/8/8/8/8/8/2Q1K3 w - - 0 1")) > 0);
    TEST_EXPECT(Evaluate(Position("3rk3/8/8/8/8/8/8/2Q1K3 b - - 0 1")) < 0);
    TEST_EXPECT(Evaluate(Position("3nk3/8/8/8/8/8/8/2Q1K3 w - - 0 1")) > 0);
    TEST_EXPECT(Evaluate(Position("3nk3/8/8/8/8/8/8/2Q1K3 b - - 0 1")) < 0);
    TEST_EXPECT(Evaluate(Position("3bk3/8/8/8/8/8/8/2Q1K3 w - - 0 1")) > 0);
    TEST_EXPECT(Evaluate(Position("3bk3/8/8/8/8/8/8/2Q1K3 b - - 0 1")) < 0);
    TEST_EXPECT(Evaluate(Position("4k3/3p4/8/8/8/8/8/2Q1K3 w - - 0 1")) >= KnownWinValue);
    TEST_EXPECT(Evaluate(Position("4k3/3p4/8/8/8/8/8/2Q1K3 b - - 0 1")) < 0);

    TEST_EXPECT(Evaluate(Position("2Q5/8/8/8/3n4/8/1b6/k2K4 b - - 0 1")) == 0);
    TEST_EXPECT(Evaluate(Position("2Q3b1/6n1/8/8/8/8/3K4/k7 w - - 0 1")) > 0);

    // the eval is from the side to move's view, so swapping the colours must not change it
    {
        const char* fens[] =
        {
            Position::InitPositionFEN,
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
            "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
            "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
            "r3k2r/8/8/8/8/8/8/R4K1R b kq - 0 1",
            "rbn1bkrq/ppppp3/4n2p/5pp1/1PN5/2P5/P2PPPPP/RBN1BKRQ w GAga - 0 9",
            "6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1",
            "8/8/4k3/8/8/3K4/3P4/8 w - - 0 1",
            "8/8/8/4k3/8/8/2B5/K1N5 b - - 0 1",
            "8/5k2/8/8/8/2KQ4/8/3r4 b - - 0 1",
            "8/2k5/8/8/3p4/8/4K3/3B4 w - - 0 1",
            "2Q3b1/6n1/8/8/8/8/3K4/k7 w - - 0 1",
        };

        for (const char* fen : fens)
        {
            const Position pos(fen);
            MoveList moves;
            GenerateMoveList(pos, moves);

            // the position itself and all its children
            for (uint32_t i = 0; i <= moves.Size(); ++i)
            {
                Position p = pos;
                if (i < moves.Size() && !p.DoMove(moves.GetMove(i)))
                {
                    continue;
                }

                const Position swapped(p.SwappedColors().ToFEN().CStr());
                TEST_EXPECT_EQ(Evaluate(p), Evaluate(swapped));
            }
        }
    }
}
