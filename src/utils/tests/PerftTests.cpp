#include "TestCommon.hpp"
#include "../ThreadPool.hpp"
#include "../../backend/MoveGen.hpp"
#include "../../backend/Position.hpp"

#include <atomic>
#include <memory>

using namespace threadpool;

// one task per root move, so a single deep perft uses all threads
static void RunParallelPerft(TaskBuilder& taskBuilder, const char* fen, uint32_t depth, uint64_t expectedNodes)
{
    taskBuilder.Task("Perft", [fen, depth, expectedNodes](const TaskContext& context)
    {
        const Position pos(fen);
        MoveList moves;
        GenerateMoveList(pos, moves);

        auto nodes = std::make_shared<std::atomic<uint64_t>>(0);
        TaskBuilder builder(context);
        for (uint32_t i = 0; i < moves.Size(); ++i)
        {
            Position child = pos;
            if (child.DoMove(moves.GetMove(i)))
            {
                builder.Task("PerftMove", [child, depth, nodes](const TaskContext&) { *nodes += child.Perft(depth - 1); });
            }
        }
        builder.Fence();
        builder.Task("PerftCheck", [nodes, expectedNodes](const TaskContext&) { TEST_EXPECT(*nodes == expectedNodes); });
    });
}

// the incrementally updated hashes must match the hashes of the same position parsed from its FEN
static void CheckHashesInPerftWalk(const Position& pos, uint32_t depth)
{
    MoveList moves;
    GenerateMoveList(pos, moves);

    for (uint32_t i = 0; i < moves.Size(); ++i)
    {
        Position child = pos;
        if (!child.DoMove(moves.GetMove(i)))
        {
            continue;
        }

        const Position fromFen(child.ToFEN().CStr());
        TEST_EXPECT(child.GetHash() == child.ComputeHash());
        TEST_EXPECT(child.GetHash() == fromFen.GetHash());
        TEST_EXPECT(child.GetPawnsHash() == fromFen.GetPawnsHash());
        TEST_EXPECT(child.GetNonPawnsHash(White) == fromFen.GetNonPawnsHash(White));
        TEST_EXPECT(child.GetNonPawnsHash(Black) == fromFen.GetNonPawnsHash(Black));

        if (depth > 1)
        {
            CheckHashesInPerftWalk(child, depth - 1);
        }
    }
}

void RunPerftTests(TaskBuilder& taskBuilder, bool deep)
{
    std::cout << "Running Perft tests..." << std::endl;

    {
        taskBuilder.Task("Perft", [](const TaskContext&)
        {
            const Position pos("rnbqkbnr/1ppppppp/p7/5B2/8/3P4/PPP1PPPP/RN1QKBNR b KQkq - 0 1");
            TEST_EXPECT(pos.Perft(1) == 18u);
        });

        taskBuilder.Task("Perft", [](const TaskContext&)
        {
            const Position pos("rnbqkbnr/1ppppppp/p7/8/8/3P4/PPP1PPPP/RNBQKBNR w KQkq - 0 1");
            TEST_EXPECT(pos.Perft(2) == 511u);
        });

        taskBuilder.Task("Perft", [](const TaskContext&)
        {
            const Position pos("rnbqkbnr/pppppppp/8/8/8/3P4/PPP1PPPP/RNBQKBNR b KQkq - 0 1");
            TEST_EXPECT(pos.Perft(3) == 11959u);
        });

        taskBuilder.Task("Perft", [](const TaskContext&)
        {
            const Position pos("rnb1kbnr/pp1ppppp/1qp5/1P6/8/8/P1PPPPPP/RNBQKBNR w KQkq - 0 1");
            TEST_EXPECT(pos.Perft(1) == 21u);
        });

        taskBuilder.Task("Perft", [](const TaskContext&)
        {
            const Position pos("rnbqkbnr/pp1ppppp/2p5/1P6/8/8/P1PPPPPP/RNBQKBNR b KQkq - 0 1");
            TEST_EXPECT(pos.Perft(2) == 458u);
        });

        taskBuilder.Task("Perft", [](const TaskContext&)
        {
            const Position pos("rnbqkbnr/pp1ppppp/2p5/8/1P6/8/P1PPPPPP/RNBQKBNR w KQkq - 0 1");
            TEST_EXPECT(pos.Perft(3) == 10257u);
        });

        taskBuilder.Task("Perft", [](const TaskContext&)
        {
            const Position pos("rnbqkbnr/pppppppp/8/8/1P6/8/P1PPPPPP/RNBQKBNR b KQkq - 0 1");
            TEST_EXPECT(pos.Perft(4) == 216145u);
        });

        // initial position
        taskBuilder.Task("Perft", [](const TaskContext&)
        {
            const Position pos(Position::InitPositionFEN);
            TEST_EXPECT(pos.Perft(1) == 20u);
            TEST_EXPECT(pos.Perft(2) == 400u);
            TEST_EXPECT(pos.Perft(3) == 8902u);
            TEST_EXPECT(pos.Perft(4) == 197281u);
        });
        if (deep)
        {
            RunParallelPerft(taskBuilder, Position::InitPositionFEN, 5, 4865609u);
            RunParallelPerft(taskBuilder, Position::InitPositionFEN, 6, 119060324u);
        }

        // kings only
        taskBuilder.Task("Perft", [](const TaskContext&)
        {
            const Position pos("2k2K2/8/8/8/8/8/8/8 w - - 0 1");
            TEST_EXPECT(pos.Perft(4) == 848u);
            TEST_EXPECT(pos.Perft(6) == 29724u);
        });

        // kings + knight vs. king
        taskBuilder.Task("Perft", [](const TaskContext&)
        {
            const Position pos("2k2K2/5N2/8/8/8/8/8/8 w - - 0 1");
            TEST_EXPECT(pos.Perft(2) == 41u);
            TEST_EXPECT(pos.Perft(4) == 2293u);
            TEST_EXPECT(pos.Perft(6) == 130360u);
        });

        // kings + rook vs. king
        taskBuilder.Task("Perft", [](const TaskContext&)
        {
            const Position pos("2k2K2/5R2/8/8/8/8/8/8 w - - 0 1");
            TEST_EXPECT(pos.Perft(1) == 17u);
            TEST_EXPECT(pos.Perft(2) == 53u);
            TEST_EXPECT(pos.Perft(4) == 3917u);
            TEST_EXPECT(pos.Perft(6) == 338276u);
        });

        // kings + bishop vs. king
        taskBuilder.Task("Perft", [](const TaskContext&)
        {
            const Position pos("2k2K2/5B2/8/8/8/8/8/8 w - - 0 1");
            TEST_EXPECT(pos.Perft(2) == 58u);
            TEST_EXPECT(pos.Perft(4) == 4269u);
            TEST_EXPECT(pos.Perft(6) == 314405u);
        });

        // kings + pawn vs. king
        taskBuilder.Task("Perft", [](const TaskContext&)
        {
            const Position pos("2k3K1/4P3/8/8/8/8/8/8 w - - 0 1");
            TEST_EXPECT(pos.Perft(2) == 33u);
            TEST_EXPECT(pos.Perft(4) == 2007u);
            TEST_EXPECT(pos.Perft(6) == 136531u);
        });

        // castlings
        taskBuilder.Task("Perft", [](const TaskContext&)
        {
            const Position pos("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1");
            TEST_EXPECT(pos.Perft(1) == 26u);
            TEST_EXPECT(pos.Perft(2) == 568u);
            TEST_EXPECT(pos.Perft(4) == 314346u);
        });

        // kings + 2 queens
        taskBuilder.Task("Perft", [](const TaskContext&)
        {
            const Position pos("q3k2q/8/8/8/8/8/8/Q3K2Q w - - 0 1");
            TEST_EXPECT(pos.Perft(2) == 1040u);
            TEST_EXPECT(pos.Perft(4) == 979543u);
        });
        if (deep)
        {
            RunParallelPerft(taskBuilder, "q3k2q/8/8/8/8/8/8/Q3K2Q w - - 0 1", 6, 923005707u);
        }

        // max moves
        taskBuilder.Task("Perft", [](const TaskContext&)
        {
            const Position pos("R6R/3Q4/1Q4Q1/4Q3/2Q4Q/Q4Q2/pp1Q4/kBNN1KB1 w - - 0 1");
            TEST_EXPECT(pos.Perft(1) == 218u);
        });

        // discovered double check via en passant
        taskBuilder.Task("Perft", [](const TaskContext&)
        {
            const Position pos("8/6p1/7k/7P/5B1R/8/8/7K b - - 0 1");
            TEST_EXPECT(pos.Perft(1) == 2u);
            TEST_EXPECT(pos.Perft(2) == 35u);
            TEST_EXPECT(pos.Perft(3) == 134u);
        });

        // Position 2 - Kiwipete
        taskBuilder.Task("Perft", [](const TaskContext&)
        {
            const Position pos("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1");
            TEST_EXPECT(pos.Perft(1) == 48u);
            TEST_EXPECT(pos.Perft(2) == 2039u);
            TEST_EXPECT(pos.Perft(3) == 97862u);
            TEST_EXPECT(pos.Perft(4) == 4085603u);
        });
        if (deep)
        {
            RunParallelPerft(taskBuilder, "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 5, 193690690u);
        }

        // Position 3
        taskBuilder.Task("Perft", [](const TaskContext&)
        {
            const Position pos("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1");
            TEST_EXPECT(pos.Perft(1) == 14u);
            TEST_EXPECT(pos.Perft(2) == 191u);
            TEST_EXPECT(pos.Perft(3) == 2812u);
            TEST_EXPECT(pos.Perft(4) == 43238u);
        });
        if (deep)
        {
            RunParallelPerft(taskBuilder, "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 5, 674624u);
        }

        // Position 4
        taskBuilder.Task("Perft", [](const TaskContext&)
        {
            const Position pos("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1");
            TEST_EXPECT(pos.Perft(1) == 6u);
            TEST_EXPECT(pos.Perft(2) == 264u);
            TEST_EXPECT(pos.Perft(3) == 9467u);
            TEST_EXPECT(pos.Perft(4) == 422333u);
        });
        if (deep)
        {
            RunParallelPerft(taskBuilder, "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 5, 15833292u);
        }

        // Position 5
        taskBuilder.Task("Perft", [](const TaskContext&)
        {
            const Position pos("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8");
            TEST_EXPECT(pos.Perft(1) == 44u);
            TEST_EXPECT(pos.Perft(2) == 1486u);
            TEST_EXPECT(pos.Perft(3) == 62379u);
            TEST_EXPECT(pos.Perft(4) == 2103487u);
        });
        if (deep)
        {
            RunParallelPerft(taskBuilder, "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", 5, 89941194u);
        }

        // Position 6
        taskBuilder.Task("Perft", [](const TaskContext&)
        {
            const Position pos("r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10");
            TEST_EXPECT(pos.Perft(1) == 46u);
            TEST_EXPECT(pos.Perft(2) == 2079u);
            TEST_EXPECT(pos.Perft(3) == 89890u);
            TEST_EXPECT(pos.Perft(4) == 3894594u);
        });
        if (deep)
        {
            RunParallelPerft(taskBuilder, "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", 5, 164075551u);
        }

        // Chess960 - Position 1
        taskBuilder.Task("Perft", [](const TaskContext&)
        {
            const Position pos("bqnb1rkr/pp3ppp/3ppn2/2p5/5P2/P2P4/NPP1P1PP/BQ1BNRKR w HFhf - 2 9");
            TEST_EXPECT(pos.Perft(1) == 21u);
            TEST_EXPECT(pos.Perft(2) == 528u);
            TEST_EXPECT(pos.Perft(3) == 12189u);
            TEST_EXPECT(pos.Perft(4) == 326672u);
        });

        // Chess960 - Position 269
        taskBuilder.Task("Perft", [](const TaskContext&)
        {
            const Position pos("nrkb1qbr/pp1pppp1/5n2/7p/2p5/1N1NPP2/PPPP2PP/1RKB1QBR w HBhb - 0 9");
            TEST_EXPECT(pos.Perft(1) == 25u);
            TEST_EXPECT(pos.Perft(2) == 712u);
            TEST_EXPECT(pos.Perft(3) == 18813u);
            TEST_EXPECT(pos.Perft(4) == 543870u);
        });

        // Chess960 - Position 472
        taskBuilder.Task("Perft", [](const TaskContext&)
        {
            const Position pos("rbn1bkrq/ppppp3/4n2p/5pp1/1PN5/2P5/P2PPPPP/RBN1BKRQ w GAga - 0 9");
            TEST_EXPECT(pos.Perft(1) == 27u);
            TEST_EXPECT(pos.Perft(2) == 859u);
            TEST_EXPECT(pos.Perft(3) == 24090u);
            TEST_EXPECT(pos.Perft(4) == 796482u);
        });

        // Chess960 - Position 650
        taskBuilder.Task("Perft", [](const TaskContext&)
        {
            const Position pos("rnkrbbq1/pppppnp1/7p/8/1B1Q1p2/3P1P2/PPP1P1PP/RNKR1B1N w DAda - 2 9");
            TEST_EXPECT(pos.Perft(1) == 43u);
            TEST_EXPECT(pos.Perft(2) == 887u);
            TEST_EXPECT(pos.Perft(3) == 36240u);
            TEST_EXPECT(pos.Perft(4) == 846858u);
        });

        static const char* hashTestPositions[] =
        {
            Position::InitPositionFEN,
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
            "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
            "8/6p1/7k/7P/5B1R/8/8/7K b - - 0 1",
            "rnkrbbq1/pppppnp1/7p/8/1B1Q1p2/3P1P2/PPP1P1PP/RNKR1B1N w DAda - 2 9",
            "rbn1bkrq/ppppp3/4n2p/5pp1/1PN5/2P5/P2PPPPP/RBN1BKRQ w GAga - 0 9",
        };
        for (const char* fen : hashTestPositions)
        {
            taskBuilder.Task("PerftHash", [fen](const TaskContext&) { CheckHashesInPerftWalk(Position(fen), 3); });
        }
    }
}
