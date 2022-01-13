#include <iostream>
#include "../backend/Position.hpp"
#include "../backend/MoveList.hpp"
#include "../backend/Search.hpp"
#include "../backend/TranspositionTable.hpp"
#include "../backend/Evaluate.hpp"
#include "../backend/Tablebase.hpp"
#include "../backend/Game.hpp"
#include "../backend/ThreadPool.hpp"

#include "../backend/nnue-probe/nnue.h"

#include <chrono>
#include <mutex>
#include <fstream>
#include <sstream>
#include <iterator>

using namespace threadpool;

#define TEST_EXPECT(x) \
    if (!(x)) { std::cout << "Test failed: " << #x << std::endl; DEBUG_BREAK(); }

void RunPerft()
{
    const Position pos("r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10");

    auto start = std::chrono::high_resolution_clock::now();
    //TEST_EXPECT(pos.Perft(4) == 3894594u);
    TEST_EXPECT(pos.Perft(5) == 164075551u);
    auto finish = std::chrono::high_resolution_clock::now();

    std::cout << "Elapsed time: " << std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count() / 1000000.0 << " s\n";
}

static void RunPositionTests()
{
    std::cout << "Running Position tests..." << std::endl;

    // empty board
    TEST_EXPECT(!Position().IsValid());

    // FEN parsing
    {
        // initial position
        TEST_EXPECT(Position().FromFEN(Position::InitPositionFEN));

        // only kings
        TEST_EXPECT(Position().FromFEN("4k3/8/8/8/8/8/8/4K3 w - - 0 1"));

        // invalid castling rights
        TEST_EXPECT(!Position().FromFEN("r3k3/8/8/8/8/8/8/R3K2R w k - 0 1"));
        TEST_EXPECT(!Position().FromFEN("4k2r/8/8/8/8/8/8/R3K2R w q - 0 1"));
        TEST_EXPECT(!Position().FromFEN("r3k2r/8/8/8/8/8/8/R3K3 w K - 0 1"));
        TEST_EXPECT(!Position().FromFEN("r3k2r/8/8/8/8/8/8/4K2R w Q - 0 1"));

        // some random position
        TEST_EXPECT(Position().FromFEN("4r1rk/1p5q/4Rb2/2pQ1P2/7p/5B2/P4P1B/7K b - - 4 39"));

        // not enough kings
        TEST_EXPECT(!Position().FromFEN("k7/8/8/8/8/8/8/8 w - - 0 1"));
        TEST_EXPECT(!Position().FromFEN("K7/8/8/8/8/8/8/8 w - - 0 1"));

        // pawn at invalid position
        TEST_EXPECT(!Position().FromFEN("rnbqkbpr/ppppppnp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"));
        TEST_EXPECT(!Position().FromFEN("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPNP/RNBQKBPR w KQkq - 0 1"));

        // opponent side can't be in check
        TEST_EXPECT(!Position().FromFEN("k6Q/8/8/8/8/8/8/K7 w - - 0 1"));

        // valid en passant square
        {
            Position p;
            TEST_EXPECT(p.FromFEN("rnbqkbnr/1pp1pppp/p7/3pP3/8/8/PPPP1PPP/RNBQKBNR w Qkq d6 0 3"));
            TEST_EXPECT(p.GetEnPassantSquare() == Square_d6);
        }

        // invalid en passant sqaure
        TEST_EXPECT(!Position().FromFEN("rnbqkbnr/1pp1pppp/p7/3pP3/8/8/PPPP1PPP/RNBQKBNR w Qkq e6 0 3"));
    }

    // FEN printing
    {
        Position pos(Position::InitPositionFEN);
        TEST_EXPECT(pos.ToFEN() == Position::InitPositionFEN);
    }

    // hash
    {
        TEST_EXPECT(Position("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").GetHash() != Position("rnbqkbnr/pppppppp/8/8/8/8/1PPPPPPP/RNBQKBNR w KQkq - 0 1").GetHash());
        TEST_EXPECT(Position("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").GetHash() != Position("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w Qkq - 0 1").GetHash());
        TEST_EXPECT(Position("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").GetHash() != Position("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w Kkq - 0 1").GetHash());
        TEST_EXPECT(Position("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").GetHash() != Position("rnbqkbnr/pppppppp/8/8/8/8/1PPPPPPP/RNBQKBNR w KQq - 0 1").GetHash());
        TEST_EXPECT(Position("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").GetHash() != Position("rnbqkbnr/pppppppp/8/8/8/8/1PPPPPPP/RNBQKBNR w KQk - 0 1").GetHash());

        TEST_EXPECT(Position("rnbqkbnr/1pp1pppp/p7/3pP3/8/8/PPPP1PPP/RNBQKBNR w Qkq d6 0 3").GetHash() != Position("rnbqkbnr/1pp1pppp/p7/3pP3/8/8/PPPP1PPP/RNBQKBNR w Qkq - 0 3").GetHash());
    }

    // king moves
    {
        // king moves (a1)
        {
            Position pos("k7/8/8/8/8/8/8/K7 w - - 0 1");
            MoveList moveList; pos.GenerateMoveList(moveList);
            TEST_EXPECT(moveList.Size() == 3u);
        }

        // king moves (h1)
        {
            Position pos("k7/8/8/8/8/8/8/7K w - - 0 1");
            MoveList moveList; pos.GenerateMoveList(moveList);
            TEST_EXPECT(moveList.Size() == 3u);
        }

        // king moves (h8)
        {
            Position pos("k6K/8/8/8/8/8/8/8 w - - 0 1");
            MoveList moveList; pos.GenerateMoveList(moveList);
            TEST_EXPECT(moveList.Size() == 3u);
        }

        // king moves (a1)
        {
            Position pos("K7/8/8/8/8/8/8/k7 w - - 0 1");
            MoveList moveList; pos.GenerateMoveList(moveList);
            TEST_EXPECT(moveList.Size() == 3u);
        }

        // king moves (b1)
        {
            Position pos("k7/8/8/8/8/8/8/1K6 w - - 0 1");
            MoveList moveList; pos.GenerateMoveList(moveList);
            TEST_EXPECT(moveList.Size() == 5u);
        }

        // king moves (h2)
        {
            Position pos("k7/8/8/8/8/8/7K/8 w - - 0 1");
            MoveList moveList; pos.GenerateMoveList(moveList);
            TEST_EXPECT(moveList.Size() == 5u);
        }

        // king moves (g8)
        {
            Position pos("k5K1/8/8/8/8/8/8/8 w - - 0 1");
            MoveList moveList; pos.GenerateMoveList(moveList);
            TEST_EXPECT(moveList.Size() == 5u);
        }

        // king moves (a7)
        {
            Position pos("8/K7/8/8/8/8/8/7k w - - 0 1");
            MoveList moveList; pos.GenerateMoveList(moveList);
            TEST_EXPECT(moveList.Size() == 5u);
        }

        // king moves (d5)
        {
            Position pos("8/8/8/3K4/8/8/8/7k w - - 0 1");
            MoveList moveList; pos.GenerateMoveList(moveList);
            TEST_EXPECT(moveList.Size() == 8u);
        }

        // castling
        {
            Position pos("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1");
            MoveList moveList; pos.GenerateMoveList(moveList);
            TEST_EXPECT(moveList.Size() == 25u);
        }

        // castling
        {
            Position pos("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RN2K2R w KQkq - 0 1");
            MoveList moveList; pos.GenerateMoveList(moveList);
            TEST_EXPECT(moveList.Size() == 23u);
        }

        // castling
        {
            Position pos("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w Kkq - 0 1");
            MoveList moveList; pos.GenerateMoveList(moveList);
            TEST_EXPECT(moveList.Size() == 24u);
        }

        // castling
        {
            Position pos("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w Qkq - 0 1");
            MoveList moveList; pos.GenerateMoveList(moveList);
            TEST_EXPECT(moveList.Size() == 24u);
        }

        // castling
        {
            Position pos("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w kq - 0 1");
            MoveList moveList; pos.GenerateMoveList(moveList);
            TEST_EXPECT(moveList.Size() == 23u);
        }
    }

    // white pawn moves
    {
        const uint32_t kingMoves = 3u;

        // 2rd rank
        {
            Position pos("k7/8/8/8/8/8/4P3/K7 w - - 0 1");
            MoveList moveList; pos.GenerateMoveList(moveList);
            TEST_EXPECT(moveList.Size() - kingMoves == 2u);
        }

        // 3rd rank
        {
            Position pos("k7/8/8/8/8/4P3/8/K7 w - - 0 1");
            MoveList moveList; pos.GenerateMoveList(moveList);
            TEST_EXPECT(moveList.Size() - kingMoves == 1u);
        }

        // 2rd rank blocked
        {
            Position pos("k7/8/8/8/8/4p3/4P3/K7 w - - 0 1");
            MoveList moveList; pos.GenerateMoveList(moveList);
            TEST_EXPECT(moveList.Size() - kingMoves == 0u);
        }

        // 3rd rank blocked
        {
            Position pos("k7/8/8/8/4p3/4P3/8/K7 w - - 0 1");
            MoveList moveList; pos.GenerateMoveList(moveList);
            TEST_EXPECT(moveList.Size() - kingMoves == 0u);
        }

        // simple capture
        {
            Position pos("k7/8/8/3p4/4P3/8/8/K7 w - - 0 1");
            MoveList moveList; pos.GenerateMoveList(moveList);
            TEST_EXPECT(moveList.Size() - kingMoves == 2u);
        }

        // two captures
        {
            Position pos("k7/8/8/3p1p2/4P3/8/8/K7 w - - 0 1");
            MoveList moveList; pos.GenerateMoveList(moveList);
            TEST_EXPECT(moveList.Size() - kingMoves == 3u);
        }

        // two captures and block
        {
            Position pos("k7/8/8/3ppp2/4P3/8/8/K7 w - - 0 1");
            MoveList moveList; pos.GenerateMoveList(moveList);
            TEST_EXPECT(moveList.Size() - kingMoves == 2u);
        }

        // promotion
        {
            Position pos("k7/4P3/8/8/8/8/8/K7 w - - 0 1");
            MoveList moveList; pos.GenerateMoveList(moveList);
            TEST_EXPECT(moveList.Size() - kingMoves == 4u);
        }

        // blocked promotion
        {
            Position pos("k3n3/4P3/8/8/8/8/8/K7 w - - 0 1");
            MoveList moveList; pos.GenerateMoveList(moveList);
            TEST_EXPECT(moveList.Size() - kingMoves == 0u);
        }

        // 3 promotions possible
        {
            Position pos("k3n1n1/5P2/8/8/8/8/8/K7 w - - 0 1");
            MoveList moveList; pos.GenerateMoveList(moveList);
            TEST_EXPECT(moveList.Size() - kingMoves == 12u);
        }
    }

    // black pawn moves
    {
        const uint32_t kingMoves = 3u;

        // simple capture
        {
            Position pos("k7/8/8/2Rp4/2P5/8/8/K7 b - - 0 1");
            MoveList moveList; pos.GenerateMoveList(moveList);
            TEST_EXPECT(moveList.Size() - kingMoves == 2u);
        }

        // promotion
        {
            Position pos("k7/8/8/8/8/8/4p3/K7 b - - 0 1");
            MoveList moveList; pos.GenerateMoveList(moveList);
            TEST_EXPECT(moveList.Size() - kingMoves == 4u);
        }
    }

    // moves from starting position
    {
        Position pos(Position::InitPositionFEN);
        MoveList moveList; pos.GenerateMoveList(moveList);
        TEST_EXPECT(moveList.Size() == 20u);
    }

    // moves parsing & execution
    {
        // move (invalid)
        {
            Position pos(Position::InitPositionFEN);
            const Move move = pos.MoveFromString("e3e4");
            TEST_EXPECT(move.IsValid());
            TEST_EXPECT(!pos.IsMoveValid(move));
        }

        // move pawn (invalid)
        {
            Position pos(Position::InitPositionFEN);
            const Move move = pos.MoveFromString("e2e2");
            TEST_EXPECT(move.IsValid());
            TEST_EXPECT(!pos.IsMoveValid(move));
        }

        // move pawn (invalid)
        {
            Position pos(Position::InitPositionFEN);
            const Move move = pos.MoveFromString("e2f3");
            TEST_EXPECT(move.IsValid());
            TEST_EXPECT(!pos.IsMoveValid(move));
        }

        // move pawn (valid)
        {
            Position pos(Position::InitPositionFEN);
            const Move move = pos.MoveFromString("e2e4");
            TEST_EXPECT(move.IsValid());
            TEST_EXPECT(move.FromSquare() == Square_e2);
            TEST_EXPECT(move.ToSquare() == Square_e4);
            TEST_EXPECT(move.GetPiece() == Piece::Pawn);
            TEST_EXPECT(move.IsCapture() == false);
            TEST_EXPECT(move.GetPromoteTo() == Piece::None);
            TEST_EXPECT(pos.IsMoveValid(move));
            TEST_EXPECT(pos.IsMoveLegal(move));
            TEST_EXPECT(pos.DoMove(move));
            TEST_EXPECT(pos.ToFEN() == "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1");
        }

        // move pawn (invalid, blocked)
        {
            Position pos("rnbqkbnr/pppp1ppp/8/8/8/4p3/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
            const Move move = pos.MoveFromString("e2e4");
            TEST_EXPECT(move.IsValid());
            TEST_EXPECT(move.FromSquare() == Square_e2);
            TEST_EXPECT(move.ToSquare() == Square_e4);
            TEST_EXPECT(move.GetPiece() == Piece::Pawn);
            TEST_EXPECT(move.IsCapture() == false);
            TEST_EXPECT(move.GetPromoteTo() == Piece::None);
            TEST_EXPECT(!pos.IsMoveValid(move));
        }

        // move pawn (invalid, blocked)
        {
            Position pos("rnbqkbnr/pppp1ppp/8/8/4p3/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
            const Move move = pos.MoveFromString("e2e4");
            TEST_EXPECT(move.IsValid());
            TEST_EXPECT(move.FromSquare() == Square_e2);
            TEST_EXPECT(move.ToSquare() == Square_e4);
            TEST_EXPECT(move.GetPiece() == Piece::Pawn);
            TEST_EXPECT(move.GetPromoteTo() == Piece::None);
            TEST_EXPECT(!pos.IsMoveValid(move));
        }

        // move pawn (invalid, blocked)
        {
            Position pos("rnbqkbnr/1ppppppp/p7/5B2/8/3P4/PPP1PPPP/RN1QKBNR b KQkq - 0 1");
            const Move move = pos.MoveFromString("f7f5");
            TEST_EXPECT(move.IsValid());
            TEST_EXPECT(move.FromSquare() == Square_f7);
            TEST_EXPECT(move.ToSquare() == Square_f5);
            TEST_EXPECT(move.GetPiece() == Piece::Pawn);
            TEST_EXPECT(move.GetPromoteTo() == Piece::None);
            TEST_EXPECT(!pos.IsMoveValid(move));
        }

        // pawn capture
        {
            Position pos("rnbqkbnr/p1pppppp/8/1p6/2P5/8/PP1PPPPP/RNBQKBNR w KQkq - 0 1");
            const Move move = pos.MoveFromString("c4b5");
            TEST_EXPECT(move.IsValid());
            TEST_EXPECT(move.FromSquare() == Square_c4);
            TEST_EXPECT(move.ToSquare() == Square_b5);
            TEST_EXPECT(move.GetPiece() == Piece::Pawn);
            TEST_EXPECT(move.IsCapture() == true);
            TEST_EXPECT(move.IsEnPassant() == false);
            TEST_EXPECT(move.GetPromoteTo() == Piece::None);
            TEST_EXPECT(pos.IsMoveValid(move));
            TEST_EXPECT(pos.IsMoveLegal(move));
            TEST_EXPECT(pos.DoMove(move));
            TEST_EXPECT(pos.ToFEN() == "rnbqkbnr/p1pppppp/8/1P6/8/8/PP1PPPPP/RNBQKBNR b KQkq - 0 1");
        }

        // en passant capture
        {
            Position pos("rnbqkbnr/pp1ppppp/8/2pP4/8/8/PPP1PPPP/RNBQKBNR w KQkq c6 0 1");
            const Move move = pos.MoveFromString("d5c6");
            TEST_EXPECT(move.IsValid());
            TEST_EXPECT(move.FromSquare() == Square_d5);
            TEST_EXPECT(move.ToSquare() == Square_c6);
            TEST_EXPECT(move.GetPiece() == Piece::Pawn);
            TEST_EXPECT(move.IsCapture() == true);
            TEST_EXPECT(move.IsEnPassant() == true);
            TEST_EXPECT(move.GetPromoteTo() == Piece::None);
            TEST_EXPECT(pos.IsMoveValid(move));
            TEST_EXPECT(pos.IsMoveLegal(move));
            TEST_EXPECT(pos.DoMove(move));
            TEST_EXPECT(pos.ToFEN() == "rnbqkbnr/pp1ppppp/2P5/8/8/8/PPP1PPPP/RNBQKBNR b KQkq - 0 1");
        }

        // can't en passant own pawn
        {
            Position pos("rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR w KQkq d3 0 1");
            const Move move = pos.MoveFromString("e2d3");
            TEST_EXPECT(move.IsValid());
            TEST_EXPECT(move.FromSquare() == Square_e2);
            TEST_EXPECT(move.ToSquare() == Square_d3);
            TEST_EXPECT(move.GetPiece() == Piece::Pawn);
            TEST_EXPECT(move.IsCapture() == true);
            TEST_EXPECT(move.IsEnPassant() == true);
            TEST_EXPECT(move.GetPromoteTo() == Piece::None);
            TEST_EXPECT(!pos.IsMoveValid(move));
        }

        // move pawn (invalid promotion)
        {
            Position pos("1k6/5P2/8/8/8/8/8/4K3 w - - 0 1");
            const Move move = pos.MoveFromString("f7f8k");
            TEST_EXPECT(move.IsValid());
            TEST_EXPECT(move.FromSquare() == Square_f7);
            TEST_EXPECT(move.ToSquare() == Square_f8);
            TEST_EXPECT(move.GetPiece() == Piece::Pawn);
            TEST_EXPECT(move.IsCapture() == false);
            TEST_EXPECT(move.GetPromoteTo() == Piece::King);
            TEST_EXPECT(!pos.IsMoveValid(move));
        }

        // move pawn (valid promotion)
        {
            Position pos("1k6/5P2/8/8/8/8/8/4K3 w - - 0 1");
            const Move move = pos.MoveFromString("f7f8q");
            TEST_EXPECT(move.IsValid());
            TEST_EXPECT(move.FromSquare() == Square_f7);
            TEST_EXPECT(move.ToSquare() == Square_f8);
            TEST_EXPECT(move.GetPiece() == Piece::Pawn);
            TEST_EXPECT(move.IsCapture() == false);
            TEST_EXPECT(move.GetPromoteTo() == Piece::Queen);
            TEST_EXPECT(pos.IsMoveValid(move));
            TEST_EXPECT(pos.IsMoveLegal(move));
            TEST_EXPECT(pos.DoMove(move));
            TEST_EXPECT(pos.ToFEN() == "1k3Q2/8/8/8/8/8/8/4K3 b - - 0 1");
        }

        // move knight (valid)
        {
            Position pos("4k3/8/8/8/8/3N4/8/4K3 w - - 0 1");
            const Move move = pos.MoveFromString("d3f4");
            TEST_EXPECT(move.IsValid());
            TEST_EXPECT(move.FromSquare() == Square_d3);
            TEST_EXPECT(move.ToSquare() == Square_f4);
            TEST_EXPECT(move.GetPiece() == Piece::Knight);
            TEST_EXPECT(move.IsCapture() == false);
            TEST_EXPECT(pos.IsMoveValid(move));
            TEST_EXPECT(pos.IsMoveLegal(move));
            TEST_EXPECT(pos.DoMove(move));
            TEST_EXPECT(pos.ToFEN() == "4k3/8/8/8/5N2/8/8/4K3 b - - 1 1");
        }

        // move knight (valid capture)
        {
            Position pos("4k3/8/8/8/5q2/3N4/8/4K3 w - - 0 1");
            const Move move = pos.MoveFromString("d3f4");
            TEST_EXPECT(move.IsValid());
            TEST_EXPECT(move.FromSquare() == Square_d3);
            TEST_EXPECT(move.ToSquare() == Square_f4);
            TEST_EXPECT(move.GetPiece() == Piece::Knight);
            TEST_EXPECT(move.IsCapture() == true);
            TEST_EXPECT(pos.IsMoveValid(move));
            TEST_EXPECT(pos.IsMoveLegal(move));
            TEST_EXPECT(pos.DoMove(move));
            TEST_EXPECT(pos.ToFEN() == "4k3/8/8/8/5N2/8/8/4K3 b - - 0 1");
        }

        // castling, whites, king side
        {
            Position pos("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQK2R w KQkq - 0 1");
            const Move move = pos.MoveFromString("e1g1");
            TEST_EXPECT(move.IsValid());
            TEST_EXPECT(move.FromSquare() == Square_e1);
            TEST_EXPECT(move.ToSquare() == Square_g1);
            TEST_EXPECT(move.GetPiece() == Piece::King);
            TEST_EXPECT(move.IsCapture() == false);
            TEST_EXPECT(move.IsCastling() == true);
            TEST_EXPECT(pos.IsMoveValid(move));
            TEST_EXPECT(pos.IsMoveLegal(move));
            TEST_EXPECT(pos.DoMove(move));
            TEST_EXPECT(pos.ToFEN() == "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQ1RK1 b kq - 1 1");
        }

        // castling, whites, king side, no rights
        {
            Position pos("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQK2R w Qkq - 0 1");
            const Move move = pos.MoveFromString("e1g1");
            TEST_EXPECT(move.IsValid());
            TEST_EXPECT(move.FromSquare() == Square_e1);
            TEST_EXPECT(move.ToSquare() == Square_g1);
            TEST_EXPECT(move.GetPiece() == Piece::King);
            TEST_EXPECT(move.IsCapture() == false);
            TEST_EXPECT(move.IsCastling() == true);
            TEST_EXPECT(!pos.IsMoveValid(move));
        }

        // castling, whites, queen side
        {
            Position pos("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/R3KBNR w KQkq - 0 1");
            const Move move = pos.MoveFromString("e1c1");
            TEST_EXPECT(move.IsValid());
            TEST_EXPECT(move.FromSquare() == Square_e1);
            TEST_EXPECT(move.ToSquare() == Square_c1);
            TEST_EXPECT(move.GetPiece() == Piece::King);
            TEST_EXPECT(move.IsCapture() == false);
            TEST_EXPECT(move.IsCastling() == true);
            TEST_EXPECT(pos.IsMoveValid(move));
            TEST_EXPECT(pos.IsMoveLegal(move));
            TEST_EXPECT(pos.DoMove(move));
            TEST_EXPECT(pos.ToFEN() == "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/2KR1BNR b kq - 1 1");
        }

        // castling, whites, queen side, no rights
        {
            Position pos("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/R3KBNR w Kkq - 0 1");
            const Move move = pos.MoveFromString("e1c1");
            TEST_EXPECT(move.IsValid());
            TEST_EXPECT(move.FromSquare() == Square_e1);
            TEST_EXPECT(move.ToSquare() == Square_c1);
            TEST_EXPECT(move.GetPiece() == Piece::King);
            TEST_EXPECT(move.IsCapture() == false);
            TEST_EXPECT(move.IsCastling() == true);
            TEST_EXPECT(!pos.IsMoveValid(move));
        }

        // castling, blacks, king side
        {
            Position pos("rnbqk2r/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1");
            const Move move = pos.MoveFromString("e8g8");
            TEST_EXPECT(move.IsValid());
            TEST_EXPECT(move.FromSquare() == Square_e8);
            TEST_EXPECT(move.ToSquare() == Square_g8);
            TEST_EXPECT(move.GetPiece() == Piece::King);
            TEST_EXPECT(move.IsCapture() == false);
            TEST_EXPECT(move.IsCastling() == true);
            TEST_EXPECT(pos.IsMoveValid(move));
            TEST_EXPECT(pos.IsMoveLegal(move));
            TEST_EXPECT(pos.DoMove(move));
            TEST_EXPECT(pos.ToFEN() == "rnbq1rk1/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQ - 1 2");
        }

        // castling, blacks, king side, no rights
        {
            Position pos("rnbqk2r/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQq - 0 1");
            const Move move = pos.MoveFromString("e8g8");
            TEST_EXPECT(move.IsValid());
            TEST_EXPECT(move.FromSquare() == Square_e8);
            TEST_EXPECT(move.ToSquare() == Square_g8);
            TEST_EXPECT(move.GetPiece() == Piece::King);
            TEST_EXPECT(move.IsCapture() == false);
            TEST_EXPECT(move.IsCastling() == true);
            TEST_EXPECT(!pos.IsMoveValid(move));
        }

        // castling, blacks, queen side
        {
            Position pos("r3kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1");
            const Move move = pos.MoveFromString("e8c8");
            TEST_EXPECT(move.IsValid());
            TEST_EXPECT(move.FromSquare() == Square_e8);
            TEST_EXPECT(move.ToSquare() == Square_c8);
            TEST_EXPECT(move.GetPiece() == Piece::King);
            TEST_EXPECT(move.IsCapture() == false);
            TEST_EXPECT(move.IsCastling() == true);
            TEST_EXPECT(pos.IsMoveValid(move));
            TEST_EXPECT(pos.IsMoveLegal(move));
            TEST_EXPECT(pos.DoMove(move));
            TEST_EXPECT(pos.ToFEN() == "2kr1bnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQ - 1 2");
        }

        // castling, blacks, queen side, no rights
        {
            Position pos("r3kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQk - 0 1");
            const Move move = pos.MoveFromString("e8c8");
            TEST_EXPECT(move.IsValid());
            TEST_EXPECT(move.FromSquare() == Square_e8);
            TEST_EXPECT(move.ToSquare() == Square_c8);
            TEST_EXPECT(move.GetPiece() == Piece::King);
            TEST_EXPECT(move.IsCapture() == false);
            TEST_EXPECT(move.IsCastling() == true);
            TEST_EXPECT(!pos.IsMoveValid(move));
        }

        // illegal castling, whites, king side, king in check
        {
            Position pos("4k3/4r3/8/8/8/8/8/R3K2R w KQ - 0 1");
            const Move move = pos.MoveFromString("e1g1");
            TEST_EXPECT(move.IsValid());
            TEST_EXPECT(move.FromSquare() == Square_e1);
            TEST_EXPECT(move.ToSquare() == Square_g1);
            TEST_EXPECT(move.GetPiece() == Piece::King);
            TEST_EXPECT(move.IsCapture() == false);
            TEST_EXPECT(move.IsCastling() == true);
            TEST_EXPECT(!pos.IsMoveValid(move));
        }

        // illegal castling, whites, king side, king crossing check
        {
            Position pos("4kr2/8/8/8/8/8/8/R3K2R w KQ - 0 1");
            const Move move = pos.MoveFromString("e1g1");
            TEST_EXPECT(move.IsValid());
            TEST_EXPECT(move.FromSquare() == Square_e1);
            TEST_EXPECT(move.ToSquare() == Square_g1);
            TEST_EXPECT(move.GetPiece() == Piece::King);
            TEST_EXPECT(move.IsCapture() == false);
            TEST_EXPECT(move.IsCastling() == true);
            TEST_EXPECT(!pos.IsMoveValid(move));
        }

        // move rook, loose castling rights
        {
            Position pos("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1");
            const Move move = pos.MoveFromString("a1b1");
            TEST_EXPECT(move.IsValid());
            TEST_EXPECT(move.FromSquare() == Square_a1);
            TEST_EXPECT(move.ToSquare() == Square_b1);
            TEST_EXPECT(move.GetPiece() == Piece::Rook);
            TEST_EXPECT(move.IsCapture() == false);
            TEST_EXPECT(move.IsCastling() == false);
            TEST_EXPECT(pos.IsMoveValid(move));
            TEST_EXPECT(pos.IsMoveLegal(move));
            TEST_EXPECT(pos.DoMove(move));
            TEST_EXPECT(pos.ToFEN() == "r3k2r/8/8/8/8/8/8/1R2K2R b Kkq - 1 1");
        }

        // move rook, loose castling rights
        {
            Position pos("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1");
            const Move move = pos.MoveFromString("h1g1");
            TEST_EXPECT(move.IsValid());
            TEST_EXPECT(move.FromSquare() == Square_h1);
            TEST_EXPECT(move.ToSquare() == Square_g1);
            TEST_EXPECT(move.GetPiece() == Piece::Rook);
            TEST_EXPECT(move.IsCapture() == false);
            TEST_EXPECT(move.IsCastling() == false);
            TEST_EXPECT(pos.IsMoveValid(move));
            TEST_EXPECT(pos.IsMoveLegal(move));
            TEST_EXPECT(pos.DoMove(move));
            TEST_EXPECT(pos.ToFEN() == "r3k2r/8/8/8/8/8/8/R3K1R1 b Qkq - 1 1");
        }

        // move rook, loose castling rights
        {
            Position pos("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1");
            const Move move = pos.MoveFromString("a8b8");
            TEST_EXPECT(move.IsValid());
            TEST_EXPECT(move.FromSquare() == Square_a8);
            TEST_EXPECT(move.ToSquare() == Square_b8);
            TEST_EXPECT(move.GetPiece() == Piece::Rook);
            TEST_EXPECT(move.IsCapture() == false);
            TEST_EXPECT(move.IsCastling() == false);
            TEST_EXPECT(pos.IsMoveValid(move));
            TEST_EXPECT(pos.IsMoveLegal(move));
            TEST_EXPECT(pos.DoMove(move));
            TEST_EXPECT(pos.ToFEN() == "1r2k2r/8/8/8/8/8/8/R3K2R w KQk - 1 2");
        }

        // move rook, loose castling rights
        {
            Position pos("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1");
            const Move move = pos.MoveFromString("h8g8");
            TEST_EXPECT(move.IsValid());
            TEST_EXPECT(move.FromSquare() == Square_h8);
            TEST_EXPECT(move.ToSquare() == Square_g8);
            TEST_EXPECT(move.GetPiece() == Piece::Rook);
            TEST_EXPECT(move.IsCapture() == false);
            TEST_EXPECT(move.IsCastling() == false);
            TEST_EXPECT(pos.IsMoveValid(move));
            TEST_EXPECT(pos.IsMoveLegal(move));
            TEST_EXPECT(pos.DoMove(move));
            TEST_EXPECT(pos.ToFEN() == "r3k1r1/8/8/8/8/8/8/R3K2R w KQq - 1 2");
        }

        // move king to close opponent's king (illegal move)
        {
            Position pos("7K/8/5k2/8/8/8/8/8 w - - 0 1");
            const Move move = pos.MoveFromString("h8g7");
            TEST_EXPECT(move.IsValid());
            TEST_EXPECT(move.FromSquare() == Square_h8);
            TEST_EXPECT(move.ToSquare() == Square_g7);
            TEST_EXPECT(move.GetPiece() == Piece::King);
            TEST_EXPECT(move.IsCapture() == false);
            TEST_EXPECT(move.IsCastling() == false);
            TEST_EXPECT(!pos.IsMoveValid(move));
        }

        // pin
        {
            Position pos("k7/8/q7/8/R7/8/8/K7 w - - 0 1");
            const Move move = pos.MoveFromString("a4b4");
            TEST_EXPECT(move.IsValid());
            TEST_EXPECT(move.FromSquare() == Square_a4);
            TEST_EXPECT(move.ToSquare() == Square_b4);
            TEST_EXPECT(move.GetPiece() == Piece::Rook);
            TEST_EXPECT(move.IsCapture() == false);
            TEST_EXPECT(move.IsCastling() == false);
            TEST_EXPECT(pos.IsMoveValid(move));
            TEST_EXPECT(!pos.IsMoveLegal(move));
        }
    }

    // Static Exchange Evaluation
    {
        // quiet move
        {
            Position pos("7k/8/1p6/8/8/1Q6/8/7K w - - 0 1");
            const Move move = pos.MoveFromString("b3b4");
            TEST_EXPECT(move.IsValid());
            TEST_EXPECT(true == pos.StaticExchangeEvaluation(move, 0));
        }

        // hanging pawn
        {
            Position pos("7k/8/1p6/8/8/1Q6/8/7K w - - 0 1");
            const Move move = pos.MoveFromString("b3b6");
            TEST_EXPECT(move.IsValid());
            TEST_EXPECT(true == pos.StaticExchangeEvaluation(move, 0));
        }

        // queen takes pawn protected by another pawn
        {
            Position pos("7k/p7/1p6/8/8/1Q6/8/7K w - - 0 1");
            const Move move = pos.MoveFromString("b3b6");
            TEST_EXPECT(move.IsValid());
            TEST_EXPECT(true == pos.StaticExchangeEvaluation(move, -801));
            TEST_EXPECT(true == pos.StaticExchangeEvaluation(move, -800));
            TEST_EXPECT(false == pos.StaticExchangeEvaluation(move, -799));
            TEST_EXPECT(false == pos.StaticExchangeEvaluation(move, 0));
        }

        // queen trade
        {
            Position pos("7k/p7/1q6/8/8/1Q6/8/7K w - - 0 1");
            const Move move = pos.MoveFromString("b3b6");
            TEST_EXPECT(move.IsValid());
            TEST_EXPECT(true == pos.StaticExchangeEvaluation(move, -1));
            TEST_EXPECT(true == pos.StaticExchangeEvaluation(move, 0));
            TEST_EXPECT(false == pos.StaticExchangeEvaluation(move, 1));
        }

        // rook trade
        {
            Position pos("7k/p7/1q6/8/8/1Q6/8/7K w - - 0 1");
            const Move move = pos.MoveFromString("b3b6");
            TEST_EXPECT(move.IsValid());
            TEST_EXPECT(true == pos.StaticExchangeEvaluation(move, -1));
            TEST_EXPECT(true == pos.StaticExchangeEvaluation(move, 0));
            TEST_EXPECT(false == pos.StaticExchangeEvaluation(move, 1));
        }

        // (rook+bishop) vs. 2 knights -> bishop
        {
            Position pos("7k/3n4/1n6/8/8/1R2B3/8/7K w - - 0 1");
            const Move move = pos.MoveFromString("b3b6");
            TEST_EXPECT(move.IsValid());
            TEST_EXPECT(true == pos.StaticExchangeEvaluation(move, 0));
            TEST_EXPECT(true == pos.StaticExchangeEvaluation(move, 100));
            TEST_EXPECT(false == pos.StaticExchangeEvaluation(move, 200));
        }

        // 4 rooks and 4 bishops
        {
            Position pos("kB2r2b/8/8/1r2p2R/8/8/1B5b/K3R3 w - - 0 1");
            const Move move = pos.MoveFromString("b2e5");
            TEST_EXPECT(move.IsValid());
            TEST_EXPECT(true == pos.StaticExchangeEvaluation(move, -200));
            TEST_EXPECT(false == pos.StaticExchangeEvaluation(move, -199));
        }

        // 2 rooks battery
        {
            Position pos("K2R4/3R4/8/8/8/3r2r1/8/7k w - - 0 1");
            const Move move = pos.MoveFromString("d7d3");
            TEST_EXPECT(move.IsValid());
            TEST_EXPECT(true == pos.StaticExchangeEvaluation(move, 500));
        }

        // pawns and bishops on diagonal
        {
            Position pos("7k/b7/8/2p5/3P4/4B3/8/7K w - - 0 1");
            const Move move = pos.MoveFromString("d4c5");
            TEST_EXPECT(move.IsValid());
            TEST_EXPECT(true == pos.StaticExchangeEvaluation(move, 100));
            TEST_EXPECT(false == pos.StaticExchangeEvaluation(move, 101));
        }

        // queen takes rook, then king take the queen
        {
            Position pos("3rk2r/2Q2p2/p3q2p/1p1p2p1/1B1P1n2/2P2P2/P3bRPP/4R1K1 w - - 0 25");
            const Move move = pos.MoveFromString("c7d8");
            TEST_EXPECT(move.IsValid());
            TEST_EXPECT(true == pos.StaticExchangeEvaluation(move, -400));
            TEST_EXPECT(false == pos.StaticExchangeEvaluation(move, -399));
        }

        // same as above, but king can't capture the queen because it's protected by a bishop
        {
            Position pos("3rk2r/2Q2p2/p3q2p/Bp1p2p1/3P1n2/2P2P2/P3bRPP/4R1K1 w - - 0 25");
            const Move move = pos.MoveFromString("c7d8");
            TEST_EXPECT(move.IsValid());
            TEST_EXPECT(true == pos.StaticExchangeEvaluation(move, 500));
            TEST_EXPECT(false == pos.StaticExchangeEvaluation(move, 501));
        }
    }

    // IsStaleMate
    {
        {
            const Position pos("7K/5k2/8/8/8/8/8/8 w - - 0 1");
            TEST_EXPECT(!pos.IsInCheck(Color::White));
            TEST_EXPECT(!pos.IsStalemate());
        }

        {
            const Position pos("7K/5k1P/8/8/8/8/8/8 w - - 0 1");
            TEST_EXPECT(!pos.IsInCheck(Color::White));
            TEST_EXPECT(pos.IsStalemate());
        }

        {
            const Position pos("7k/8/7r/K2P3q/P7/8/8/1r6 w - - 0 1");
            TEST_EXPECT(!pos.IsInCheck(Color::White));
            TEST_EXPECT(pos.IsStalemate());
        }
    }
}

static void RunPerftTests()
{
    std::cout << "Running perft tests..." << std::endl;

    {
        const Position pos("rnbqkbnr/1ppppppp/p7/5B2/8/3P4/PPP1PPPP/RN1QKBNR b KQkq - 0 1");
        TEST_EXPECT(pos.Perft(1) == 18u);
    }

    {
        const Position pos("rnbqkbnr/1ppppppp/p7/8/8/3P4/PPP1PPPP/RNBQKBNR w KQkq - 0 1");
        TEST_EXPECT(pos.Perft(2) == 511u);
    }

    {
        const Position pos("rnbqkbnr/pppppppp/8/8/8/3P4/PPP1PPPP/RNBQKBNR b KQkq - 0 1");
        TEST_EXPECT(pos.Perft(3) == 11959u);
    }

    {
        const Position pos("rnb1kbnr/pp1ppppp/1qp5/1P6/8/8/P1PPPPPP/RNBQKBNR w KQkq - 0 1");
        TEST_EXPECT(pos.Perft(1) == 21u);
    }

    {
        const Position pos("rnbqkbnr/pp1ppppp/2p5/1P6/8/8/P1PPPPPP/RNBQKBNR b KQkq - 0 1");
        TEST_EXPECT(pos.Perft(2) == 458u);
    }

    {
        const Position pos("rnbqkbnr/pp1ppppp/2p5/8/1P6/8/P1PPPPPP/RNBQKBNR w KQkq - 0 1");
        TEST_EXPECT(pos.Perft(3) == 10257u);
    }

    {
        const Position pos("rnbqkbnr/pppppppp/8/8/1P6/8/P1PPPPPP/RNBQKBNR b KQkq - 0 1");
        TEST_EXPECT(pos.Perft(4) == 216145u);
    }

    // Perft
    {
        // initial position
        {
            const Position pos(Position::InitPositionFEN);
            TEST_EXPECT(pos.Perft(1) == 20u);
            TEST_EXPECT(pos.Perft(2) == 400u);
            TEST_EXPECT(pos.Perft(3) == 8902u);
            TEST_EXPECT(pos.Perft(4) == 197281u);
            //TEST_EXPECT(pos.Perft(5) == 4865609u);
            //TEST_EXPECT(pos.Perft(6) == 119060324u);
        }

        // kings only
        {
            const Position pos("2k2K2/8/8/8/8/8/8/8 w - - 0 1");
            TEST_EXPECT(pos.Perft(4) == 848u);
            TEST_EXPECT(pos.Perft(6) == 29724u);
        }

        // kings + knight vs. king
        {
            const Position pos("2k2K2/5N2/8/8/8/8/8/8 w - - 0 1");
            TEST_EXPECT(pos.Perft(2) == 41u);
            TEST_EXPECT(pos.Perft(4) == 2293u);
            TEST_EXPECT(pos.Perft(6) == 130360u);
        }

        // kings + rook vs. king
        {
            const Position pos("2k2K2/5R2/8/8/8/8/8/8 w - - 0 1");
            TEST_EXPECT(pos.Perft(1) == 17u);
            TEST_EXPECT(pos.Perft(2) == 53u);
            TEST_EXPECT(pos.Perft(4) == 3917u);
            TEST_EXPECT(pos.Perft(6) == 338276u);
        }

        // kings + bishop vs. king
        {
            const Position pos("2k2K2/5B2/8/8/8/8/8/8 w - - 0 1");
            TEST_EXPECT(pos.Perft(2) == 58u);
            TEST_EXPECT(pos.Perft(4) == 4269u);
            TEST_EXPECT(pos.Perft(6) == 314405u);
        }

        // kings + pawn vs. king
        {
            const Position pos("2k3K1/4P3/8/8/8/8/8/8 w - - 0 1");
            TEST_EXPECT(pos.Perft(2) == 33u);
            TEST_EXPECT(pos.Perft(4) == 2007u);
            TEST_EXPECT(pos.Perft(6) == 136531u);
        }

        // castlings
        {
            const Position pos("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1");
            TEST_EXPECT(pos.Perft(1) == 26u);
            TEST_EXPECT(pos.Perft(2) == 568u);
            //TEST_EXPECT(pos.Perft(4) == 314346u);
        }

        // kings + 2 queens
        {
            const Position pos("q3k2q/8/8/8/8/8/8/Q3K2Q w - - 0 1");
            TEST_EXPECT(pos.Perft(2) == 1040u);
            TEST_EXPECT(pos.Perft(4) == 979543u);
            //TEST_EXPECT(pos.Perft(6) == 923005707u);
        }

        // max moves
        {
            const Position pos("R6R/3Q4/1Q4Q1/4Q3/2Q4Q/Q4Q2/pp1Q4/kBNN1KB1 w - - 0 1");
            TEST_EXPECT(pos.Perft(1) == 218u);
        }

        // discovered double check via en passant
        {
            const Position pos("8/6p1/7k/7P/5B1R/8/8/7K b - - 0 1");
            TEST_EXPECT(pos.Perft(1) == 2u);
            TEST_EXPECT(pos.Perft(2) == 35u);
            TEST_EXPECT(pos.Perft(3) == 134u);
        }

        // Kiwipete
        {
            const Position pos("r3k2r/p1ppqpb1/1n2pnp1/3PN3/1p2P3/2N2Q1p/PPPB1PPP/R2BKb1R w KQkq - 0 1");
            TEST_EXPECT(pos.Perft(1) == 40u);
        }

        // Kiwipete
        {
            const Position pos("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPB1PPP/R2BK2R b KQkq - 0 1");
            TEST_EXPECT(pos.Perft(1) == 44u);
            TEST_EXPECT(pos.Perft(2) == 1733u);
        }

        // Position 2 - Kiwipete
        {
            const Position pos("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1");
            TEST_EXPECT(pos.Perft(1) == 48u);
            TEST_EXPECT(pos.Perft(2) == 2039u);
            TEST_EXPECT(pos.Perft(3) == 97862u);
            TEST_EXPECT(pos.Perft(4) == 4085603u);
        }

        // Position 3
        {
            const Position pos("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1");
            TEST_EXPECT(pos.Perft(1) == 14u);
            TEST_EXPECT(pos.Perft(2) == 191u);
            TEST_EXPECT(pos.Perft(3) == 2812u);
            TEST_EXPECT(pos.Perft(4) == 43238u);
            //TEST_EXPECT(pos.Perft(5) == 674624u);
        }

        // Position 4
        {
            const Position pos("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1");
            TEST_EXPECT(pos.Perft(1) == 6u);
            TEST_EXPECT(pos.Perft(2) == 264u);
            TEST_EXPECT(pos.Perft(3) == 9467u);
            TEST_EXPECT(pos.Perft(4) == 422333u);
            //TEST_EXPECT(pos.Perft(5) == 15833292u);
        }

        // Position 5
        {
            const Position pos("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8");
            TEST_EXPECT(pos.Perft(1) == 44u);
            TEST_EXPECT(pos.Perft(2) == 1486u);
            TEST_EXPECT(pos.Perft(3) == 62379u);
            TEST_EXPECT(pos.Perft(4) == 2103487u);
            //TEST_EXPECT(pos.Perft(5) == 89941194u);
        }

        // Position 6
        {
            const Position pos("r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10");
            TEST_EXPECT(pos.Perft(1) == 46u);
            TEST_EXPECT(pos.Perft(2) == 2079u);
            TEST_EXPECT(pos.Perft(3) == 89890u);
            TEST_EXPECT(pos.Perft(4) == 3894594u);
            //TEST_EXPECT(pos.Perft(5) == 164075551u);
            //TEST_EXPECT(pos.Perft(6) == 6923051137llu);
            //TEST_EXPECT(pos.Perft(7) == 287188994746llu);
        }
    }
}

static void RunEvalTests()
{
    // incufficient material
    {
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

        // KvBB (same color)
        TEST_EXPECT(0 == Evaluate(Position("KB6/B7/8/8/8/8/8/7k w - - 0 1")));
        TEST_EXPECT(0 == Evaluate(Position("KB6/B7/8/8/8/8/8/7k b - - 0 1")));
        TEST_EXPECT(0 == Evaluate(Position("K7/8/8/8/8/8/7b/6bk w - - 0 1")));
        TEST_EXPECT(0 == Evaluate(Position("K7/8/8/8/8/8/7b/6bk b - - 0 1")));

        // KvBB (opposite colors)
        TEST_EXPECT(KnownWinValue <= Evaluate(Position("K7/B7/B7/8/8/8/8/7k w - - 0 1")));
        TEST_EXPECT(KnownWinValue <= Evaluate(Position("K7/B7/B7/8/8/8/8/7k b - - 0 1")));
        TEST_EXPECT(-KnownWinValue >= Evaluate(Position("K7/8/8/8/8/7b/7b/7k w - - 0 1")));
        TEST_EXPECT(-KnownWinValue >= Evaluate(Position("K7/8/8/8/8/7b/7b/7k w - - 0 1")));

        // KvR
        TEST_EXPECT(KnownWinValue <= Evaluate(Position("K7/R7/8/8/8/8/8/7k w - - 0 1")));
        TEST_EXPECT(KnownWinValue <= Evaluate(Position("K7/R7/8/8/8/8/8/7k w - - 0 1")));
        TEST_EXPECT(-KnownWinValue >= Evaluate(Position("K7/8/8/8/8/8/8/6rk w - - 0 1")));
        TEST_EXPECT(-KnownWinValue >= Evaluate(Position("K7/8/8/8/8/8/8/6rk w - - 0 1")));

        // KvQ
        TEST_EXPECT(KnownWinValue <= Evaluate(Position("K7/Q7/8/8/8/8/8/7k w - - 0 1")));
        TEST_EXPECT(KnownWinValue <= Evaluate(Position("K7/Q7/8/8/8/8/8/7k w - - 0 1")));
        TEST_EXPECT(-KnownWinValue >= Evaluate(Position("K7/8/8/8/8/8/8/6qk w - - 0 1")));
        TEST_EXPECT(-KnownWinValue >= Evaluate(Position("K7/8/8/8/8/8/8/6qk w - - 0 1")));

        // KvP (white winning)
        TEST_EXPECT(KnownWinValue <= Evaluate(Position("7k/8/8/8/8/8/P7/K7 w - - 0 1")));
        TEST_EXPECT(KnownWinValue <= Evaluate(Position("7k/8/8/8/8/8/P7/K7 b - - 0 1")));
        TEST_EXPECT(KnownWinValue <= Evaluate(Position("8/8/1k6/8/8/1K6/1P6/8 w - - 0 1")));
        TEST_EXPECT(0 == Evaluate(Position("8/8/1k6/8/8/1K6/1P6/8 b - - 0 1")));
        TEST_EXPECT(0 == Evaluate(Position("5k2/8/8/8/8/8/P7/K7 w - - 0 1")));
        TEST_EXPECT(0 == Evaluate(Position("5k2/8/8/8/8/8/P7/K7 w - - 0 1")));

        // KvP (black winning)
        TEST_EXPECT(-KnownWinValue >= Evaluate(Position("7k/7p/8/8/8/8/8/K7 w - - 0 1")));
        TEST_EXPECT(-KnownWinValue >= Evaluate(Position("7k/7p/8/8/8/8/8/K7 b - - 0 1")));
        TEST_EXPECT(-KnownWinValue >= Evaluate(Position("8/6p1/6k1/8/8/6K1/8/8 b - - 0 1")));
        TEST_EXPECT(0 == Evaluate(Position("8/6p1/6k1/8/8/6K1/8/8 w - - 0 1")));
        TEST_EXPECT(0 == Evaluate(Position("7k/7p/8/8/8/8/8/2K5 w - - 0 1")));
        TEST_EXPECT(0 == Evaluate(Position("7k/7p/8/8/8/8/8/2K5 b - - 0 1")));

        // KBPvK (drawn)
        TEST_EXPECT(0 == Evaluate(Position("k7/P7/8/K7/3B4/8/P7/B7 w - - 0 1")));
        TEST_EXPECT(0 == Evaluate(Position("7k/7P/8/8/2B5/3B4/7P/6K1 w - - 0 1")));
        TEST_EXPECT(0 == Evaluate(Position("b7/p7/8/3b4/k7/8/p7/K7 b - - 0 1")));
        TEST_EXPECT(0 == Evaluate(Position("6k1/7p/3b4/2b5/8/8/7p/7K b - - 0 1")));

        // KBPvK (winning)
        TEST_EXPECT(0 < Evaluate(Position("7k/7P/8/8/2B5/3B4/6P1/6K1 w - - 0 1")));
        TEST_EXPECT(0 < Evaluate(Position("7k/7P/8/8/2B5/8/3B3P/6K1 w - - 0 1")));
        TEST_EXPECT(0 < Evaluate(Position("k7/P7/8/8/5B2/4B3/1P6/1K6 w - - 0 1")));
        TEST_EXPECT(0 < Evaluate(Position("k7/P7/8/8/5B2/8/P3B3/1K6 w - - 0 1")));
    }
}

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

    // zero depth search should return zero result
    {
        param.limits.maxDepth = 0;
        param.numPvLines = UINT32_MAX;

        game.Reset(Position(Position::InitPositionFEN));
        search.DoSearch(game, param, result);

        TEST_EXPECT(result.size() == 0);
    }

    // incufficient material draw
    {
        param.limits.maxDepth = 4;
        param.numPvLines = UINT32_MAX;

        game.Reset(Position("4k2K/8/8/8/8/8/8/8 w - - 0 1"));
        search.DoSearch(game, param, result);

        TEST_EXPECT(result.size() == 3);
        TEST_EXPECT(result[0].score == 0);
        TEST_EXPECT(result[1].score == 0);
        TEST_EXPECT(result[2].score == 0);
    }

    // stalemate (no legal move)
    {
        param.limits.maxDepth = 1;
        param.numPvLines = UINT32_MAX;

        game.Reset(Position("k7/2Q5/1K6/8/8/8/8/8 b - - 0 1"));
        search.DoSearch(game, param, result);

        TEST_EXPECT(result.size() == 0);
    }

    // mate in one
    {
        param.limits.maxDepth = 4;
        param.numPvLines = UINT32_MAX;

        game.Reset(Position("k7/7Q/1K6/8/8/8/8/8 w - - 0 1"));
        search.DoSearch(game, param, result);

        TEST_EXPECT(result.size() == 27);
        TEST_EXPECT(result[0].score == CheckmateValue - 1);
        TEST_EXPECT(result[1].score == CheckmateValue - 1);
        TEST_EXPECT(result[2].score == CheckmateValue - 1);
        TEST_EXPECT(result[3].score == CheckmateValue - 1);
    }

    // mate in two
    {
        param.limits.maxDepth = 4;
        param.limits.mateSearch = true;
        param.numPvLines = 1;

        game.Reset(Position("K4BB1/1Q6/5p2/8/2R2r1r/N2N2q1/kp1p1p1p/b7 w - - 0 1"));
        search.DoSearch(game, param, result);

        TEST_EXPECT(result.size() == 1);
        TEST_EXPECT(result[0].score == CheckmateValue - 3);
        TEST_EXPECT(result[0].moves.front() == Move::Make(Square_b7, Square_f3, Piece::Queen));

        param.limits.mateSearch = false;
    }

    // perpetual check
    {
        param.limits.maxDepth = 12;
        param.limits.mateSearch = true;
        param.numPvLines = 1;

        game.Reset(Position("6k1/6p1/8/6KQ/1r6/q2b4/8/8 w - - 0 1"));
        search.DoSearch(game, param, result);

        TEST_EXPECT(result.size() == 1);
        TEST_EXPECT(result[0].score == 0);
        TEST_EXPECT(result[0].moves.front() == Move::Make(Square_h5, Square_e8, Piece::Queen));

        param.limits.mateSearch = false;
    }

    // winnnig KPvK
    {
        param.limits.maxDepth = 1;
        param.numPvLines = UINT32_MAX;

        game.Reset(Position("4k3/8/8/8/8/8/5P2/5K2 w - - 0 1"));
        search.DoSearch(game, param, result);

        TEST_EXPECT(result.size() == 6);
        TEST_EXPECT(result[0].score > KnownWinValue);
        TEST_EXPECT(result[1].score > KnownWinValue);
        TEST_EXPECT(result[2].score == 0);
        TEST_EXPECT(result[3].score == 0);
        TEST_EXPECT(result[4].score == 0);
        TEST_EXPECT(result[5].score == 0);
    }

    // drawing KPvK
    {
        param.limits.maxDepth = 1;
        param.numPvLines = UINT32_MAX;

        game.Reset(Position("4k3/8/8/8/8/8/7P/7K w - - 0 1"));
        search.DoSearch(game, param, result);

        TEST_EXPECT(result.size() == 4);
        TEST_EXPECT(result[0].score == 0);
        TEST_EXPECT(result[1].score == 0);
        TEST_EXPECT(result[2].score == 0);
        TEST_EXPECT(result[3].score == 0);
    }

    // chess-rook skewer
    {
        param.limits.maxDepth = 1;
        param.numPvLines = UINT32_MAX;

        game.Reset(Position("3k3r/8/8/8/8/8/8/KR6 w - - 0 1"));
        search.DoSearch(game, param, result);

        TEST_EXPECT(result.size() == 15);

        TEST_EXPECT(result[0].moves.front() == Move::Make(Square_b1, Square_b8, Piece::Rook));
        TEST_EXPECT(result[0].score >= KnownWinValue);      // Rb8 is winning

        TEST_EXPECT(result[1].score < KnownWinValue);       // draw
        TEST_EXPECT(result[13].score < KnownWinValue);      // draw

        TEST_EXPECT(result[14].moves.front() == Move::Make(Square_b1, Square_h1, Piece::Rook));
        TEST_EXPECT(result[14].score <= -KnownWinValue);    // Rh1 is loosing
    }

    // Lasker-Reichhelm (TT test)
    {
        param.limits.maxDepth = 25;
        param.numPvLines = 1;

        game.Reset(Position("8/k7/3p4/p2P1p2/P2P1P2/8/8/K7 w - - 0 1"));
        search.DoSearch(game, param, result);

        TEST_EXPECT(result.size() == 1);
        TEST_EXPECT(result[0].score >= 100);
        TEST_EXPECT(result[0].moves.front() == Move::Make(Square_a1, Square_b1, Piece::King));
    }
}

void RunGameTests()
{
    std::cout << "Running Game tests..." << std::endl;

    Search search;
    TranspositionTable tt{ 16 * 1024 };
   
    SearchParam param{ tt };
    param.debugLog = false;
    param.numPvLines = UINT32_MAX;
    param.limits.maxDepth = 6;
    param.numPvLines = 1;

    Game game;
    game.Reset(Position(Position::InitPositionFEN));
    game.DoMove(Move::Make(Square_d2, Square_d4, Piece::Pawn));
    game.DoMove(Move::Make(Square_e7, Square_e5, Piece::Pawn));

    SearchResult result;
    search.DoSearch(game, param, result);

    TEST_EXPECT(result.size() == 1);
    TEST_EXPECT(result[0].moves.front() == Move::Make(Square_d4, Square_e5, Piece::Pawn, Piece::None, true));
    TEST_EXPECT(result[0].score > 0);
}

void RunUnitTests()
{
    RunPositionTests();
    RunEvalTests();
    RunSearchTests();
    RunPerftTests();
    RunGameTests();
}

bool RunPerformanceTests(const char* path)
{
    using MovesListType = std::vector<std::string>;

    struct TestCaseEntry
    {
        std::string positionStr;
        MovesListType bestMoves;
        MovesListType avoidMoves;
    };

    enum class ParsingMode
    {
        Position,
        BestMoves,
        AvoidMoves
    };

    std::vector<TestCaseEntry> testVector;
    {
        std::ifstream file(path);
        if (!file.good())
        {
            std::cout << "Failed to open testcases file" << std::endl;
            return false;
        }

        std::string lineStr;
        while (std::getline(file, lineStr))
        {
            size_t endPos = lineStr.find(';');
            if (endPos != std::string::npos)
            {
                lineStr = lineStr.substr(0, endPos);
            }

            // tokenize line
            std::istringstream iss(lineStr);
            std::vector<std::string> tokens(
                std::istream_iterator<std::string>{iss},
                std::istream_iterator<std::string>());

            std::string positionStr;
            MovesListType bestMoves;
            MovesListType avoidMoves;
            ParsingMode parsingMode = ParsingMode::Position;

            for (size_t i = 0; i < tokens.size(); ++i)
            {
                if (tokens[i] == "bm")
                {
                    parsingMode = ParsingMode::BestMoves;
                    continue;
                }
                else if (tokens[i] == "am")
                {
                    parsingMode = ParsingMode::AvoidMoves;
                    continue;
                }
                else if (tokens[i] == ";")
                {
                    break;
                }
                else
                {
                    if (parsingMode == ParsingMode::BestMoves)
                    {
                        bestMoves.push_back(tokens[i]);
                    }
                    else if (parsingMode == ParsingMode::AvoidMoves)
                    {
                        avoidMoves.push_back(tokens[i]);
                    }
                    else
                    {
                        if (!positionStr.empty()) positionStr += ' ';
                        positionStr += tokens[i];
                    }
                }
            }

            Position pos;
            if (!pos.FromFEN(positionStr))
            {
                std::cout << "Test case has invalid position: " << positionStr << std::endl;
                return false;
            }

            if (bestMoves.empty() && avoidMoves.empty())
            {
                std::cout << "Test case is missing best move: " << positionStr << std::endl;
                return false;
            }

            // TODO check if move is valid

            testVector.push_back({positionStr, bestMoves, avoidMoves});
        }
    }
    std::cout << testVector.size() << " test positions loaded" << std::endl;

    bool verbose = false;

    TranspositionTable tt(2048ull * 1024ull * 1024ull);
    std::vector<Search> searchArray{ std::thread::hardware_concurrency() };

    uint32_t maxSearchTime = 4;

    for (;;)
    {
        std::mutex mutex;
        std::atomic<uint32_t> success = 0;

        auto startTimeAll = std::chrono::high_resolution_clock::now();

        tt.NextGeneration();

        Waitable waitable;
        {
            TaskBuilder taskBuilder(waitable);

            for (const TestCaseEntry& testCase : testVector)
            {
                taskBuilder.Task("SearchTest", [testCase, &searchArray, maxSearchTime, &mutex, verbose, &success, &tt](const TaskContext& ctx)
                {
                    Search& search = searchArray[ctx.threadId];

                    const Position position(testCase.positionStr);
                    TEST_EXPECT(position.IsValid());

                    Game game;
                    game.Reset(position);

                    const TimePoint startTimePoint = TimePoint::GetCurrent();

                    SearchParam searchParam{ tt };
                    searchParam.debugLog = false;
                    searchParam.numThreads = 1;
                    searchParam.limits.maxDepth = UINT8_MAX;
                    searchParam.limits.maxTime = startTimePoint + TimePoint::FromSeconds(maxSearchTime * 0.001f);
                    searchParam.limits.maxTimeSoft = startTimePoint + TimePoint::FromSeconds(maxSearchTime * 0.001f / 2.0f);
                    searchParam.limits.analysisMode = true;

                    SearchResult searchResult;
                    search.DoSearch(game, searchParam, searchResult);

                    Move foundMove = Move::Invalid();
                    if (!searchResult[0].moves.empty())
                    {
                        foundMove = searchResult[0].moves[0];
                    }

                    if (!foundMove.IsValid())
                    {
                        std::unique_lock<std::mutex> lock(mutex);
                        std::cout << "[FAILURE] No move found! position: " << testCase.positionStr << std::endl;
                        return;
                    }

                    const std::string foundMoveStrLAN = position.MoveToString(foundMove, MoveNotation::LAN);
                    const std::string foundMoveStrSAN = position.MoveToString(foundMove, MoveNotation::SAN);
                    bool correctMoveFound = false;
                    if (!testCase.bestMoves.empty())
                    {
                        for (const std::string& bestMoveStr : testCase.bestMoves)
                        {
                            if (foundMoveStrLAN == bestMoveStr || foundMoveStrSAN == bestMoveStr)
                            {
                                correctMoveFound = true;
                            }
                        }
                    }
                    else
                    {
                        correctMoveFound = true;
                        for (const std::string& avoidMoveStr : testCase.avoidMoves)
                        {
                            if (foundMoveStrLAN == avoidMoveStr || foundMoveStrSAN == avoidMoveStr)
                            {
                                correctMoveFound = false;
                            }
                        }
                    }

                    if (!correctMoveFound)
                    {
                        if (verbose)
                        {
                            std::unique_lock<std::mutex> lock(mutex);
                            std::cout << "[FAILURE] Wrong move found! ";

                            if (!testCase.bestMoves.empty())
                            {
                                std::cout << "expected: ";
                                for (const std::string& bestMoveStr : testCase.bestMoves) std::cout << bestMoveStr << " ";
                            }
                            else if (!testCase.avoidMoves.empty())
                            {
                                std::cout << "not expected: ";
                                for (const std::string& bestMoveStr : testCase.avoidMoves) std::cout << bestMoveStr << " ";
                            }

                            std::cout << "found: " << foundMoveStrLAN << " position: " << testCase.positionStr << std::endl;
                        }
                        return;
                    }

                    {
                        if (verbose)
                        {
                            std::unique_lock<std::mutex> lock(mutex);
                            std::cout << "[SUCCESS] Found valid move: " << foundMoveStrLAN << std::endl;
                        }
                        success++;
                    }
                });
            }

            //taskBuilder.Fence();
        }

        waitable.Wait();

        auto endTimeAll = std::chrono::high_resolution_clock::now();
        const float time = std::chrono::duration_cast<std::chrono::microseconds>(endTimeAll - startTimeAll).count() / 1000000.0f;
        
        const float passRate = !testVector.empty() ? (float)success / (float)testVector.size() : 0.0f;
        const float factor = passRate / time;

        std::cout
            << maxSearchTime << "; "
            << success << "; "
            << passRate << "; "
            << time << "; "
            << factor << std::endl;

        maxSearchTime *= 3;
        maxSearchTime /= 2;

        tt.Clear();

        //std::cout << "Passed: " << success << "/" << testVector.size() << std::endl;
        //std::cout << "Time:   " << std::chrono::duration_cast<std::chrono::milliseconds>(endTimeAll - startTimeAll).count() << "ms" << std::endl << std::endl;
    }

    return true;
}

int main(int argc, const char* argv[])
{
    InitEngine();

    nnue_init("D:/CHESS/NNUE/nn-04cf2b4ed1da.nnue");

    LoadTablebase("C:/Program Files (x86)/syzygy/");

    if (argc > 1 && strcmp(argv[1], "unittest") == 0)
    {
        RunUnitTests();
    }
    else if (argc > 2 && strcmp(argv[1], "perftest") == 0)
    {
        RunPerformanceTests(argv[2]);
    }
    else
    {
        return 1;
    }

    return 0;
}
