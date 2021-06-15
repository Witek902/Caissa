#include <iostream>
#include "Position.hpp"
#include "MoveList.hpp"
#include "Search.hpp"
#include "Evaluate.hpp"
#include "UCI.hpp"

#include <chrono>
#include <mutex>
#include <fstream>

#include "ThreadPool.hpp"

using namespace threadpool;

#define TEST_EXPECT(x) \
    if (!(x)) { std::cout << "Test failed: " << #x << std::endl; __debugbreak();}

void RunPerft()
{
    const Position pos("r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10");

    auto start = std::chrono::high_resolution_clock::now();
    //TEST_EXPECT(pos.Perft(4) == 3894594u);
    TEST_EXPECT(pos.Perft(5) == 164075551u);
    auto finish = std::chrono::high_resolution_clock::now();

    std::cout << "Elapsed time: " << std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count() / 1000000.0 << " s\n";
}

void RunUnitTests()
{
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
    }

    // FEN printing
    {
        Position pos(Position::InitPositionFEN);
        TEST_EXPECT(pos.ToFEN() == Position::InitPositionFEN);
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
            TEST_EXPECT(move.fromSquare == Square_e2);
            TEST_EXPECT(move.toSquare == Square_e4);
            TEST_EXPECT(move.piece == Piece::Pawn);
            TEST_EXPECT(move.isCapture == false);
            TEST_EXPECT(move.promoteTo == Piece::None);
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
            TEST_EXPECT(move.fromSquare == Square_e2);
            TEST_EXPECT(move.toSquare == Square_e4);
            TEST_EXPECT(move.piece == Piece::Pawn);
            TEST_EXPECT(move.isCapture == false);
            TEST_EXPECT(move.promoteTo == Piece::None);
            TEST_EXPECT(!pos.IsMoveValid(move));
        }

        // move pawn (invalid, blocked)
        {
            Position pos("rnbqkbnr/pppp1ppp/8/8/4p3/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
            const Move move = pos.MoveFromString("e2e4");
            TEST_EXPECT(move.IsValid());
            TEST_EXPECT(move.fromSquare == Square_e2);
            TEST_EXPECT(move.toSquare == Square_e4);
            TEST_EXPECT(move.piece == Piece::Pawn);
            TEST_EXPECT(move.promoteTo == Piece::None);
            TEST_EXPECT(!pos.IsMoveValid(move));
        }

        // move pawn (invalid, blocked)
        {
            Position pos("rnbqkbnr/1ppppppp/p7/5B2/8/3P4/PPP1PPPP/RN1QKBNR b KQkq - 0 1");
            const Move move = pos.MoveFromString("f7f5");
            TEST_EXPECT(move.IsValid());
            TEST_EXPECT(move.fromSquare == Square_f7);
            TEST_EXPECT(move.toSquare == Square_f5);
            TEST_EXPECT(move.piece == Piece::Pawn);
            TEST_EXPECT(move.promoteTo == Piece::None);
            TEST_EXPECT(!pos.IsMoveValid(move));
        }

        // pawn capture
        {
            Position pos("rnbqkbnr/p1pppppp/8/1p6/2P5/8/PP1PPPPP/RNBQKBNR w KQkq - 0 1");
            const Move move = pos.MoveFromString("c4b5");
            TEST_EXPECT(move.IsValid());
            TEST_EXPECT(move.fromSquare == Square_c4);
            TEST_EXPECT(move.toSquare == Square_b5);
            TEST_EXPECT(move.piece == Piece::Pawn);
            TEST_EXPECT(move.isCapture == true);
            TEST_EXPECT(move.isEnPassant == false);
            TEST_EXPECT(move.promoteTo == Piece::None);
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
            TEST_EXPECT(move.fromSquare == Square_d5);
            TEST_EXPECT(move.toSquare == Square_c6);
            TEST_EXPECT(move.piece == Piece::Pawn);
            TEST_EXPECT(move.isCapture == true);
            TEST_EXPECT(move.isEnPassant == true);
            TEST_EXPECT(move.promoteTo == Piece::None);
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
            TEST_EXPECT(move.fromSquare == Square_e2);
            TEST_EXPECT(move.toSquare == Square_d3);
            TEST_EXPECT(move.piece == Piece::Pawn);
            TEST_EXPECT(move.isCapture == true);
            TEST_EXPECT(move.isEnPassant == true);
            TEST_EXPECT(move.promoteTo == Piece::None);
            TEST_EXPECT(!pos.IsMoveValid(move));
        }

        // move pawn (invalid promotion)
        {
            Position pos("1k6/5P2/8/8/8/8/8/4K3 w - - 0 1");
            const Move move = pos.MoveFromString("f7f8k");
            TEST_EXPECT(move.IsValid());
            TEST_EXPECT(move.fromSquare == Square_f7);
            TEST_EXPECT(move.toSquare == Square_f8);
            TEST_EXPECT(move.piece == Piece::Pawn);
            TEST_EXPECT(move.isCapture == false);
            TEST_EXPECT(move.promoteTo == Piece::King);
            TEST_EXPECT(!pos.IsMoveValid(move));
        }

        // move pawn (valid promotion)
        {
            Position pos("1k6/5P2/8/8/8/8/8/4K3 w - - 0 1");
            const Move move = pos.MoveFromString("f7f8q");
            TEST_EXPECT(move.IsValid());
            TEST_EXPECT(move.fromSquare == Square_f7);
            TEST_EXPECT(move.toSquare == Square_f8);
            TEST_EXPECT(move.piece == Piece::Pawn);
            TEST_EXPECT(move.isCapture == false);
            TEST_EXPECT(move.promoteTo == Piece::Queen);
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
            TEST_EXPECT(move.fromSquare == Square_d3);
            TEST_EXPECT(move.toSquare == Square_f4);
            TEST_EXPECT(move.piece == Piece::Knight);
            TEST_EXPECT(move.isCapture == false);
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
            TEST_EXPECT(move.fromSquare == Square_d3);
            TEST_EXPECT(move.toSquare == Square_f4);
            TEST_EXPECT(move.piece == Piece::Knight);
            TEST_EXPECT(move.isCapture == true);
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
            TEST_EXPECT(move.fromSquare == Square_e1);
            TEST_EXPECT(move.toSquare == Square_g1);
            TEST_EXPECT(move.piece == Piece::King);
            TEST_EXPECT(move.isCapture == false);
            TEST_EXPECT(move.isCastling == true);
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
            TEST_EXPECT(move.fromSquare == Square_e1);
            TEST_EXPECT(move.toSquare == Square_g1);
            TEST_EXPECT(move.piece == Piece::King);
            TEST_EXPECT(move.isCapture == false);
            TEST_EXPECT(move.isCastling == true);
            TEST_EXPECT(!pos.IsMoveValid(move));
        }

        // castling, whites, queen side
        {
            Position pos("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/R3KBNR w KQkq - 0 1");
            const Move move = pos.MoveFromString("e1c1");
            TEST_EXPECT(move.IsValid());
            TEST_EXPECT(move.fromSquare == Square_e1);
            TEST_EXPECT(move.toSquare == Square_c1);
            TEST_EXPECT(move.piece == Piece::King);
            TEST_EXPECT(move.isCapture == false);
            TEST_EXPECT(move.isCastling == true);
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
            TEST_EXPECT(move.fromSquare == Square_e1);
            TEST_EXPECT(move.toSquare == Square_c1);
            TEST_EXPECT(move.piece == Piece::King);
            TEST_EXPECT(move.isCapture == false);
            TEST_EXPECT(move.isCastling == true);
            TEST_EXPECT(!pos.IsMoveValid(move));
        }

        // castling, blacks, king side
        {
            Position pos("rnbqk2r/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1");
            const Move move = pos.MoveFromString("e8g8");
            TEST_EXPECT(move.IsValid());
            TEST_EXPECT(move.fromSquare == Square_e8);
            TEST_EXPECT(move.toSquare == Square_g8);
            TEST_EXPECT(move.piece == Piece::King);
            TEST_EXPECT(move.isCapture == false);
            TEST_EXPECT(move.isCastling == true);
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
            TEST_EXPECT(move.fromSquare == Square_e8);
            TEST_EXPECT(move.toSquare == Square_g8);
            TEST_EXPECT(move.piece == Piece::King);
            TEST_EXPECT(move.isCapture == false);
            TEST_EXPECT(move.isCastling == true);
            TEST_EXPECT(!pos.IsMoveValid(move));
        }

        // castling, blacks, queen side
        {
            Position pos("r3kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1");
            const Move move = pos.MoveFromString("e8c8");
            TEST_EXPECT(move.IsValid());
            TEST_EXPECT(move.fromSquare == Square_e8);
            TEST_EXPECT(move.toSquare == Square_c8);
            TEST_EXPECT(move.piece == Piece::King);
            TEST_EXPECT(move.isCapture == false);
            TEST_EXPECT(move.isCastling == true);
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
            TEST_EXPECT(move.fromSquare == Square_e8);
            TEST_EXPECT(move.toSquare == Square_c8);
            TEST_EXPECT(move.piece == Piece::King);
            TEST_EXPECT(move.isCapture == false);
            TEST_EXPECT(move.isCastling == true);
            TEST_EXPECT(!pos.IsMoveValid(move));
        }

        // illegal castling, whites, king side, king in check
        {
            Position pos("4k3/4r3/8/8/8/8/8/R3K2R w KQ - 0 1");
            const Move move = pos.MoveFromString("e1g1");
            TEST_EXPECT(move.IsValid());
            TEST_EXPECT(move.fromSquare == Square_e1);
            TEST_EXPECT(move.toSquare == Square_g1);
            TEST_EXPECT(move.piece == Piece::King);
            TEST_EXPECT(move.isCapture == false);
            TEST_EXPECT(move.isCastling == true);
            TEST_EXPECT(!pos.IsMoveValid(move));
        }

        // illegal castling, whites, king side, king crossing check
        {
            Position pos("4kr2/8/8/8/8/8/8/R3K2R w KQ - 0 1");
            const Move move = pos.MoveFromString("e1g1");
            TEST_EXPECT(move.IsValid());
            TEST_EXPECT(move.fromSquare == Square_e1);
TEST_EXPECT(move.toSquare == Square_g1);
TEST_EXPECT(move.piece == Piece::King);
TEST_EXPECT(move.isCapture == false);
TEST_EXPECT(move.isCastling == true);
TEST_EXPECT(!pos.IsMoveValid(move));
        }

        // move rook, loose castling rights
        {
        Position pos("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1");
        const Move move = pos.MoveFromString("a1b1");
        TEST_EXPECT(move.IsValid());
        TEST_EXPECT(move.fromSquare == Square_a1);
        TEST_EXPECT(move.toSquare == Square_b1);
        TEST_EXPECT(move.piece == Piece::Rook);
        TEST_EXPECT(move.isCapture == false);
        TEST_EXPECT(move.isCastling == false);
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
            TEST_EXPECT(move.fromSquare == Square_h1);
            TEST_EXPECT(move.toSquare == Square_g1);
            TEST_EXPECT(move.piece == Piece::Rook);
            TEST_EXPECT(move.isCapture == false);
            TEST_EXPECT(move.isCastling == false);
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
            TEST_EXPECT(move.fromSquare == Square_a8);
            TEST_EXPECT(move.toSquare == Square_b8);
            TEST_EXPECT(move.piece == Piece::Rook);
            TEST_EXPECT(move.isCapture == false);
            TEST_EXPECT(move.isCastling == false);
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
            TEST_EXPECT(move.fromSquare == Square_h8);
            TEST_EXPECT(move.toSquare == Square_g8);
            TEST_EXPECT(move.piece == Piece::Rook);
            TEST_EXPECT(move.isCapture == false);
            TEST_EXPECT(move.isCastling == false);
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
            TEST_EXPECT(move.fromSquare == Square_h8);
            TEST_EXPECT(move.toSquare == Square_g7);
            TEST_EXPECT(move.piece == Piece::King);
            TEST_EXPECT(move.isCapture == false);
            TEST_EXPECT(move.isCastling == false);
            TEST_EXPECT(pos.IsMoveValid(move));
            TEST_EXPECT(!pos.IsMoveLegal(move));
        }

        // pin
        {
            Position pos("k7/8/q7/8/R7/8/8/K7 w - - 0 1");
            const Move move = pos.MoveFromString("a4b4");
            TEST_EXPECT(move.IsValid());
            TEST_EXPECT(move.fromSquare == Square_a4);
            TEST_EXPECT(move.toSquare == Square_b4);
            TEST_EXPECT(move.piece == Piece::Rook);
            TEST_EXPECT(move.isCapture == false);
            TEST_EXPECT(move.isCastling == false);
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
            TEST_EXPECT(1 == pos.StaticExchangeEvaluation(move));
        }

        // hanging pawn
        {
            Position pos("7k/8/1p6/8/8/1Q6/8/7K w - - 0 1");
            const Move move = pos.MoveFromString("b3b6");
            TEST_EXPECT(move.IsValid());
            TEST_EXPECT(1 == pos.StaticExchangeEvaluation(move));
        }

        // queen takes pawn protected by another pawn
        {
            Position pos("7k/p7/1p6/8/8/1Q6/8/7K w - - 0 1");
            const Move move = pos.MoveFromString("b3b6");
            TEST_EXPECT(move.IsValid());
            TEST_EXPECT(0 == pos.StaticExchangeEvaluation(move));
        }

        // queen trade
        {
            Position pos("7k/p7/1q6/8/8/1Q6/8/7K w - - 0 1");
            const Move move = pos.MoveFromString("b3b6");
            TEST_EXPECT(move.IsValid());
            TEST_EXPECT(1 == pos.StaticExchangeEvaluation(move));
        }

        // rook trade
        {
            Position pos("7k/p7/1q6/8/8/1Q6/8/7K w - - 0 1");
            const Move move = pos.MoveFromString("b3b6");
            TEST_EXPECT(move.IsValid());
            TEST_EXPECT(1 == pos.StaticExchangeEvaluation(move));
        }

        // (rook+bishop) vs. 2 knights -> bishop
        {
            Position pos("7k/3n4/1n6/8/8/1R2B3/8/7K w - - 0 1");
            const Move move = pos.MoveFromString("b3b6");
            TEST_EXPECT(move.IsValid());
            TEST_EXPECT(1 == pos.StaticExchangeEvaluation(move));
        }
    }

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

bool RunSearchTests()
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
        std::ifstream file("data/testPositions.txt");
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

            testVector.emplace_back(std::move(positionStr), std::move(bestMoves), std::move(avoidMoves));
        }
    }
    std::cout << testVector.size() << " test positions loaded" << std::endl;

    const uint32_t minDepth = 1;
    const uint32_t maxDepth = 10;

    bool verbose = false;

    std::vector<Search> searchArray{ std::thread::hardware_concurrency() };

    for (uint32_t depth = minDepth; depth <= maxDepth; ++depth)
    {
        std::mutex mutex;
        std::atomic<uint32_t> success = 0;

        for (Search& search : searchArray)
        {
            search.GetTranspositionTable().Clear();
        }

        auto startTimeAll = std::chrono::high_resolution_clock::now();

        Waitable waitable;
        {
            TaskBuilder taskBuilder(waitable);

            for (const TestCaseEntry& testCase : testVector)
            {
                taskBuilder.Task("SearchTest", [testCase, &searchArray, depth, &mutex, verbose, &success](const TaskContext& ctx)
                {
                    Search& search = searchArray[ctx.threadId];

                    const Position position(testCase.positionStr);
                    TEST_EXPECT(position.IsValid());

                    SearchParam searchParam;
                    searchParam.debugLog = false;
                    searchParam.maxDepth = depth;

                    SearchResult searchResult;
                    search.DoSearch(position, searchParam, searchResult);

                    Move foundMove;
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

                    const std::string foundMoveStr = position.MoveToString(foundMove);
                    bool correctMoveFound = false;
                    if (!testCase.bestMoves.empty())
                    {
                        for (const std::string& bestMoveStr : testCase.bestMoves)
                        {
                            if (foundMoveStr == bestMoveStr)
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
                            if (foundMoveStr == avoidMoveStr)
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

                            std::cout << "found: " << foundMoveStr << " position: " << testCase.positionStr << std::endl;
                        }
                        return;
                    }

                    {
                        if (verbose)
                        {
                            std::unique_lock<std::mutex> lock(mutex);
                            std::cout << "[SUCCESS] Found valid move: " << foundMoveStr << std::endl;
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
        
        const float passRate = (float)success / (float)testVector.size();
        const float factor = passRate / time;

        std::cout
            << depth << "; "
            << success << "; "
            << passRate << "; "
            << time << "; "
            << factor << std::endl;

        //std::cout << "Passed: " << success << "/" << testVector.size() << std::endl;
        //std::cout << "Time:   " << std::chrono::duration_cast<std::chrono::milliseconds>(endTimeAll - startTimeAll).count() << "ms" << std::endl << std::endl;
    }

    return true;
    //return success == testVector.size();
}

void RunSearchPerfTest()
{
    Search search;

    //const Position position("k7/3Q4/pp6/8/8/1q3PPP/5PPP/7K w - - 0 1"); // repetition
    const Position position("r2q1r1k/pb3p1p/2n1p2Q/5p2/8/3B2N1/PP3PPP/R3R1K1 w - - 0 1");
    //const Position position("r1k4r/ppp1bq1p/2n1N3/6B1/3p2Q1/8/PPP2PPP/R5K1 w - - 0 1"); // mate in 6
    //const Position position("1K1k4/1P6/8/8/8/8/r7/2R5 w - - 0 1"); // Lucena
    //const Position position("8/6k1/8/8/8/8/P7/7K w - - 0 1");
    //const Position position("8/8/8/3k4/8/8/8/3KQ3 w - - 0 1");
    //const Position position("r4r1k/1p2p1b1/2ppb2p/p1Pn1pnq/N1NP2pP/1P2P1P1/P1Q1BP2/2RRB1K1 w - - 1 25");
    TEST_EXPECT(position.IsValid());

    SearchParam searchParam;
    searchParam.debugLog = true;
    searchParam.maxDepth = 8;
    searchParam.numPvLines = 1;

    SearchResult searchResult;
    search.DoSearch(position, searchParam, searchResult);
    search.DoSearch(position, searchParam, searchResult);
    search.DoSearch(position, searchParam, searchResult);
}
