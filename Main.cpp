#include <iostream>
#include "Position.hpp"
#include "Move.hpp"
#include "Search.hpp"
#include "UCI.hpp"

#include <chrono>
#include <mutex>

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

void RunTests()
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
            TEST_EXPECT(pos.ToFEN() == "4k3/8/8/8/5N2/8/8/4K3 b - - 0 1");
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
            TEST_EXPECT(pos.ToFEN() == "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQ1RK1 b kq - 0 1");
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
            TEST_EXPECT(pos.ToFEN() == "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/2KR1BNR b kq - 0 1");
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
            TEST_EXPECT(pos.ToFEN() == "rnbq1rk1/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQ - 0 1");
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
            TEST_EXPECT(pos.ToFEN() == "2kr1bnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQ - 0 1");
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
            TEST_EXPECT(pos.ToFEN() == "r3k2r/8/8/8/8/8/8/1R2K2R b Kkq - 0 1");
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
            TEST_EXPECT(pos.ToFEN() == "r3k2r/8/8/8/8/8/8/R3K1R1 b Qkq - 0 1");
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
            TEST_EXPECT(pos.ToFEN() == "1r2k2r/8/8/8/8/8/8/R3K2R w KQk - 0 1");
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
            TEST_EXPECT(pos.ToFEN() == "r3k1r1/8/8/8/8/8/8/R3K2R w KQq - 0 1");
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
    using BestMovesType = std::vector<const char*>;
    using TestCaseType = std::pair<const char*, BestMovesType>;

    std::vector<TestCaseType> testVector =
    {
        // K v KP
        { "8/6k1/8/8/8/8/P7/7K w - - 0 1", { "a4" } },

        // K v KQ
        { "8/8/8/3k4/8/8/8/3KQ3 w - - 0 1", { "Qe7" } }, // mate in 7
        { "k7/8/8/8/8/8/8/K5Q1 w - - 0 1", { "Qg7" } },
        { "8/8/8/4k3/8/8/1K6/Q7 w - - 0 1", { "Qf1", "Qa6" } },

        // Zugzwang
        { "8/6p1/4p3/2K1Pk2/p7/P7/7P/8 w - - 0 1", { "Kd6" } },

        // Null Move tests
        { "8/8/p1p5/1p5p/1P5p/8/PPP2K1p/4R1rk w - - 0 1", { "Rf1" } },
        { "1q1k4/2Rr4/8/2Q3K1/8/8/8/8 w - - 0 1", { "Kh6" } },
        { "7k/5K2/5P1p/3p4/6P1/3p4/8/8 w - - 0 1", { "g5", "Ke7" } },
        { "8/6B1/p5p1/Pp4kp/1P5r/5P1Q/4q1PK/8 w - - 0 32", { "Qxh4+" } },
        { "8/8/1p1r1k2/p1pPN1p1/P3KnP1/1P6/8/3R4 b - - 0 1", { "Nxd5" } },

        { "2rr3k/pp3pp1/1nnqbN1p/3pN3/2pP4/2P3Q1/PPB4P/R4RK1 w - - 0 1", { "Qg3" } }, // mate in 3

        // Bratko-Kopec test suite (1982)
        { "1k1r4/pp1b1R2/3q2pp/4p3/2B5/4Q3/PPP2B2/2K5 b - - 0 1", { "Qd1+" } },
        { "3r1k2/4npp1/1ppr3p/p6P/P2PPPP1/1NR5/5K2/2R5 w - - 0 1", { "d5" } },
        { "2q1rr1k/3bbnnp/p2p1pp1/2pPp3/PpP1P1P1/1P2BNNP/2BQ1PRK/7R b - - 0 1", { "f5" } },
        { "rnbqkb1r/p3pppp/1p6/2ppP3/3N4/2P5/PPP1QPPP/R1B1KB1R w KQkq - 0 1", { "e6" } },
        { "r1b2rk1/2q1b1pp/p2ppn2/1p6/3QP3/1BN1B3/PPP3PP/R4RK1 w - - 0 1", { "Nd5", "a4" } },
        { "2r3k1/pppR1pp1/4p3/4P1P1/5P2/1P4K1/P1P5/8 w - - 0 1", { "g6" } },
        { "1nk1r1r1/pp2n1pp/4p3/q2pPp1N/b1pP1P2/B1P2R2/2P1B1PP/R2Q2K1 w - - 0 1", { "Nf6" } },
        { "4b3/p3kp2/6p1/3pP2p/2pP1P2/4K1P1/P3N2P/8 w - - 0 1", { "f5" } },
        { "2kr1bnr/pbpq4/2n1pp2/3p3p/3P1P1B/2N2N1Q/PPP3PP/2KR1B1R w - - 0 1", { "f5" } },
        { "3rr1k1/pp3pp1/1qn2np1/8/3p4/PP1R1P2/2P1NQPP/R1B3K1 b - - 0 1", { "Ne5" } },
        { "2r1nrk1/p2q1ppp/bp1p4/n1pPp3/P1P1P3/2PBB1N1/4QPPP/R4RK1 w - - 0 1", { "f4" } },
        { "r3r1k1/ppqb1ppp/8/4p1NQ/8/2P5/PP3PPP/R3R1K1 b - - 0 1", { "Bf5" } },
        { "r2q1rk1/4bppp/p2p4/2pP4/3pP3/3Q4/PP1B1PPP/R3R1K1 w - - 0 1", { "b4" } },
        { "rnb2r1k/pp2p2p/2pp2p1/q2P1p2/8/1Pb2NP1/PB2PPBP/R2Q1RK1 w - - 0 1", { "Qd2", "Qe1" } },
        { "2r3k1/1p2q1pp/2b1pr2/p1pp4/6Q1/1P1PP1R1/P1PN2PP/5RK1 w - - 0 1", { "Qxg7+" } },
        { "r1bqkb1r/4npp1/p1p4p/1p1pP1B1/8/1B6/PPPN1PPP/R2Q1RK1 w kq - 0 1", { "Ne4" } },
        { "r2q1rk1/1ppnbppp/p2p1nb1/3Pp3/2P1P1P1/2N2N1P/PPB1QP2/R1B2RK1 b - - 0 1", { "h5" } },
        { "r1bq1rk1/pp2ppbp/2np2p1/2n5/P3PP2/N1P2N2/1PB3PP/R1B1QRK1 b - - 0 1", { "Nb3" } },
        { "3rr3/2pq2pk/p2p1pnp/8/2QBPP2/1P6/P5PP/4RRK1 b - - 0 1", { "Rxe4" } },
        { "r4k2/pb2bp1r/1p1qp2p/3pNp2/3P1P2/2N3P1/PPP1Q2P/2KRR3 w - - 0 1", { "g4" } },
        { "3rn2k/ppb2rpp/2ppqp2/5N2/2P1P3/1P5Q/PB3PPP/3RR1K1 w - - 0 1", { "Nh6" } },
        { "2r2rk1/1bqnbpp1/1p1ppn1p/pP6/N1P1P3/P2B1N1P/1B2QPP1/R2R2K1 b - - 0 1", { "Bxe4" } },
        { "r1bqk2r/pp2bppp/2p5/3pP3/P2Q1P2/2N1B3/1PP3PP/R4RK1 b kq - 0 1", { "f6" } },
        { "r2qnrnk/p2b2b1/1p1p2pp/2pPpp2/1PP1P3/PRNBB3/3QNPPP/5RK1 w - - 0 1", { "f4" } },
    };

    std::mutex mutex;
    uint32_t success = 0;

    Waitable waitable;
    {
        TaskBuilder taskBuilder(waitable);

        for (const auto& iter : testVector)
        {
            taskBuilder.Task("SearchTest", [&iter, &mutex, &success](const TaskContext&)
            {
                Search search;

                const char* fenStr = iter.first;
                const BestMovesType& bestMoves = iter.second;

                const Position position(fenStr);
                TEST_EXPECT(position.IsValid());

                SearchParam searchParam;
                searchParam.debugLog = false;
                searchParam.transpositionTableSize = 8 * 1024 * 1024;
                searchParam.maxDepth = 12;

                Move foundMove;
                search.DoSearch(position, foundMove, searchParam);

                if (!foundMove.IsValid())
                {
                    std::unique_lock<std::mutex> lock(mutex);
                    std::cout << "[FAILURE] No move found! position: " << fenStr << std::endl;
                    return;
                }

                const std::string foundMoveStr = position.MoveToString(foundMove);
                bool correctMoveFound = false;
                for (const char* bestMoveStr : bestMoves)
                {
                    if (foundMoveStr == bestMoveStr)
                    {
                        correctMoveFound = true;
                    }
                }

                if (!correctMoveFound)
                {
                    std::unique_lock<std::mutex> lock(mutex);
                    std::cout << "[FAILURE] Wrong move found! expected: ";
                    for (const char* bestMoveStr : bestMoves) std::cout << bestMoveStr << " ";
                    std::cout << "found: " << foundMoveStr << " position: " << fenStr << std::endl;
                    return;
                }

                {
                    std::unique_lock<std::mutex> lock(mutex);
                    std::cout << "[SUCCESS] Found valid move: " << foundMoveStr << std::endl;
                    success++;
                }
            });
        }
    }

    waitable.Wait();

    std::cout << "Move test summary:" << std::endl;
    std::cout << "Test cases: " << testVector.size() << std::endl;
    std::cout << "Passed: " << success << std::endl;
    std::cout << "Failed: " << (testVector.size() - success) << std::endl;

    return success == testVector.size();
}

void RunSearchPerfTest()
{
    Search search;

    //const Position position("k7/3Q4/pp6/8/8/1q3PPP/5PPP/7K w - - 0 1"); // repetition
    //const Position position("r2q1r1k/pb3p1p/2n1p2Q/5p2/8/3B2N1/PP3PPP/R3R1K1 w - - 0 1");
    const Position position("r1k4r/ppp1bq1p/2n1N3/6B1/3p2Q1/8/PPP2PPP/R5K1 w - - 0 1"); // mate in 6
    //const Position position("1K1k4/1P6/8/8/8/8/r7/2R5 w - - 0 1"); // Lucena
    //const Position position("8/6k1/8/8/8/8/P7/7K w - - 0 1");
    //const Position position("8/8/8/3k4/8/8/8/3KQ3 w - - 0 1");
    //const Position position("r4r1k/1p2p1b1/2ppb2p/p1Pn1pnq/N1NP2pP/1P2P1P1/P1Q1BP2/2RRB1K1 w - - 1 25");
    TEST_EXPECT(position.IsValid());

    SearchParam searchParam;
    searchParam.debugLog = true;
    searchParam.maxDepth = 5;

    Move foundMove;
    search.DoSearch(position, foundMove, searchParam);
}

void SelfMatch()
{
    Position position(Position::InitPositionFEN);

    Search search;

    for (int i = 0; i < 1000; ++i)
    {
        std::cout << "Move #" << i << std::endl;
        std::cout << position.Print() << std::endl;
        std::cout << position.ToFEN() << std::endl;

        SearchParam searchParam;
        searchParam.debugLog = true;
        searchParam.maxDepth = 12;

        Move bestMove;
        const Search::ScoreType score = search.DoSearch(position, bestMove, searchParam);

        if (!bestMove.IsValid())
        {
            if (score > 0)
            {
                std::cout << "WHITES WON!" << std::endl;
            }
            else if (score < 0)
            {
                std::cout << "BLACKS WON!" << std::endl;
            }
            else
            {
                std::cout << "GAME ENDED WITH A DRAW!" << std::endl;
            }
            break;
        }

        search.RecordBoardPosition(position);
        position.DoMove(bestMove);
    }
}

void PlayGame()
{
    Position position(Position::InitPositionFEN);

    Search search;
    for (;;)
    {
        std::cout << position.Print() << std::endl;

        Move move;
        for (;;)
        {
            std::string moveStr;
            std::cout << "Type move: ";
            std::cin >> moveStr;

            move = position.MoveFromString(moveStr);
            if (!move.IsValid())
            {
                std::cout << "Invalid move!" << std::endl;
                continue;
            }

            if (!position.IsMoveValid(move))
            {
                std::cout << "Invalid move!" << std::endl;
                continue;
            }

            Position posAfterMove = position;
            if (!posAfterMove.IsMoveLegal(move))
            {
                std::cout << "Illegal move!" << std::endl;
                continue;
            }

            break;
        }

        {
            const bool moveOK = position.DoMove(move);
            ASSERT(moveOK);
        }

        std::cout << position.Print() << std::endl;

        SearchParam searchParam;
        searchParam.debugLog = true;
        searchParam.maxDepth = 8;

        Move bestMove;
        Search::ScoreType score = search.DoSearch(position, bestMove, searchParam);

        if (score <= Search::CheckmateValue)
        {
            std::cout << "Whites win!" << std::endl;
            return;
        }

        if (score >= -Search::CheckmateValue)
        {
            std::cout << "Blacks win!" << std::endl;
            return;
        }

        {
            const bool moveOK = position.DoMove(bestMove);
            ASSERT(moveOK);
        }
    }
}

int main()
{
    InitBitboards(); 
    InitZobristHash();

    //RunTests();
    //RunPerft();
    //RunSearchTests();
    //RunSearchPerfTest();
    //SelfMatch();

   // PlayGame();

    bool uciLoopResult = false;
    {
        UniversalChessInterface uci;
        uciLoopResult = uci.Loop();
    }

    return uciLoopResult ? 0 : 1;
}
