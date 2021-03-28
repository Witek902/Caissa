#include <iostream>
#include "Position.hpp"
#include "Move.hpp"

#define TEST_EXPECT(x) \
    if (!(x)) { std::cout << "Test failed: " << #x << std::endl; __debugbreak();}


uint64_t Perft(const Position& position, uint32_t depth, bool print = true)
{
    if (print)
    {
        std::cout << "Running Perft... depth=" << depth << std::endl;
    }

    MoveList moveList;
    position.GenerateMoveList(moveList);

    uint64_t nodes = 0;
    for (uint32_t i = 0; i < moveList.Size(); i++)
    {
        const Move& move = moveList.GetMove(i);

        Position child = position;
        if (!child.DoMove(move))
        {
            continue;
        }

        uint64_t numChildNodes = depth == 1 ? 1 : Perft(child, depth - 1, false);

        if (print)
        {
            std::cout << position.MoveToString(move) << ": " << numChildNodes << std::endl;
        }

        nodes += numChildNodes;
    }

    if (print)
    {
        std::cout << "Total nodes: " << nodes << std::endl;
    }

    return nodes;
}

void RunTests()
{
    const char* initPositionFEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

    // empty board
    TEST_EXPECT(!Position().IsValid());

    // FEN parsing
    {
        // initial position
        TEST_EXPECT(Position().FromFEN(initPositionFEN));

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
        Position pos(initPositionFEN);
        TEST_EXPECT(pos.ToFEN() == initPositionFEN);
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
        Position pos(initPositionFEN);
        MoveList moveList; pos.GenerateMoveList(moveList);
        TEST_EXPECT(moveList.Size() == 20u);
    }

    // moves parsing & execution
    {
        // move (invalid)
        {
            Position pos(initPositionFEN);
            const Move move = pos.MoveFromString("e3e4");
            TEST_EXPECT(move.IsValid());
            TEST_EXPECT(!pos.IsMoveValid(move));
        }

        // move pawn (invalid)
        {
            Position pos(initPositionFEN);
            const Move move = pos.MoveFromString("e2e2");
            TEST_EXPECT(move.IsValid());
            TEST_EXPECT(!pos.IsMoveValid(move));
        }

        // move pawn (invalid)
        {
            Position pos(initPositionFEN);
            const Move move = pos.MoveFromString("e2f3");
            TEST_EXPECT(move.IsValid());
            TEST_EXPECT(!pos.IsMoveValid(move));
        }

        // move pawn (valid)
        {
            Position pos(initPositionFEN);
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
        TEST_EXPECT(Perft(pos, 1) == 18u);
    }

    {
        const Position pos("rnbqkbnr/1ppppppp/p7/8/8/3P4/PPP1PPPP/RNBQKBNR w KQkq - 0 1");
        TEST_EXPECT(Perft(pos, 2) == 511u);
    }

    {
        const Position pos("rnbqkbnr/pppppppp/8/8/8/3P4/PPP1PPPP/RNBQKBNR b KQkq - 0 1");
        TEST_EXPECT(Perft(pos, 3) == 11959u);
    }

    {
        const Position pos("rnb1kbnr/pp1ppppp/1qp5/1P6/8/8/P1PPPPPP/RNBQKBNR w KQkq - 0 1");
        TEST_EXPECT(Perft(pos, 1) == 21u);
    }

    {
        const Position pos("rnbqkbnr/pp1ppppp/2p5/1P6/8/8/P1PPPPPP/RNBQKBNR b KQkq - 0 1");
        TEST_EXPECT(Perft(pos, 2) == 458u);
    }

    {
        const Position pos("rnbqkbnr/pp1ppppp/2p5/8/1P6/8/P1PPPPPP/RNBQKBNR w KQkq - 0 1");
        TEST_EXPECT(Perft(pos, 3) == 10257u);
    }

    {
        const Position pos("rnbqkbnr/pppppppp/8/8/1P6/8/P1PPPPPP/RNBQKBNR b KQkq - 0 1");
        TEST_EXPECT(Perft(pos, 4) == 216145u);
    }

    // Perft
    {
        // initial position
        {
            const Position pos(initPositionFEN);
            TEST_EXPECT(Perft(pos, 1) == 20u);
            TEST_EXPECT(Perft(pos, 2) == 400u);
            TEST_EXPECT(Perft(pos, 3) == 8902u);
            TEST_EXPECT(Perft(pos, 4) == 197281u);
            TEST_EXPECT(Perft(pos, 5) == 4865609u);
            //TEST_EXPECT(Perft(pos, 6) == 119060324u);
        }

        // kings only
        {
            const Position pos("2k2K2/8/8/8/8/8/8/8 w - - 0 1");
            TEST_EXPECT(Perft(pos, 4) == 848u);
            TEST_EXPECT(Perft(pos, 6) == 29724u);
        }

        // kings + knight vs. king
        {
            const Position pos("2k2K2/5N2/8/8/8/8/8/8 w - - 0 1");
            TEST_EXPECT(Perft(pos, 2) == 41u);
            TEST_EXPECT(Perft(pos, 4) == 2293u);
            TEST_EXPECT(Perft(pos, 6) == 130360u);
        }

        // kings + rook vs. king
        {
            const Position pos("2k2K2/5R2/8/8/8/8/8/8 w - - 0 1");
            TEST_EXPECT(Perft(pos, 1) == 17u);
            TEST_EXPECT(Perft(pos, 2) == 53u);
            TEST_EXPECT(Perft(pos, 4) == 3917u);
            TEST_EXPECT(Perft(pos, 6) == 338276u);
        }

        // kings + bishop vs. king
        {
            const Position pos("2k2K2/5B2/8/8/8/8/8/8 w - - 0 1");
            TEST_EXPECT(Perft(pos, 2) == 58u);
            TEST_EXPECT(Perft(pos, 4) == 4269u);
            TEST_EXPECT(Perft(pos, 6) == 314405u);
        }

        // kings + pawn vs. king
        {
            const Position pos("2k3K1/4P3/8/8/8/8/8/8 w - - 0 1");
            TEST_EXPECT(Perft(pos, 2) == 33u);
            TEST_EXPECT(Perft(pos, 4) == 2007u);
            TEST_EXPECT(Perft(pos, 6) == 136531u);
        }

        // castlings
        {
            const Position pos("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1");
            TEST_EXPECT(Perft(pos, 1) == 26u);
            TEST_EXPECT(Perft(pos, 2) == 568u);
            //TEST_EXPECT(Perft(pos, 4) == 314346u);
        }

        // kings + 2 queens
        {
            const Position pos("q3k2q/8/8/8/8/8/8/Q3K2Q w - - 0 1");
            TEST_EXPECT(Perft(pos, 2) == 1040u);
            TEST_EXPECT(Perft(pos, 4) == 979543u);
            //TEST_EXPECT(Perft(pos, 6) == 923005707u);
        }

        // max moves
        {
            const Position pos("R6R/3Q4/1Q4Q1/4Q3/2Q4Q/Q4Q2/pp1Q4/kBNN1KB1 w - - 0 1");
            TEST_EXPECT(Perft(pos, 1) == 218u);
        }

        // discovered double check via en passant
        {
            const Position pos("8/6p1/7k/7P/5B1R/8/8/7K b - - 0 1");
            TEST_EXPECT(Perft(pos, 1) == 2u);
            TEST_EXPECT(Perft(pos, 2) == 35u);
            TEST_EXPECT(Perft(pos, 3) == 134u);
        }

        // Kiwipete
        {
            const Position pos("r3k2r/p1ppqpb1/1n2pnp1/3PN3/1p2P3/2N2Q1p/PPPB1PPP/R2BKb1R w KQkq - 0 1");
            TEST_EXPECT(Perft(pos, 1) == 40u);
        }

        // Kiwipete
        {
            const Position pos("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPB1PPP/R2BK2R b KQkq - 0 1");
            TEST_EXPECT(Perft(pos, 1) == 44u);
            TEST_EXPECT(Perft(pos, 2) == 1733u);
        }

        // Position 2 - Kiwipete
        {
            const Position pos("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1");
            TEST_EXPECT(Perft(pos, 1) == 48u);
            TEST_EXPECT(Perft(pos, 2) == 2039u);
            TEST_EXPECT(Perft(pos, 3) == 97862u);
            TEST_EXPECT(Perft(pos, 4) == 4085603u);
        }

        // Position 3
        {
            const Position pos("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1");
            TEST_EXPECT(Perft(pos, 1) == 14u);
            TEST_EXPECT(Perft(pos, 2) == 191u);
            TEST_EXPECT(Perft(pos, 3) == 2812u);
            TEST_EXPECT(Perft(pos, 4) == 43238u);
            //TEST_EXPECT(Perft(pos, 5) == 674624u);
        }

        // Position 4
        {
            const Position pos("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1");
            TEST_EXPECT(Perft(pos, 1) == 6u);
            TEST_EXPECT(Perft(pos, 2) == 264u);
            TEST_EXPECT(Perft(pos, 3) == 9467u);
            TEST_EXPECT(Perft(pos, 4) == 422333u);
            //TEST_EXPECT(Perft(pos, 5) == 15833292u);
        }

        // Position 5
        {
            const Position pos("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8");
            TEST_EXPECT(Perft(pos, 1) == 44u);
            TEST_EXPECT(Perft(pos, 2) == 1486u);
            TEST_EXPECT(Perft(pos, 3) == 62379u);
            TEST_EXPECT(Perft(pos, 4) == 2103487u);
            //TEST_EXPECT(Perft(pos, 5) == 89941194u);
        }

        // Position 6
        {
            const Position pos("r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10");
            TEST_EXPECT(Perft(pos, 1) == 46u);
            TEST_EXPECT(Perft(pos, 2) == 2079u);
            TEST_EXPECT(Perft(pos, 3) == 89890u);
            TEST_EXPECT(Perft(pos, 4) == 3894594u);
            //TEST_EXPECT(Perft(pos, 5) == 164075551u);
        }
    }
}

int main()
{
    InitBitboards();

    RunTests();

    Position position("rnbqkbnr/pppppppp/8/P2R4/8/2QB1B2/PPP1PPPP/1N2K1NR w kq - 0 1");

    std::cout << position.Print();

    MoveList moveList;
    position.GenerateMoveList(moveList);

    for (uint32_t i = 0; i < moveList.Size(); ++i)
    {
        std::cout << position.MoveToString(moveList.GetMove(i)) << " ";
    }

    return 0;
}
