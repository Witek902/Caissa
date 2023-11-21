#include "ThreadPool.hpp"

#include "../backend/Position.hpp"
#include "../backend/MoveList.hpp"
#include "../backend/MoveGen.hpp"
#include "../backend/Search.hpp"
#include "../backend/TranspositionTable.hpp"
#include "../backend/Evaluate.hpp"
#include "../backend/Tablebase.hpp"
#include "../backend/Game.hpp"
#include "../backend/Material.hpp"
#include "../backend/Pawns.hpp"
#include "../backend/MovePicker.hpp"
#include "../backend/MoveOrderer.hpp"
#include "../backend/Waitable.hpp"

#include <iostream>
#include <chrono>
#include <mutex>
#include <fstream>
#include <sstream>
#include <iterator>
#include <algorithm>
#include <iomanip>

using namespace threadpool;

#define TEST_EXPECT(x) \
    if (!(x)) { std::cout << "Test failed: " << #x << std::endl; DEBUG_BREAK(); }

extern void RunGameTests();
extern void RunPackedPositionTests();

static void RunBitboardTests()
{
    // attacks on empty board
    for (uint32_t sq = 0; sq < 64; ++sq)
    {
        const Square square(sq);
        TEST_EXPECT(Bitboard::GenerateRookAttacks(square, 0) == (Bitboard::GetRookAttacks(square) & ~square.GetBitboard()));
        TEST_EXPECT(Bitboard::GenerateBishopAttacks(square, 0) == (Bitboard::GetBishopAttacks(square) & ~square.GetBitboard()));
    }

    // "GetBetween"
    {
        TEST_EXPECT(Bitboard::GetBetween(Square_f3, Square_b6) == 0);
        TEST_EXPECT(Bitboard::GetBetween(Square_a1, Square_a1) == 0);
        TEST_EXPECT(Bitboard::GetBetween(Square_a1, Square_a2) == 0);
        TEST_EXPECT(Bitboard::GetBetween(Square_a2, Square_a1) == 0);
        TEST_EXPECT(Bitboard::GetBetween(Square_a1, Square_b2) == 0);
        TEST_EXPECT(Bitboard::GetBetween(Square_a1, Square_a3) == Square(Square_a2).GetBitboard());
        TEST_EXPECT(Bitboard::GetBetween(Square_a3, Square_a1) == Square(Square_a2).GetBitboard());
        TEST_EXPECT(Bitboard::GetBetween(Square_f3, Square_f6) == (Square(Square_f4).GetBitboard() | Square(Square_f5).GetBitboard()));
        TEST_EXPECT(Bitboard::GetBetween(Square_f6, Square_f3) == (Square(Square_f4).GetBitboard() | Square(Square_f5).GetBitboard()));
        TEST_EXPECT(Bitboard::GetBetween(Square_c2, Square_f2) == (Square(Square_d2).GetBitboard() | Square(Square_e2).GetBitboard()));
        TEST_EXPECT(Bitboard::GetBetween(Square_f2, Square_c2) == (Square(Square_d2).GetBitboard() | Square(Square_e2).GetBitboard()));
        TEST_EXPECT(Bitboard::GetBetween(Square_b2, Square_e5) == (Square(Square_c3).GetBitboard() | Square(Square_d4).GetBitboard()));
        TEST_EXPECT(Bitboard::GetBetween(Square_e5, Square_b2) == (Square(Square_c3).GetBitboard() | Square(Square_d4).GetBitboard()));
    }
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

        // missing side to move
        TEST_EXPECT(!Position().FromFEN("r3k3/8/8/8/8/8/8/R3K2R "));

        // some random position
        TEST_EXPECT(Position().FromFEN("4r1rk/1p5q/4Rb2/2pQ1P2/7p/5B2/P4P1B/7K b - - 4 39"));

        // not enough kings
        TEST_EXPECT(!Position().FromFEN("k7/8/8/8/8/8/8/8 w - - 0 1"));
        TEST_EXPECT(!Position().FromFEN("K7/8/8/8/8/8/8/8 w - - 0 1"));
        TEST_EXPECT(!Position().FromFEN("rnbq1bnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQ1BNR w HAha - 0 1"));

        // too many kings
        TEST_EXPECT(!Position().FromFEN("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNKQKBNR w HAkq - 0 1"));
        TEST_EXPECT(!Position().FromFEN("rnkqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQha - 0 1"));

        // black pawn at invalid position
        {
            Position pos;
            TEST_EXPECT(pos.FromFEN("rnbqkbpr/ppppppnp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"));
            TEST_EXPECT(pos.IsValid(false));
            TEST_EXPECT(!pos.IsValid(true));
        }

        // white pawn at invalid position
        {
            Position pos;
            TEST_EXPECT(pos.FromFEN("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPNP/RNBQKBPR w KQkq - 0 1"));
            TEST_EXPECT(pos.IsValid(false));
            TEST_EXPECT(!pos.IsValid(true));
        }

        // opponent side can't be in check
        TEST_EXPECT(!Position().FromFEN("k6Q/8/8/8/8/8/8/K7 w - - 0 1"));
        TEST_EXPECT(!Position().FromFEN("8/8/2Q3k1/8/8/8/2K3q1/8 w - - 0 1"));

        // valid en passant square
        {
            Position p;
            TEST_EXPECT(p.FromFEN("rnbqkbnr/1pp1pppp/p7/3pP3/8/8/PPPP1PPP/RNBQKBNR w Qkq d6 0 3"));
            TEST_EXPECT(p.GetEnPassantSquare() == Square_d6);
        }

        // invalid en passant square
        TEST_EXPECT(!Position().FromFEN("rnbqkbnr/1pp1pppp/p7/3pP3/8/8/PPPP1PPP/RNBQKBNR w Qkq e6 0 3"));

        // invalid syntax
        TEST_EXPECT(!Position().FromFEN("4k3/8/8/9/8/8/8/4K3 w - - 0 1"));
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

    // equality
    {
        TEST_EXPECT(Position("rn1qkb1r/pp2pppp/5n2/3p1b2/3P4/1QN1P3/PP3PPP/R1B1KBNR b KQkq - 0 1") == Position("rn1qkb1r/pp2pppp/5n2/3p1b2/3P4/1QN1P3/PP3PPP/R1B1KBNR b KQkq - 0 1"));
        TEST_EXPECT(Position("rn1qkb1r/pp2pppp/5n2/3p1b2/3P4/1QN1P3/PP3PPP/R1B1KBNR b KQkq - 0 1") != Position("rn1qkb1r/pp2pppp/5n2/3p1b2/3P4/PQN1P3/1P3PPP/R1B1KBNR b KQkq - 0 1"));
    }

    // mirror / flipping
    {
        TEST_EXPECT(Position("rn1qkb1r/pp2pppp/5n2/3p1b2/3P4/1QN1P3/PP3PPP/R1B1KBNR b KQkq - 0 1").MirroredHorizontally() == Position("r1bkq1nr/pppp2pp/2n5/2b1p3/4P3/3P1NQ1/PPP3PP/RNBK1B1R b AHah - 0 1"));
        TEST_EXPECT(Position("rn1qkb1r/pp2pppp/5n2/3p1b2/3P4/1QN1P3/PP3PPP/R1B1KBNR b KQkq - 0 1").MirroredVertically() == Position("R1B1KBNR/PP3PPP/1QN1P3/3P4/3p1b2/5n2/pp2pppp/rn1qkb1r b AHah - 0 1"));
    }

    // king moves
    {
        // king moves (a1)
        {
            Position pos("k7/8/8/8/8/8/8/K7 w - - 0 1");
            MoveList moveList; GenerateMoveList(pos, moveList);
            TEST_EXPECT(moveList.Size() == 3u);
        }

        // king moves (h1)
        {
            Position pos("k7/8/8/8/8/8/8/7K w - - 0 1");
            MoveList moveList; GenerateMoveList(pos, moveList);
            TEST_EXPECT(moveList.Size() == 3u);
        }

        // king moves (h8)
        {
            Position pos("k6K/8/8/8/8/8/8/8 w - - 0 1");
            MoveList moveList; GenerateMoveList(pos, moveList);
            TEST_EXPECT(moveList.Size() == 3u);
        }

        // king moves (a1)
        {
            Position pos("K7/8/8/8/8/8/8/k7 w - - 0 1");
            MoveList moveList; GenerateMoveList(pos, moveList);
            TEST_EXPECT(moveList.Size() == 3u);
        }

        // king moves (b1)
        {
            Position pos("k7/8/8/8/8/8/8/1K6 w - - 0 1");
            MoveList moveList; GenerateMoveList(pos, moveList);
            TEST_EXPECT(moveList.Size() == 5u);
        }

        // king moves (h2)
        {
            Position pos("k7/8/8/8/8/8/7K/8 w - - 0 1");
            MoveList moveList; GenerateMoveList(pos, moveList);
            TEST_EXPECT(moveList.Size() == 5u);
        }

        // king moves (g8)
        {
            Position pos("k5K1/8/8/8/8/8/8/8 w - - 0 1");
            MoveList moveList; GenerateMoveList(pos, moveList);
            TEST_EXPECT(moveList.Size() == 5u);
        }

        // king moves (a7)
        {
            Position pos("8/K7/8/8/8/8/8/7k w - - 0 1");
            MoveList moveList; GenerateMoveList(pos, moveList);
            TEST_EXPECT(moveList.Size() == 5u);
        }

        // king moves (d5)
        {
            Position pos("8/8/8/3K4/8/8/8/7k w - - 0 1");
            MoveList moveList; GenerateMoveList(pos, moveList);
            TEST_EXPECT(moveList.Size() == 8u);
        }

        // castling
        {
            Position pos("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1");
            MoveList moveList; GenerateMoveList(pos, moveList);
            TEST_EXPECT(moveList.Size() == 25u);
        }

        // castling
        {
            Position pos("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RN2K2R w KQkq - 0 1");
            MoveList moveList; GenerateMoveList(pos, moveList);
            TEST_EXPECT(moveList.Size() == 23u);
        }

        // castling
        {
            Position pos("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w Kkq - 0 1");
            MoveList moveList; GenerateMoveList(pos, moveList);
            TEST_EXPECT(moveList.Size() == 24u);
        }

        // castling
        {
            Position pos("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w Qkq - 0 1");
            MoveList moveList; GenerateMoveList(pos, moveList);
            TEST_EXPECT(moveList.Size() == 24u);
        }

        // castling
        {
            Position pos("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w kq - 0 1");
            MoveList moveList; GenerateMoveList(pos, moveList);
            TEST_EXPECT(moveList.Size() == 23u);
        }
    }

    // white pawn moves
    {
        const uint32_t kingMoves = 3u;

        // 2rd rank
        {
            Position pos("k7/8/8/8/8/8/4P3/K7 w - - 0 1");
            MoveList moveList; GenerateMoveList(pos, moveList);
            TEST_EXPECT(moveList.Size() - kingMoves == 2u);
        }

        // 3rd rank
        {
            Position pos("k7/8/8/8/8/4P3/8/K7 w - - 0 1");
            MoveList moveList; GenerateMoveList(pos, moveList);
            TEST_EXPECT(moveList.Size() - kingMoves == 1u);
        }

        // 2rd rank blocked
        {
            Position pos("k7/8/8/8/8/4p3/4P3/K7 w - - 0 1");
            MoveList moveList; GenerateMoveList(pos, moveList);
            TEST_EXPECT(moveList.Size() - kingMoves == 0u);
        }

        // 3rd rank blocked
        {
            Position pos("k7/8/8/8/4p3/4P3/8/K7 w - - 0 1");
            MoveList moveList; GenerateMoveList(pos, moveList);
            TEST_EXPECT(moveList.Size() - kingMoves == 0u);
        }

        // simple capture
        {
            Position pos("k7/8/8/3p4/4P3/8/8/K7 w - - 0 1");
            MoveList moveList; GenerateMoveList(pos, moveList);
            TEST_EXPECT(moveList.Size() - kingMoves == 2u);
        }

        // two captures
        {
            Position pos("k7/8/8/3p1p2/4P3/8/8/K7 w - - 0 1");
            MoveList moveList; GenerateMoveList(pos, moveList);
            TEST_EXPECT(moveList.Size() - kingMoves == 3u);
        }

        // two captures and block
        {
            Position pos("k7/8/8/3ppp2/4P3/8/8/K7 w - - 0 1");
            MoveList moveList; GenerateMoveList(pos, moveList);
            TEST_EXPECT(moveList.Size() - kingMoves == 2u);
        }

        // promotion
        {
            Position pos("k7/4P3/8/8/8/8/8/K7 w - - 0 1");
            MoveList moveList; GenerateMoveList(pos, moveList);
            TEST_EXPECT(moveList.Size() - kingMoves == 4u);
        }

        // blocked promotion
        {
            Position pos("k3n3/4P3/8/8/8/8/8/K7 w - - 0 1");
            MoveList moveList; GenerateMoveList(pos, moveList);
            TEST_EXPECT(moveList.Size() - kingMoves == 0u);
        }

        // 3 promotions possible
        {
            Position pos("k3n1n1/5P2/8/8/8/8/8/K7 w - - 0 1");
            MoveList moveList; GenerateMoveList(pos, moveList);
            TEST_EXPECT(moveList.Size() - kingMoves == 12u);
        }
    }

    // black pawn moves
    {
        const uint32_t kingMoves = 3u;

        // simple capture
        {
            Position pos("k7/8/8/2Rp4/2P5/8/8/K7 b - - 0 1");
            MoveList moveList; GenerateMoveList(pos, moveList);
            TEST_EXPECT(moveList.Size() - kingMoves == 2u);
        }

        // promotion
        {
            Position pos("k7/8/8/8/8/8/4p3/K7 b - - 0 1");
            MoveList moveList; GenerateMoveList(pos, moveList);
            TEST_EXPECT(moveList.Size() - kingMoves == 4u);
        }
    }

    // moves from starting position
    {
        Position pos(Position::InitPositionFEN);
        MoveList moveList; GenerateMoveList(pos, moveList);
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

        // move pawn (invalid)
        {
            Position pos(Position::InitPositionFEN);
            TEST_EXPECT(!pos.MoveFromString("e2", MoveNotation::SAN).IsValid());
            TEST_EXPECT(!pos.MoveFromString("e5", MoveNotation::SAN).IsValid());
            TEST_EXPECT(!pos.MoveFromString("e6", MoveNotation::SAN).IsValid());
            TEST_EXPECT(!pos.MoveFromString("e7", MoveNotation::SAN).IsValid());
            TEST_EXPECT(!pos.MoveFromString("e8", MoveNotation::SAN).IsValid());
        }

        // move pawn (valid)
        {
            Position pos(Position::InitPositionFEN);
            const Move move = pos.MoveFromString("e2e4");
            TEST_EXPECT(move.IsValid());
            TEST_EXPECT(move == pos.MoveFromString("e4", MoveNotation::SAN));
            TEST_EXPECT(move.FromSquare() == Square_e2);
            TEST_EXPECT(move.ToSquare() == Square_e4);
            TEST_EXPECT(move.GetPiece() == Piece::Pawn);
            TEST_EXPECT(move.IsCapture() == false);
            TEST_EXPECT(move.GetPromoteTo() == Piece::None);
            TEST_EXPECT(pos.IsMoveValid(move));
            TEST_EXPECT(pos.IsMoveLegal(move));
            TEST_EXPECT(pos.DoMove(move));
            TEST_EXPECT(pos.ToFEN() == "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1");
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
            TEST_EXPECT(move == pos.MoveFromString("cxb5", MoveNotation::SAN));
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
            TEST_EXPECT(move == pos.MoveFromString("dxc6", MoveNotation::SAN));
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

        // move pawn (valid queen promotion)
        {
            Position pos("1k6/5P2/8/8/8/8/8/4K3 w - - 0 1");
            const Move move = pos.MoveFromString("f7f8q");
            TEST_EXPECT(move.IsValid());
            TEST_EXPECT(move == pos.MoveFromString("f8=Q", MoveNotation::SAN));
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

        // move pawn (valid knight promotion)
        {
            Position pos("1k6/5P2/8/8/8/8/8/4K3 w - - 0 1");
            const Move move = pos.MoveFromString("f7f8n");
            TEST_EXPECT(move.IsValid());
            TEST_EXPECT(move == pos.MoveFromString("f8=N", MoveNotation::SAN));
            TEST_EXPECT(move.FromSquare() == Square_f7);
            TEST_EXPECT(move.ToSquare() == Square_f8);
            TEST_EXPECT(move.GetPiece() == Piece::Pawn);
            TEST_EXPECT(move.IsCapture() == false);
            TEST_EXPECT(move.GetPromoteTo() == Piece::Knight);
            TEST_EXPECT(pos.IsMoveValid(move));
            TEST_EXPECT(pos.IsMoveLegal(move));
            TEST_EXPECT(pos.DoMove(move));
            TEST_EXPECT(pos.ToFEN() == "1k3N2/8/8/8/8/8/8/4K3 b - - 0 1");
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
            TEST_EXPECT(move.ToSquare() == Square_h1);
            TEST_EXPECT(move.GetPiece() == Piece::King);
            TEST_EXPECT(move.IsCapture() == false);
            TEST_EXPECT(move.IsShortCastle() == true);
            TEST_EXPECT(move == pos.MoveFromString("O-O", MoveNotation::SAN));
            TEST_EXPECT(move == pos.MoveFromString("e1h1"));
            TEST_EXPECT(pos.IsMoveValid(move));
            TEST_EXPECT(pos.IsMoveLegal(move));
            TEST_EXPECT(pos.DoMove(move));
            TEST_EXPECT(pos.ToFEN() == "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQ1RK1 b kq - 1 1");
        }

        // castling, whites, king side, no rights
        {
            Position pos("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQK2R w Qkq - 0 1");
            const Move move = pos.MoveFromString("e1g1");
            TEST_EXPECT(!move.IsValid());
        }

        // castling, whites, queen side
        {
            Position pos("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/R3KBNR w KQkq - 0 1");
            const Move move = pos.MoveFromString("e1c1");
            TEST_EXPECT(move.IsValid());
            TEST_EXPECT(move.FromSquare() == Square_e1);
            TEST_EXPECT(move.ToSquare() == Square_a1);
            TEST_EXPECT(move.GetPiece() == Piece::King);
            TEST_EXPECT(move.IsCapture() == false);
            TEST_EXPECT(move.IsLongCastle() == true);
            TEST_EXPECT(move == pos.MoveFromString("O-O-O", MoveNotation::SAN));
            TEST_EXPECT(move == pos.MoveFromString("e1a1"));
            TEST_EXPECT(pos.IsMoveValid(move));
            TEST_EXPECT(pos.IsMoveLegal(move));
            TEST_EXPECT(pos.DoMove(move));
            TEST_EXPECT(pos.ToFEN() == "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/2KR1BNR b kq - 1 1");
        }

        // castling, whites, queen side, no rights
        {
            Position pos("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/R3KBNR w Kkq - 0 1");
            const Move move = pos.MoveFromString("e1c1");
            TEST_EXPECT(!move.IsValid());
        }

        // castling, blacks, king side
        {
            Position pos("rnbqk2r/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1");
            const Move move = pos.MoveFromString("e8g8");
            TEST_EXPECT(move.IsValid());
            TEST_EXPECT(move.FromSquare() == Square_e8);
            TEST_EXPECT(move.ToSquare() == Square_h8);
            TEST_EXPECT(move.GetPiece() == Piece::King);
            TEST_EXPECT(move.IsCapture() == false);
            TEST_EXPECT(move.IsShortCastle() == true);
            TEST_EXPECT(move == pos.MoveFromString("O-O", MoveNotation::SAN));
            TEST_EXPECT(move == pos.MoveFromString("e8h8"));
            TEST_EXPECT(pos.IsMoveValid(move));
            TEST_EXPECT(pos.IsMoveLegal(move));
            TEST_EXPECT(pos.DoMove(move));
            TEST_EXPECT(pos.ToFEN() == "rnbq1rk1/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQ - 1 2");
        }

        // castling, blacks, king side, no rights
        {
            Position pos("rnbqk2r/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQq - 0 1");
            const Move move = pos.MoveFromString("e8g8");
            TEST_EXPECT(!move.IsValid());
        }

        // castling, blacks, queen side
        {
            Position pos("r3kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1");
            const Move move = pos.MoveFromString("e8c8");
            TEST_EXPECT(move.IsValid());
            TEST_EXPECT(move.FromSquare() == Square_e8);
            TEST_EXPECT(move.ToSquare() == Square_a8);
            TEST_EXPECT(move.GetPiece() == Piece::King);
            TEST_EXPECT(move.IsCapture() == false);
            TEST_EXPECT(move.IsLongCastle() == true);
            TEST_EXPECT(move == pos.MoveFromString("O-O-O", MoveNotation::SAN));
            TEST_EXPECT(move == pos.MoveFromString("e8a8"));
            TEST_EXPECT(pos.IsMoveValid(move));
            TEST_EXPECT(pos.IsMoveLegal(move));
            TEST_EXPECT(pos.DoMove(move));
            TEST_EXPECT(pos.ToFEN() == "2kr1bnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQ - 1 2");
        }

        // castling, blacks, queen side, no rights
        {
            Position pos("r3kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQk - 0 1");
            const Move move = pos.MoveFromString("e8c8");
            TEST_EXPECT(!move.IsValid());
        }

        // illegal castling, whites, king side, king in check
        {
            Position pos("4k3/4r3/8/8/8/8/8/R3K2R w KQ - 0 1");
            const Move move = pos.MoveFromString("e1g1");
            TEST_EXPECT(!move.IsValid());
        }

        // illegal castling, whites, king side, king crossing check
        {
            Position pos("4kr2/8/8/8/8/8/8/R3K2R w KQ - 0 1");
            const Move move = pos.MoveFromString("e1g1");
            TEST_EXPECT(!move.IsValid());
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
            TEST_EXPECT(!move.IsValid());
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

    // castling through pawn attacks
    {
        {
            Position pos("r3k2r/2P5/8/8/8/8/2p5/R3K2R b KQkq - 0 1");
            TEST_EXPECT(pos.IsMoveLegal(pos.MoveFromString("e8g8")));
            TEST_EXPECT(!pos.MoveFromString("e8c8").IsValid());
        }

        {
            Position pos("r3k2r/2P5/8/8/8/8/2p5/R3K2R w KQkq - 0 1");
            TEST_EXPECT(pos.IsMoveLegal(pos.MoveFromString("e1g1")));
            TEST_EXPECT(!pos.MoveFromString("e1c1").IsValid());
        }
    }

    // Chess960 tests
    {
        Position::s_enableChess960 = true;

        // K/Q should map to A/H
        {
            Position posA, posB, posC;
            TEST_EXPECT(posA.FromFEN("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w AHah - 0 1"));
            TEST_EXPECT(posB.FromFEN("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"));
            TEST_EXPECT(posC.FromFEN("r3k1r1/pppppppp/8/8/8/8/PPPPPPPP/R3K1R1 w AGag - 0 1"));
            TEST_EXPECT(posA == posB);
            TEST_EXPECT(posA.GetHash() != posC.GetHash());
        }

        // parsing/printing
        {
            Position pos;
            TEST_EXPECT(pos.FromFEN("rkr5/pppppppp/8/8/8/8/PPPPPPPP/RKR5 w ACac - 0 1"));
            TEST_EXPECT(pos.GetWhitesCastlingRights() == 0b00000101);
            TEST_EXPECT(pos.GetBlacksCastlingRights() == 0b00000101);
            TEST_EXPECT(pos.ToFEN() == "rkr5/pppppppp/8/8/8/8/PPPPPPPP/RKR5 w ACac - 0 1");
        }

        // parsing incorrect castling rights
        {
            Position pos;
            TEST_EXPECT(pos.FromFEN("rkr5/pppppppp/8/8/8/8/PPPPPPPP/RKR5 w BDbd - 0 1"));
            TEST_EXPECT(pos.GetWhitesCastlingRights() == 0);
            TEST_EXPECT(pos.GetBlacksCastlingRights() == 0);
            TEST_EXPECT(pos.ToFEN() == "rkr5/pppppppp/8/8/8/8/PPPPPPPP/RKR5 w - - 0 1");
        }

        // parsing incorrect castling rights
        {
            Position pos;
            TEST_EXPECT(pos.FromFEN("rkr5/pppppppp/8/8/8/8/PPPPPPPP/RKR5 w BDbd - 0 1"));
            TEST_EXPECT(pos.GetWhitesCastlingRights() == 0);
            TEST_EXPECT(pos.GetBlacksCastlingRights() == 0);
            TEST_EXPECT(pos.ToFEN() == "rkr5/pppppppp/8/8/8/8/PPPPPPPP/RKR5 w - - 0 1");
        }

        {
            Position pos;
            TEST_EXPECT(pos.FromFEN("rk2r3/8/8/8/8/8/8/RK2R3 w KQkq - 0 1"));
            {
                const Move move = pos.MoveFromString("b1a1");
                TEST_EXPECT(move.IsValid());
                TEST_EXPECT(move.FromSquare() == Square_b1);
                TEST_EXPECT(move.ToSquare() == Square_a1);
                TEST_EXPECT(move.GetPiece() == Piece::King);
                TEST_EXPECT(move.IsCapture() == false);
                TEST_EXPECT(move.IsLongCastle() == true);
                TEST_EXPECT(move.IsShortCastle() == false);
                TEST_EXPECT(pos.IsMoveValid(move));
                TEST_EXPECT(pos.IsMoveLegal(move));
                TEST_EXPECT(pos.DoMove(move));
                TEST_EXPECT(pos.ToFEN() == "rk2r3/8/8/8/8/8/8/2KRR3 b ae - 1 1");
            }
        }

        // can't long castle because target square is blocked
        {
            Position pos;

            TEST_EXPECT(pos.FromFEN("5rkr/pppppppp/8/8/8/8/PPPPPPPP/5RKR w KQkq - 0 1"));
            TEST_EXPECT(!pos.MoveFromString("g1h1").IsValid());

            TEST_EXPECT(pos.FromFEN("5rkr/pppppppp/8/8/8/8/PPPPPPPP/5RKR b KQkq - 0 1"));
            TEST_EXPECT(!pos.MoveFromString("g8h8").IsValid());
        }

        // "very" long castle
        {
            Position pos;
            TEST_EXPECT(pos.FromFEN("rkr4n/pppppppp/8/8/8/8/PPPPPPPP/RKR4N w ACac - 0 1"));

            Move move = pos.MoveFromString("b1c1");
            TEST_EXPECT(move.IsValid());
            TEST_EXPECT(move.FromSquare() == Square_b1);
            TEST_EXPECT(move.ToSquare() == Square_c1);
            TEST_EXPECT(move.GetPiece() == Piece::King);
            TEST_EXPECT(move.IsCapture() == false);
            TEST_EXPECT(move.IsLongCastle() == false);
            TEST_EXPECT(move.IsShortCastle() == true);
            TEST_EXPECT(pos.IsMoveValid(move));
            TEST_EXPECT(pos.IsMoveLegal(move));
            TEST_EXPECT(pos.MoveToString(move, MoveNotation::LAN) == "b1c1");
            TEST_EXPECT(pos.MoveToString(move, MoveNotation::SAN) == "O-O");
            TEST_EXPECT(pos.DoMove(move));
            TEST_EXPECT(pos.ToFEN() == "rkr4n/pppppppp/8/8/8/8/PPPPPPPP/R4RKN b ac - 1 1");

            move = pos.MoveFromString("b8c8");
            TEST_EXPECT(move.IsValid());
            TEST_EXPECT(move.FromSquare() == Square_b8);
            TEST_EXPECT(move.ToSquare() == Square_c8);
            TEST_EXPECT(move.GetPiece() == Piece::King);
            TEST_EXPECT(move.IsCapture() == false);
            TEST_EXPECT(move.IsLongCastle() == false);
            TEST_EXPECT(move.IsShortCastle() == true);
            TEST_EXPECT(pos.IsMoveValid(move));
            TEST_EXPECT(pos.IsMoveLegal(move));
            TEST_EXPECT(pos.MoveToString(move, MoveNotation::LAN) == "b8c8");
            TEST_EXPECT(pos.MoveToString(move, MoveNotation::SAN) == "O-O");
            TEST_EXPECT(pos.DoMove(move));
            TEST_EXPECT(pos.ToFEN() == "r4rkn/pppppppp/8/8/8/8/PPPPPPPP/R4RKN w - - 2 2");
        }

        // various 960 castlings
        {
            Position pos; Move move;

            TEST_EXPECT(pos.FromFEN("rk5r/pppppppp/8/8/8/8/PPPPPPPP/RK5R w KQkq - 0 1"));
            move = pos.MoveFromString("b1a1");
            TEST_EXPECT(move.IsValid());
            TEST_EXPECT(pos.DoMove(move));
            TEST_EXPECT(pos.ToFEN() == "rk5r/pppppppp/8/8/8/8/PPPPPPPP/2KR3R b ah - 1 1");
            move = pos.MoveFromString("b8a8");
            TEST_EXPECT(move.IsValid());
            TEST_EXPECT(pos.DoMove(move));
            TEST_EXPECT(pos.ToFEN() == "2kr3r/pppppppp/8/8/8/8/PPPPPPPP/2KR3R w - - 2 2");

            TEST_EXPECT(pos.FromFEN("rk5r/pppppppp/8/8/8/8/PPPPPPPP/RK5R w KQkq - 0 1"));
            move = pos.MoveFromString("b1h1");
            TEST_EXPECT(move.IsValid());
            TEST_EXPECT(pos.DoMove(move));
            TEST_EXPECT(pos.ToFEN() == "rk5r/pppppppp/8/8/8/8/PPPPPPPP/R4RK1 b ah - 1 1");
            move = pos.MoveFromString("b8h8");
            TEST_EXPECT(move.IsValid());
            TEST_EXPECT(pos.DoMove(move));
            TEST_EXPECT(pos.ToFEN() == "r4rk1/pppppppp/8/8/8/8/PPPPPPPP/R4RK1 w - - 2 2");

            TEST_EXPECT(pos.FromFEN("1rk3r1/pppppppp/8/8/8/8/PPPPPPPP/1RK3R1 w KQkq - 0 1"));
            move = pos.MoveFromString("c1b1");
            TEST_EXPECT(move.IsValid());
            TEST_EXPECT(pos.DoMove(move));
            TEST_EXPECT(pos.ToFEN() == "1rk3r1/pppppppp/8/8/8/8/PPPPPPPP/2KR2R1 b bg - 1 1");
            move = pos.MoveFromString("c8b8");
            TEST_EXPECT(move.IsValid());
            TEST_EXPECT(pos.DoMove(move));
            TEST_EXPECT(pos.ToFEN() == "2kr2r1/pppppppp/8/8/8/8/PPPPPPPP/2KR2R1 w - - 2 2");

            TEST_EXPECT(pos.FromFEN("1rk3r1/pppppppp/8/8/8/8/PPPPPPPP/1RK3R1 w KQkq - 0 1"));
            move = pos.MoveFromString("c1g1");
            TEST_EXPECT(move.IsValid());
            TEST_EXPECT(pos.DoMove(move));
            TEST_EXPECT(pos.ToFEN() == "1rk3r1/pppppppp/8/8/8/8/PPPPPPPP/1R3RK1 b bg - 1 1");
            move = pos.MoveFromString("c8g8");
            TEST_EXPECT(move.IsValid());
            TEST_EXPECT(pos.DoMove(move));
            TEST_EXPECT(pos.ToFEN() == "1r3rk1/pppppppp/8/8/8/8/PPPPPPPP/1R3RK1 w - - 2 2");

            TEST_EXPECT(pos.FromFEN("5rkr/pppppppp/8/8/8/8/PPPPPPPP/5RKR w KQkq - 0 1"));
            move = pos.MoveFromString("g1f1");
            TEST_EXPECT(move.IsValid());
            TEST_EXPECT(pos.DoMove(move));
            TEST_EXPECT(pos.ToFEN() == "5rkr/pppppppp/8/8/8/8/PPPPPPPP/2KR3R b fh - 1 1");
            move = pos.MoveFromString("g8f8");
            TEST_EXPECT(move.IsValid());
            TEST_EXPECT(pos.DoMove(move));
            TEST_EXPECT(pos.ToFEN() == "2kr3r/pppppppp/8/8/8/8/PPPPPPPP/2KR3R w - - 2 2");

            TEST_EXPECT(pos.FromFEN("rk4b1/p1bpqp2/1ppn1p1r/6pp/1PP1P2P/PNBB1P2/3P2P1/4QRKR b Ha - 0 12"));
            move = pos.MoveFromString("O-O-O", MoveNotation::SAN);
            TEST_EXPECT(move.IsValid());
            TEST_EXPECT(pos.DoMove(move));
            TEST_EXPECT(pos.ToFEN() == "2kr2b1/p1bpqp2/1ppn1p1r/6pp/1PP1P2P/PNBB1P2/3P2P1/4QRKR w H - 1 13");
        }

        Position::s_enableChess960 = false;
    }

    // Position::MoveFromPacked
    {
        const Position pos("k7/4P3/8/1pP5/8/3p1q2/5PPP/KQ1B1RN1 w - b6 0 1");

        TEST_EXPECT(Move::Make(Square_h2, Square_h4, Piece::Pawn) == pos.MoveFromPacked(PackedMove(Square_h2, Square_h4)));
        TEST_EXPECT(Move::Make(Square_b1, Square_d3, Piece::Queen, Piece::None, true) == pos.MoveFromPacked(PackedMove(Square_b1, Square_d3)));
    }

    // Position::IsCapture
    {
        const Position pos("k7/4P3/8/1pP5/8/3p1q2/5PPP/KQ1B1RN1 w - b6 0 1");

        TEST_EXPECT(pos.IsCapture(PackedMove(Square_d1, Square_f3)));
        TEST_EXPECT(!pos.IsCapture(PackedMove(Square_g1, Square_e2)));
        TEST_EXPECT(!pos.IsCapture(PackedMove(Square_f3, Square_d1)));
        TEST_EXPECT(!pos.IsCapture(PackedMove(Square_f3, Square_f4)));
    }

    // Move picker
    {
        std::unique_ptr<MoveOrderer> moveOrderer = std::make_unique<MoveOrderer>();

        //const Position pos("r2q1rk1/1Q2npp1/p1p1b2p/b2p4/2nP4/2N1PNP1/PP1B1PBP/R4RK1 w - - 0 17");
        //const Position pos("r2q1rk1/1Q2npp1/p1p1b2p/b2p4/2nP3P/2N1PNP1/PP1B1PB1/R4RK1 b - - 0 17");
        const Position pos("k2r4/4P3/8/1pP5/8/3p1q2/5PPP/KQ1B1RN1 w - b6 0 1");
        const NodeInfo node{ pos };

        MoveList allMoves;
        GenerateMoveList<MoveGenerationMode::Captures>(pos, allMoves);
        GenerateMoveList<MoveGenerationMode::Quiets>(pos, allMoves);
        moveOrderer->ScoreMoves(node, Game(), allMoves);

        int32_t moveScore = 0;
        Move move;
        uint32_t moveIndex = 0;

        MovePicker movePicker(pos, *moveOrderer, nullptr, Move::Invalid(), true);
        while (movePicker.PickMove(node, Game(), move, moveScore))
        {
            bool found = false;
            for (uint32_t i = 0; i < allMoves.Size(); ++i)
            {
                if (allMoves.GetMove(i) == move)
                {
                    ASSERT(allMoves.GetScore(i) == moveScore);
                    found = true;
                }
            }
            TEST_EXPECT(found);
            moveIndex++;
        }
        TEST_EXPECT(moveIndex == allMoves.Size());
    }

    // Standard Algebraic Notation tests
    {
        // promote to queen and check
        {
            Position pos("2Q5/6p1/6kp/8/3K4/6P1/p7/1rR5 b - - 0 51");
            const Move move = pos.MoveFromString("a2a1q");
            TEST_EXPECT(move.IsValid());
            TEST_EXPECT(move.FromSquare() == Square_a2);
            TEST_EXPECT(move.ToSquare() == Square_a1);
            TEST_EXPECT(move.GetPromoteTo() == Piece::Queen);
            TEST_EXPECT(move.GetPiece() == Piece::Pawn);
            TEST_EXPECT(move.IsPromotion());
            TEST_EXPECT(pos.MoveToString(move) == "a1=Q+");
        }
        // bishop takes pawn
        {
            Position pos("rnbqkbnr/p1pppppp/8/1p6/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1");
            const Move move = pos.MoveFromString("f1b5");
            TEST_EXPECT(move.IsValid());
            TEST_EXPECT(pos.MoveToString(move) == "Bxb5");
        }
        // 2 rooks, ambiguous piece
        {
            Position pos("2r1kr2/8/8/8/3R4/8/1K6/7R w - - 0 1");
            const Move move = pos.MoveFromString("d4h4");
            TEST_EXPECT(move.IsValid());
            TEST_EXPECT(pos.MoveToString(move) == "Rdh4");
        }
        // 2 rooks, ambiguous piece
        {
            Position pos("2r1kr2/8/8/8/3R4/8/1K6/7R w - - 0 1");
            const Move move = pos.MoveFromString("h1h4");
            TEST_EXPECT(move.IsValid());
            TEST_EXPECT(pos.MoveToString(move) == "Rhh4");
        }
        // 2 rooks, ambiguous piece, but one rook is pinned
        {
            Position pos("3k4/8/3r3r/8/8/8/8/2KQ4 b - - 0 1");
            const Move move = pos.MoveFromString("h6f6");
            TEST_EXPECT(move.IsValid());
            TEST_EXPECT(pos.MoveToString(move) == "Rf6");
        }
        // 2 rooks, ambiguous file
        {
            Position pos("3r3r/4k3/8/8/3R4/8/1K6/7R b - - 0 1");
            const Move move = pos.MoveFromString("d8f8");
            TEST_EXPECT(move.IsValid());
            TEST_EXPECT(pos.MoveToString(move) == "Rdf8");
        }
        // 2 rooks ambiguous rank
        {
            Position pos("3r3r/1K3k2/8/R7/4Q2Q/8/8/R6Q w - - 0 1");
            const Move move = pos.MoveFromString("a1a3");
            TEST_EXPECT(move.IsValid());
            TEST_EXPECT(pos.MoveToString(move) == "R1a3");
        }
        // 3 queens, ambiguous both file and rank
        {
            Position pos("3r3r/1K3k2/8/R7/4Q2Q/8/8/R6Q w - - 0 1");
            const Move move = pos.MoveFromString("h4e1");
            TEST_EXPECT(move.IsValid());
            TEST_EXPECT(pos.MoveToString(move) == "Qh4e1");
        }
        // pawn push
        {
            Position pos(Position::InitPositionFEN);
            const Move move = pos.MoveFromString("d2d4");
            TEST_EXPECT(move.IsValid());
            TEST_EXPECT(move.GetPiece() == Piece::Pawn);
            TEST_EXPECT(pos.MoveToString(move) == "d4");
        }
        // pawn capture
        {
            Position pos("rnbqkbnr/pppp1ppp/8/4p3/3P1P2/8/PPP1P1PP/RNBQKBNR b KQkq - 0 2");
            const Move move = pos.MoveFromString("e5f4");
            TEST_EXPECT(move.IsValid());
            TEST_EXPECT(move.GetPiece() == Piece::Pawn);
            TEST_EXPECT(pos.MoveToString(move) == "exf4");
        }
        // en passant
        {
            Position pos("rnbqkbnr/ppp2ppp/3p4/3Pp3/8/8/PPP1PPPP/RNBQKBNR w KQkq e6 0 3");
            const Move move = pos.MoveFromString("d5e6");
            TEST_EXPECT(move.IsValid());
            TEST_EXPECT(move.GetPiece() == Piece::Pawn);
            TEST_EXPECT(pos.MoveToString(move) == "dxe6");
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

        // promotion
        {
            Position pos("k7/5P2/8/8/8/8/8/K7 w - - 0 1");
            const Move move = pos.MoveFromString("f7f8q");
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

        // 2 rooks battery + bishop
        {
            Position pos("K2R4/3R4/6b1/8/8/3r3r/8/7k w - - 0 1");
            const Move move = pos.MoveFromString("d7d3");
            TEST_EXPECT(move.IsValid());
            TEST_EXPECT(true == pos.StaticExchangeEvaluation(move, 0));
            TEST_EXPECT(false == pos.StaticExchangeEvaluation(move, 1));
        }

        // 3 rooks battery
        {
            Position pos("K2R4/3R4/3R4/8/8/3r2rr/8/7k w - - 0 1");
            const Move move = pos.MoveFromString("d7d3");
            TEST_EXPECT(move.IsValid());
            TEST_EXPECT(true == pos.StaticExchangeEvaluation(move, 500));
            TEST_EXPECT(false == pos.StaticExchangeEvaluation(move, 501));
        }

        // complex
        {
            Position pos("6k1/1pp4p/p1pb4/6q1/3P1pRr/2P4P/PP1Br1P1/5RKN w - - 0 1");
            const Move move = pos.MoveFromString("f1f4");
            TEST_EXPECT(move.IsValid());
            TEST_EXPECT(true == pos.StaticExchangeEvaluation(move, -100));
            TEST_EXPECT(false == pos.StaticExchangeEvaluation(move, -99));
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

        // pawn push (losing)
        {
            Position pos("k7/8/8/5p2/8/6P1/8/K7 w - - 0 1");
            const Move move = pos.MoveFromString("g3g4");
            TEST_EXPECT(move.IsValid());
            TEST_EXPECT(true == pos.StaticExchangeEvaluation(move, -100));
            TEST_EXPECT(false == pos.StaticExchangeEvaluation(move, -99));
        }

        // pawn push (equal)
        {
            Position pos("k7/8/8/5p2/8/6PP/8/K7 w - - 0 1");
            const Move move = pos.MoveFromString("g3g4");
            TEST_EXPECT(move.IsValid());
            TEST_EXPECT(true == pos.StaticExchangeEvaluation(move, 0));
            TEST_EXPECT(false == pos.StaticExchangeEvaluation(move, 1));
        }

        // pawn push (equal)
        {
            Position pos("r2q1rk1/1Q2npp1/p1p1b2p/b2p4/2nP4/4PNP1/PP1B1PBP/RN3RK1 b - - 1 17");
            const Move move = pos.MoveFromString("c4a3");
            TEST_EXPECT(move.IsValid());
            TEST_EXPECT(true == pos.StaticExchangeEvaluation(move, -300));
            TEST_EXPECT(false == pos.StaticExchangeEvaluation(move, -299));
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

    // IsMate / IsFiftyMoveRuleDraw
    {
        {
            const Position pos("7k/7p/2Q5/8/2Br1PK1/6P1/4P3/5q2 w - - 99 100");
            TEST_EXPECT(!pos.IsMate());
            TEST_EXPECT(!pos.IsFiftyMoveRuleDraw());
        }

        {
            const Position pos("7k/7p/5Q2/8/2Br1PK1/6P1/4P3/5q2 b - - 100 100");
            TEST_EXPECT(pos.IsMate());
            TEST_EXPECT(!pos.IsFiftyMoveRuleDraw());
        }

        {
            const Position pos("5r1k/7p/3Q4/8/2B2PK1/6P1/4P3/5q2 b - - 100 100");
            TEST_EXPECT(!pos.IsMate());
            TEST_EXPECT(pos.IsFiftyMoveRuleDraw());
        }
    }

    // Passed pawns
    {
        const Position pos("k7/5pP1/1P2P3/pP6/P7/3pP3/1P2p1Pp/K7 w - - 0 1");

        TEST_EXPECT(!IsPassedPawn(Square_a4, pos.Whites().pawns, pos.Blacks().pawns));
        TEST_EXPECT(!IsPassedPawn(Square_b2, pos.Whites().pawns, pos.Blacks().pawns));
        TEST_EXPECT(!IsPassedPawn(Square_b5, pos.Whites().pawns, pos.Blacks().pawns));
        TEST_EXPECT(IsPassedPawn(Square_b6, pos.Whites().pawns, pos.Blacks().pawns));
        TEST_EXPECT(!IsPassedPawn(Square_e3, pos.Whites().pawns, pos.Blacks().pawns));
        TEST_EXPECT(!IsPassedPawn(Square_e6, pos.Whites().pawns, pos.Blacks().pawns));
        TEST_EXPECT(!IsPassedPawn(Square_g2, pos.Whites().pawns, pos.Blacks().pawns));
    }

    // GivesCheck
    {
        {
            Position pos("3n4/3n4/pppk2pp/8/5R2/3n4/3n4/3n3K w - - 0 1");
            TEST_EXPECT(true == pos.GivesCheck_Approx(pos.MoveFromString("f4d4")));
            TEST_EXPECT(true == pos.GivesCheck_Approx(pos.MoveFromString("f4f6")));
            TEST_EXPECT(false == pos.GivesCheck_Approx(pos.MoveFromString("f4a4")));
            TEST_EXPECT(false == pos.GivesCheck_Approx(pos.MoveFromString("f4b4")));
            TEST_EXPECT(false == pos.GivesCheck_Approx(pos.MoveFromString("f4c4")));
            TEST_EXPECT(false == pos.GivesCheck_Approx(pos.MoveFromString("f4e4")));
            TEST_EXPECT(false == pos.GivesCheck_Approx(pos.MoveFromString("f4h4")));
        }

        {
            Position pos("5n2/5n2/3R4/8/ppp2kpp/5n2/5n2/5n1K w - - 0 1");
            TEST_EXPECT(true == pos.GivesCheck_Approx(pos.MoveFromString("d6d4")));
            TEST_EXPECT(true == pos.GivesCheck_Approx(pos.MoveFromString("d6f6")));
            TEST_EXPECT(false == pos.GivesCheck_Approx(pos.MoveFromString("d6a6")));
            TEST_EXPECT(false == pos.GivesCheck_Approx(pos.MoveFromString("d6b6")));
            TEST_EXPECT(false == pos.GivesCheck_Approx(pos.MoveFromString("d6c6")));
            TEST_EXPECT(false == pos.GivesCheck_Approx(pos.MoveFromString("d6e6")));
            TEST_EXPECT(false == pos.GivesCheck_Approx(pos.MoveFromString("d6h6")));
        }

        {
            Position pos("8/1R6/6n1/8/8/5bk1/8/7K w - - 0 1");
            TEST_EXPECT(false == pos.GivesCheck_Approx(pos.MoveFromString("b7g7")));
            TEST_EXPECT(false == pos.GivesCheck_Approx(pos.MoveFromString("b7b3")));
        }

        {
            Position pos("8/3ppp2/4k3/8/8/1P5P/4B3/7K w - - 0 1");
            TEST_EXPECT(true == pos.GivesCheck_Approx(pos.MoveFromString("e2g4")));
            TEST_EXPECT(true == pos.GivesCheck_Approx(pos.MoveFromString("e2c4")));
            TEST_EXPECT(false == pos.GivesCheck_Approx(pos.MoveFromString("e2f3")));
            TEST_EXPECT(false == pos.GivesCheck_Approx(pos.MoveFromString("e2h5")));
            TEST_EXPECT(false == pos.GivesCheck_Approx(pos.MoveFromString("e2e2")));
            TEST_EXPECT(false == pos.GivesCheck_Approx(pos.MoveFromString("e2d3")));
            TEST_EXPECT(false == pos.GivesCheck_Approx(pos.MoveFromString("e2b5")));
            TEST_EXPECT(false == pos.GivesCheck_Approx(pos.MoveFromString("e2h4")));
        }

        {
            Position pos("8/3ppp2/4k3/3n1n2/8/1P5P/4B3/7K w - - 0 1");
            TEST_EXPECT(false == pos.GivesCheck_Approx(pos.MoveFromString("e2g4")));
            TEST_EXPECT(false == pos.GivesCheck_Approx(pos.MoveFromString("e2c4")));
            TEST_EXPECT(false == pos.GivesCheck_Approx(pos.MoveFromString("e2f3")));
            TEST_EXPECT(false == pos.GivesCheck_Approx(pos.MoveFromString("e2h5")));
            TEST_EXPECT(false == pos.GivesCheck_Approx(pos.MoveFromString("e2e2")));
            TEST_EXPECT(false == pos.GivesCheck_Approx(pos.MoveFromString("e2d3")));
            TEST_EXPECT(false == pos.GivesCheck_Approx(pos.MoveFromString("e2b5")));
            TEST_EXPECT(false == pos.GivesCheck_Approx(pos.MoveFromString("e2h4")));
        }

        {
            Position pos("8/4k3/6p1/6K1/8/q2b1Q2/8/8 w - - 8 8");
            TEST_EXPECT(true == pos.GivesCheck_Approx(pos.MoveFromString("f3b7")));
            TEST_EXPECT(true == pos.GivesCheck_Approx(pos.MoveFromString("f3e4")));
            TEST_EXPECT(true == pos.GivesCheck_Approx(pos.MoveFromString("f3e2")));
            TEST_EXPECT(true == pos.GivesCheck_Approx(pos.MoveFromString("f3f6")));
            TEST_EXPECT(true == pos.GivesCheck_Approx(pos.MoveFromString("f3f7")));
            TEST_EXPECT(true == pos.GivesCheck_Approx(pos.MoveFromString("f3f8")));
            TEST_EXPECT(true == pos.GivesCheck_Approx(pos.MoveFromString("f3e3")));
            TEST_EXPECT(false == pos.GivesCheck_Approx(pos.MoveFromString("f3d3")));
            TEST_EXPECT(false == pos.GivesCheck_Approx(pos.MoveFromString("f3h1")));
            TEST_EXPECT(false == pos.GivesCheck_Approx(pos.MoveFromString("f3g2")));
            TEST_EXPECT(false == pos.GivesCheck_Approx(pos.MoveFromString("f3g3")));
            TEST_EXPECT(false == pos.GivesCheck_Approx(pos.MoveFromString("f3g4")));
            TEST_EXPECT(false == pos.GivesCheck_Approx(pos.MoveFromString("f3d5")));
            TEST_EXPECT(false == pos.GivesCheck_Approx(pos.MoveFromString("f3d1")));
            TEST_EXPECT(false == pos.GivesCheck_Approx(pos.MoveFromString("f3a8")));
        }
    }
}

static void RunMaterialTests()
{
    {
        MaterialKey key{ 1,0,0,1,0,1,0,0,1,0 };
        TEST_EXPECT(key.IsSymetric());
    }
    {
        MaterialKey key{ 63,63,63,63,63,63,63,63,63,63 };
        TEST_EXPECT(key.IsSymetric());
    }
    {
        MaterialKey key{ 0,0,0,1,0,1,0,0,1,0 };
        TEST_EXPECT(!key.IsSymetric());
    }
    {
        MaterialKey key{ 1,0,0,1,0,0,0,0,1,0 };
        TEST_EXPECT(!key.IsSymetric());
    }
}

static void RunMovesListTests()
{
    // no more space in A
    {
        MovesArray<PackedMove, 3> movesA;
        movesA[0] = PackedMove(Square_a1, Square_b1, Piece::None);
        movesA[1] = PackedMove(Square_a1, Square_b2, Piece::None);
        movesA[2] = PackedMove(Square_a1, Square_b3, Piece::None);

        MovesArray<PackedMove, 3> movesB;
        movesB[0] = PackedMove(Square_a1, Square_b4, Piece::None);
        movesB[1] = PackedMove(Square_a1, Square_b5, Piece::None);
        movesB[2] = PackedMove(Square_a1, Square_b6, Piece::None);

        movesA.MergeWith(movesB);

        TEST_EXPECT(movesA[0] == PackedMove(Square_a1, Square_b1, Piece::None));
        TEST_EXPECT(movesA[1] == PackedMove(Square_a1, Square_b2, Piece::None));
        TEST_EXPECT(movesA[2] == PackedMove(Square_a1, Square_b3, Piece::None));
    }

    // take some from B
    {
        MovesArray<PackedMove, 3> movesA;
        movesA[0] = PackedMove(Square_a1, Square_b1, Piece::None);

        MovesArray<PackedMove, 3> movesB;
        movesB[0] = PackedMove(Square_a1, Square_b4, Piece::None);
        movesB[1] = PackedMove(Square_a1, Square_b5, Piece::None);
        movesB[2] = PackedMove(Square_a1, Square_b6, Piece::None);

        movesA.MergeWith(movesB);

        TEST_EXPECT(movesA[0] == PackedMove(Square_a1, Square_b1, Piece::None));
        TEST_EXPECT(movesA[1] == PackedMove(Square_a1, Square_b4, Piece::None));
        TEST_EXPECT(movesA[2] == PackedMove(Square_a1, Square_b5, Piece::None));
    }

    // take everything from B
    {
        MovesArray<PackedMove, 3> movesA;

        MovesArray<PackedMove, 3> movesB;
        movesB[0] = PackedMove(Square_a1, Square_b4, Piece::None);
        movesB[1] = PackedMove(Square_a1, Square_b5, Piece::None);
        movesB[2] = PackedMove(Square_a1, Square_b6, Piece::None);

        movesA.MergeWith(movesB);

        TEST_EXPECT(movesA[0] == PackedMove(Square_a1, Square_b4, Piece::None));
        TEST_EXPECT(movesA[1] == PackedMove(Square_a1, Square_b5, Piece::None));
        TEST_EXPECT(movesA[2] == PackedMove(Square_a1, Square_b6, Piece::None));
    }

    // mix
    {
        MovesArray<PackedMove, 3> movesA;
        movesA[0] = PackedMove(Square_a1, Square_b1, Piece::None);
        movesA[1] = PackedMove(Square_a1, Square_b2, Piece::None);

        MovesArray<PackedMove, 3> movesB;
        movesB[0] = PackedMove(Square_a1, Square_b1, Piece::None);
        movesB[1] = PackedMove(Square_a1, Square_b5, Piece::None);
        movesB[2] = PackedMove(Square_a1, Square_b2, Piece::None);

        movesA.MergeWith(movesB);

        TEST_EXPECT(movesA[0] == PackedMove(Square_a1, Square_b1, Piece::None));
        TEST_EXPECT(movesA[1] == PackedMove(Square_a1, Square_b2, Piece::None));
        TEST_EXPECT(movesA[2] == PackedMove(Square_a1, Square_b5, Piece::None));
    }
}

static void RunPerftTests()
{
    std::cout << "Running Perft tests..." << std::endl;

    Waitable waitable;
    {
        TaskBuilder taskBuilder(waitable);

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
            TEST_EXPECT(pos.Perft(5) == 4865609u);
            //TEST_EXPECT(pos.Perft(6) == 119060324u);
        });

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
            //TEST_EXPECT(pos.Perft(6) == 923005707u);
        });

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
            //TEST_EXPECT(pos.Perft(5) == 193690690u);
        });

        // Position 3
        taskBuilder.Task("Perft", [](const TaskContext&)
        {
            const Position pos("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1");
            TEST_EXPECT(pos.Perft(1) == 14u);
            TEST_EXPECT(pos.Perft(2) == 191u);
            TEST_EXPECT(pos.Perft(3) == 2812u);
            TEST_EXPECT(pos.Perft(4) == 43238u);
            TEST_EXPECT(pos.Perft(5) == 674624u);
        });

        // Position 4
        taskBuilder.Task("Perft", [](const TaskContext&)
        {
            const Position pos("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1");
            TEST_EXPECT(pos.Perft(1) == 6u);
            TEST_EXPECT(pos.Perft(2) == 264u);
            TEST_EXPECT(pos.Perft(3) == 9467u);
            TEST_EXPECT(pos.Perft(4) == 422333u);
            TEST_EXPECT(pos.Perft(5) == 15833292u);
        });

        // Position 5
        taskBuilder.Task("Perft", [](const TaskContext&)
        {
            const Position pos("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8");
            TEST_EXPECT(pos.Perft(1) == 44u);
            TEST_EXPECT(pos.Perft(2) == 1486u);
            TEST_EXPECT(pos.Perft(3) == 62379u);
            TEST_EXPECT(pos.Perft(4) == 2103487u);
            //TEST_EXPECT(pos.Perft(5) == 89941194u);
        });

        // Position 6
        taskBuilder.Task("Perft", [](const TaskContext&)
        {
            const Position pos("r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10");
            TEST_EXPECT(pos.Perft(1) == 46u);
            TEST_EXPECT(pos.Perft(2) == 2079u);
            TEST_EXPECT(pos.Perft(3) == 89890u);
            TEST_EXPECT(pos.Perft(4) == 3894594u);
            //TEST_EXPECT(pos.Perft(5) == 164075551u);
            //TEST_EXPECT(pos.Perft(6) == 6923051137llu);
            //TEST_EXPECT(pos.Perft(7) == 287188994746llu);
        });

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
    }
    waitable.Wait();
}

static void RunEvalTests()
{
    TEST_EXPECT(Evaluate(Position("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")) > 0);
    TEST_EXPECT(Evaluate(Position("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1")) < 0);
    TEST_EXPECT(Evaluate(Position("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")) == -Evaluate(Position("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1")));
    TEST_EXPECT(Evaluate(Position("r6r/1p3p2/1n1p1kpp/pPpPp1nP/P1P1PqPR/4NP2/3NK2R/Q7 w - - 0 1")) == -Evaluate(Position("q7/3nk2r/4np2/p1p1pQpr/PpPpP1Np/1N1P1KPP/1P3P2/R6R b - - 0 1")));

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
    TEST_EXPECT(KnownWinValue <= Evaluate(Position("K7/B7/B7/8/8/8/8/7k b - - 0 1")));
    TEST_EXPECT(-KnownWinValue >= Evaluate(Position("K7/8/8/8/8/7b/7b/7k w - - 0 1")));
    TEST_EXPECT(-KnownWinValue >= Evaluate(Position("K7/8/8/8/8/7b/7b/7k w - - 0 1")));

    // KvR
    TEST_EXPECT(KnownWinValue <= Evaluate(Position("K7/R7/8/8/8/8/8/7k w - - 0 1")));
    TEST_EXPECT(KnownWinValue <= Evaluate(Position("K7/R7/8/8/8/8/8/7k w - - 0 1")));
    TEST_EXPECT(-KnownWinValue >= Evaluate(Position("K7/8/8/8/8/8/8/6rk w - - 0 1")));
    TEST_EXPECT(-KnownWinValue >= Evaluate(Position("K7/8/8/8/8/8/8/6rk w - - 0 1")));
    TEST_EXPECT(KnownWinValue <= Evaluate(Position("8/8/8/8/8/8/6k1/KRR5 b - - 0 1")));

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
    TEST_EXPECT(0 < Evaluate(Position("8/8/P2k4/8/8/8/7B/7K b - - 0 1")));
    TEST_EXPECT(0 < Evaluate(Position("8/8/P2k4/8/8/P7/7B/7K b - - 0 1")));
    TEST_EXPECT(0 < Evaluate(Position("8/8/4k2P/8/8/8/B7/K7 b - - 0 1")));
    TEST_EXPECT(0 < Evaluate(Position("8/8/4k2P/8/8/8/B6P/K7 b - - 0 1")));
    TEST_EXPECT(0 < Evaluate(Position("1k6/8/8/8/8/8/B6P/K7 w - - 0 1")));
    TEST_EXPECT(0 < Evaluate(Position("6k1/8/8/8/8/8/P6B/7K w - - 0 1")));
    TEST_EXPECT(0 < Evaluate(Position("4k3/8/8/7K/8/8/B6P/8 w - - 0 1")));
    TEST_EXPECT(0 < Evaluate(Position("4k3/8/8/8/7K/8/B6P/8 w - - 0 1")));
    TEST_EXPECT(0 < Evaluate(Position("4k3/8/8/7K/8/8/B6P/8 b - - 0 1")));

    // KBPvK (winning)
    TEST_EXPECT(KnownWinValue <= Evaluate(Position("4k3/8/8/8/8/8/8/2NBK3 w - - 0 1")));
    TEST_EXPECT(KnownWinValue <= Evaluate(Position("4k3/8/8/8/8/8/8/2NBK3 b - - 0 1")));
    TEST_EXPECT(-KnownWinValue >= Evaluate(Position("2nbk3/8/8/8/8/8/8/4K3 b - - 0 1")));
    TEST_EXPECT(-KnownWinValue >= Evaluate(Position("2nbk3/8/8/8/8/8/8/4K3 w - - 0 1")));

    // KNNNvK
    TEST_EXPECT(Evaluate(Position("3k4/8/8/8/8/8/8/2NKNN2 w - - 0 1")) >= KnownWinValue);
    TEST_EXPECT(Evaluate(Position("3k4/8/8/8/8/8/8/2NKNN2 b - - 0 1")) >= KnownWinValue);

    // KBBBvK
    TEST_EXPECT(Evaluate(Position("3k4/8/8/8/8/8/8/2BKBB2 w - - 0 1")) >= KnownWinValue);
    TEST_EXPECT(Evaluate(Position("3k4/8/8/8/8/8/8/2BKBB2 b - - 0 1")) >= KnownWinValue);

    // KPPvK
    TEST_EXPECT(Evaluate(Position("K7/8/8/8/7k/7P/6P1/8 w - - 0 1")) >= KnownWinValue);
    TEST_EXPECT(Evaluate(Position("K7/8/8/3PP3/4k3/8/8/8 w - - 0 1")) >= KnownWinValue);
    TEST_EXPECT(Evaluate(Position("8/8/8/8/8/6P1/5Pk1/K7 b - - 0 1")) >= KnownWinValue);

    // extreme imbalance
    {
        {
            const ScoreType score = Evaluate(Position("QQQQQQpk/QQQQQQpp/QQQQQQQQ/QQQQQQQQ/QQQQQQQQ/QQQQQQQQ/QQQQQQQQ/KQQQQQQQ w - - 0 1"));
            TEST_EXPECT(score > 6000);
            TEST_EXPECT(score < KnownWinValue);
        }
        {
            const ScoreType score = Evaluate(Position("qqqqkqqq/qqqqqqqq/qqqqqqqq/qqqqqqqq/pppppppp/8/PPPPPPPP/4K3 w - - 0 1"));
            TEST_EXPECT(score < -6000);
            TEST_EXPECT(score > -KnownWinValue);
        }
        {
            const ScoreType score = Evaluate(Position("RRRRRRpk/RRRRRRpp/RRRRRRRR/RRRRRRRR/RRRRRRRR/RRRRRRRR/RRRRRRRR/KRRRRRRR w - - 0 1"));
            TEST_EXPECT(score > 4000);
            TEST_EXPECT(score < KnownWinValue);
        }
        {
            const ScoreType score = Evaluate(Position("rrrrkrrr/rrrrrrrr/rrrrrrrr/rrrrrrrr/pppppppp/8/PPPPPPPP/4K3 w - - 0 1"));
            TEST_EXPECT(score < -4000);
            TEST_EXPECT(score > -KnownWinValue);
        }
    }

    // pawns endgame
    TEST_EXPECT(Evaluate(Position("k7/p7/8/8/8/8/PP6/K7 w - - 0 1")) >= 0);
    //TEST_EXPECT(Evaluate(Position("k7/p7/8/8/8/8/PP6/K7 b - - 0 1")) >= 0);
    TEST_EXPECT(Evaluate(Position("k7/p7/8/8/8/8/PPP5/K7 w - - 0 1")) > 0);
    TEST_EXPECT(Evaluate(Position("k7/p7/8/8/8/8/PPP5/K7 b - - 0 1")) > 0);
    TEST_EXPECT(Evaluate(Position("k7/pp6/8/8/8/8/PPP5/K7 w - - 0 1")) >= 0);
    TEST_EXPECT(Evaluate(Position("k7/pp6/8/8/8/8/PPP5/K7 w - - 0 1")) >= 0);
    TEST_EXPECT(Evaluate(Position("k7/p7/8/8/8/8/PPPP4/K7 w - - 0 1")) > 0);
    TEST_EXPECT(Evaluate(Position("k7/p7/8/8/8/8/PPPP4/K7 b - - 0 1")) > 0);

    // queen vs. weaker piece
    TEST_EXPECT(Evaluate(Position("3rk3/8/8/8/8/8/8/2Q1K3 w - - 0 1")) > 0);
    TEST_EXPECT(Evaluate(Position("3rk3/8/8/8/8/8/8/2Q1K3 b - - 0 1")) > 0);
    TEST_EXPECT(Evaluate(Position("3nk3/8/8/8/8/8/8/2Q1K3 w - - 0 1")) > 0);
    TEST_EXPECT(Evaluate(Position("3nk3/8/8/8/8/8/8/2Q1K3 b - - 0 1")) > 0);
    TEST_EXPECT(Evaluate(Position("3bk3/8/8/8/8/8/8/2Q1K3 w - - 0 1")) > 0);
    TEST_EXPECT(Evaluate(Position("3bk3/8/8/8/8/8/8/2Q1K3 b - - 0 1")) > 0);
    TEST_EXPECT(Evaluate(Position("4k3/3p4/8/8/8/8/8/2Q1K3 w - - 0 1")) >= KnownWinValue);
    TEST_EXPECT(Evaluate(Position("4k3/3p4/8/8/8/8/8/2Q1K3 b - - 0 1")) > 0);

    TEST_EXPECT(Evaluate(Position("2Q5/8/8/8/3n4/8/1b6/k2K4 b - - 0 1")) == 0);
    TEST_EXPECT(Evaluate(Position("2Q3b1/6n1/8/8/8/8/3K4/k7 w - - 0 1")) > 0);
}

// this test suite runs full search on well known/easy positions
void RunSearchTests(uint32_t numThreads)
{
    std::cout << "Running Search tests... (numThreads=" << numThreads << ")" << std::endl;

    Search search;
    TranspositionTable tt{ 16 * 1024 * 1024 };
    SearchResult result;
    Game game;

    SearchParam param{ tt };
    param.debugLog = false;
    param.numPvLines = UINT32_MAX;
    param.numThreads = numThreads;

    // insufficient material draw
    {
        param.limits.maxDepth = 4;
        param.numPvLines = UINT32_MAX;

        game.Reset(Position("4k2K/8/8/8/8/8/8/8 w - - 0 1"));
        search.DoSearch(game, param, result);

        TEST_EXPECT(result.size() == 3);
        TEST_EXPECT(std::abs(result[0].score) <= DrawScoreRandomness);
        TEST_EXPECT(std::abs(result[1].score) <= DrawScoreRandomness);
        TEST_EXPECT(std::abs(result[2].score) <= DrawScoreRandomness);
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
        param.limits.maxDepth = 12;
        param.numPvLines = UINT32_MAX;

        game.Reset(Position("k7/7Q/1K6/8/8/8/8/8 w - - 0 1"));
        search.DoSearch(game, param, result);

        TEST_EXPECT(result.size() == 27);
        TEST_EXPECT(result[0].score == CheckmateValue - 1);
        TEST_EXPECT(result[1].score == CheckmateValue - 1);
        TEST_EXPECT(result[2].score == CheckmateValue - 1);
        TEST_EXPECT(result[3].score == CheckmateValue - 1);
    }

    // mate in one
    {
        param.limits.maxDepth = 12;
        param.numPvLines = UINT32_MAX;

        game.Reset(Position("7k/7p/2Q5/8/2Br1PK1/6P1/4P3/5q2 w - - 99 100"));
        search.DoSearch(game, param, result);

        TEST_EXPECT(result.size() == 36);
        TEST_EXPECT(result[0].score == CheckmateValue - 1);
        TEST_EXPECT(result[1].score == 0);
    }

    // mate in two
    {
        param.limits.maxDepth = 40;
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
        TEST_EXPECT(std::abs(result[0].score) <= DrawScoreRandomness);
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
        param.limits.maxDepth = 4;
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

    // search explosion test 1
    {
        param.limits.maxDepth = 1;
        param.numPvLines = 1;

        game.Reset(Position("KNnNnNnk/NnNnNnNn/nNnNnNnN/NnNnNnNn/nNnNnNnN/NnNnNnNn/nNnNnNnN/NnNnNnNn w - - 0 1"));
        search.DoSearch(game, param, result);

        TEST_EXPECT(result.size() == 1);
    }

    // search explosion test 2
    {
        param.limits.maxDepth = 1;
        param.numPvLines = 1;

        game.Reset(Position("qQqqkqqQ/Qqqqqqqq/qQqqqqqQ/QqQqQqQq/qQqQqQqQ/QqQQQQQq/qQQQQQQQ/QqQQKQQq w - - 0 1"));
        search.DoSearch(game, param, result);

        TEST_EXPECT(result.size() == 1);
    }

    // search explosion test 3
    {
        param.limits.maxDepth = 1;
        param.numPvLines = 1;

        game.Reset(Position("q2k2q1/2nqn2b/1n1P1n1b/2rnr2Q/1NQ1QN1Q/3Q3B/2RQR2B/Q2K2Q1 w - - 0 1"));
        search.DoSearch(game, param, result);

        TEST_EXPECT(result.size() == 1);
    }

    // mate in 1 with huge material disadvantage
    {
        param.limits.maxDepth = 5;
        param.numPvLines = 1;

        game.Reset(Position("qqqqqqqq/qkqqqqqq/qqNqqqqq/qqq1qqqq/qqqq1qqq/qqqqq1qq/qqqqqqBn/qqqqqqnK w - - 0 1"));
        search.DoSearch(game, param, result);

        TEST_EXPECT(result.size() == 1);
        TEST_EXPECT(result[0].score == CheckmateValue - 1);
        TEST_EXPECT(result[0].moves.front() == PackedMove(Square_c6, Square_a5) ||
                    result[0].moves.front() == PackedMove(Square_c6, Square_d8));
    }

    // mate in 1, more than 218 moves possible
    {
        param.limits.maxDepth = 8;
        param.numPvLines = 1;

        game.Reset(Position("QQQQQQBk/Q6B/Q6Q/Q6Q/Q6Q/Q6Q/Q6Q/KQQQQQQQ w - - 0 1"));
        search.DoSearch(game, param, result);

        TEST_EXPECT(result.size() == 1);
        TEST_EXPECT(result[0].score == CheckmateValue - 1);
    }

    // mate on 50th move is a draw
    {
        param.limits.maxDepth = 10;
        param.numPvLines = 1;

        game.Reset(Position("8/6B1/8/8/2K2n2/k7/1R6/8 b - - 98 2"));
        search.DoSearch(game, param, result);

        TEST_EXPECT(result.size() == 1);
        TEST_EXPECT(std::abs(result[0].score) <= DrawScoreRandomness);
    }

    ASSERT(param.numThreads == numThreads); // don't modify number of threads!
}

void RunUnitTests()
{
    RunBitboardTests();
    RunPositionTests();
    RunMaterialTests();
    RunMovesListTests();
    RunEvalTests();
    RunPackedPositionTests();
    RunGameTests();
    RunPerftTests();
    RunSearchTests(1); // single-threaded
    RunSearchTests(4); // multi-threaded
}

bool RunPerformanceTests(const std::vector<std::string>& paths)
{
    using MovesListType = std::vector<std::string>;

    struct TestCaseEntry
    {
        std::string positionStr;
        MovesListType bestMoves;
        MovesListType avoidMoves;

        bool operator < (const TestCaseEntry& rhs) const { return positionStr < rhs.positionStr; }
        bool operator == (const TestCaseEntry& rhs) const { return positionStr == rhs.positionStr; }
    };

    enum class ParsingMode
    {
        Position,
        BestMoves,
        AvoidMoves
    };

    std::vector<TestCaseEntry> testVector;
    for (const std::string& path : paths)
    {
        std::ifstream file(path);
        if (!file.good())
        {
            std::cout << "Failed to open testcases file: " << path << std::endl;
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

    // remove duplicates
    {
        const size_t size = testVector.size();
        std::sort(testVector.begin(), testVector.end());
        testVector.erase(std::unique(testVector.begin(), testVector.end()), testVector.end());
        if (testVector.size() != size)
        {
            std::cout << "Found " << (size - testVector.size()) << " duplicate positions" << std::endl;
        }
    }

    std::cout << testVector.size() << " test positions loaded" << std::endl << std::endl;

    std::cout << "MaxNodes; Correct; CorrectRate; Time; Time/Correct" << std::endl;

    bool verbose = false;

    std::vector<Search> searchArray{ std::thread::hardware_concurrency() };

    std::vector<TranspositionTable> ttArray;
    ttArray.resize(std::thread::hardware_concurrency());

    uint32_t maxNodes = 2048;

    for (;;)
    {
        std::mutex mutex;
        std::atomic<uint32_t> success = 0;
        float accumTime = 0.0f;

        Waitable waitable;
        {
            TaskBuilder taskBuilder(waitable);

            for (const TestCaseEntry& testCase : testVector)
            {
                taskBuilder.Task("SearchTest", [testCase, &searchArray, maxNodes, &mutex, verbose, &success, &ttArray, &accumTime](const TaskContext& ctx)
                {
                    Search& search = searchArray[ctx.threadId];
                    search.Clear();

                    TranspositionTable& tt = ttArray[ctx.threadId];
                    if (tt.GetSize() == 0)
                    {
                        tt.Resize(16 * 1024 * 1024);
                    }
                    tt.Clear();

                    const Position position(testCase.positionStr);
                    TEST_EXPECT(position.IsValid());

                    Game game;
                    game.Reset(position);

                    SearchParam searchParam{ tt };
                    searchParam.debugLog = false;
                    searchParam.limits.maxNodes = maxNodes;

                    const TimePoint startTimePoint = TimePoint::GetCurrent();

                    SearchResult searchResult;
                    search.DoSearch(game, searchParam, searchResult);

                    const TimePoint endTimePoint = TimePoint::GetCurrent();

                    {
                        std::unique_lock<std::mutex> lock(mutex);
                        accumTime += (endTimePoint - startTimePoint).ToSeconds();
                    }

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

        const float passRate = !testVector.empty() ? (float)success / (float)testVector.size() : 0.0f;
        const float factor = accumTime / passRate;

        std::cout
            << std::setw(10) << maxNodes << "; "
            << std::setw(4) << success << "; "
            << std::setw(8) << std::setprecision(4) << passRate << "; "
            << std::setw(8) << std::setprecision(4) << accumTime << "; "
            << std::setw(8) << std::setprecision(4) << factor << std::endl;

        maxNodes *= 2;
    }

    return true;
}
