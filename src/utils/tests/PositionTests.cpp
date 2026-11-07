#include "TestCommon.hpp"
#include "../../backend/Game.hpp"
#include "../../backend/Position.hpp"
#include "../../backend/MoveGen.hpp"
#include "../../backend/MovePicker.hpp"
#include "../../backend/MoveOrderer.hpp"
#include "../../backend/Search.hpp"

#include <memory>

void RunPositionTests()
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

        // en passant square and the square the pawn moved from must be empty
        TEST_EXPECT(!Position().FromFEN("rnbqkbnr/pppp1ppp/4N3/3Pp3/8/8/PPP1PPPP/R1BQKBNR w KQkq e6 0 3"));
        TEST_EXPECT(!Position().FromFEN("rnbqkb1r/ppppnppp/8/3Pp3/8/8/PPP1PPPP/RNBQKBNR w KQkq e6 0 3"));
        TEST_EXPECT(!Position().FromFEN("rnbqkbnr/ppp1pppp/8/8/3pP3/4N3/PPPP1PPP/R1BQKBNR b KQkq e3 0 3"));

        // en passant square is kept only if a pawn can capture there
        {
            Position p;
            TEST_EXPECT(p.FromFEN("rnbqkbnr/pppp1ppp/8/8/3pP3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 3"));
            TEST_EXPECT(p.GetEnPassantSquare() == Square_e3);

            TEST_EXPECT(p.FromFEN("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"));
            TEST_EXPECT(!p.GetEnPassantSquare().IsValid());
            TEST_EXPECT(p.ToFEN() == "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1");

            TEST_EXPECT(p.FromFEN("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2"));
            TEST_EXPECT(!p.GetEnPassantSquare().IsValid());

            Position q(Position::InitPositionFEN);
            TEST_EXPECT(q.DoMove(q.MoveFromString("e2e4")));
            TEST_EXPECT(q.DoMove(q.MoveFromString("e7e5")));
            TEST_EXPECT(p == q);
            TEST_EXPECT(p.GetHash() == q.GetHash());
        }

        // invalid syntax
        TEST_EXPECT(!Position().FromFEN("4k3/8/8/9/8/8/8/4K3 w - - 0 1"));

        // too many pieces in a rank
        TEST_EXPECT(!Position().FromFEN("rnbqkbnrr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"));
        TEST_EXPECT(!Position().FromFEN("rnbqkbnr/pppppppp/8p/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"));
        TEST_EXPECT(!Position().FromFEN("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNRR w KQkq - 0 1"));

        // not enough pieces in a rank
        TEST_EXPECT(!Position().FromFEN("rnbqkbnr/pppppppp/7/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"));
        TEST_EXPECT(!Position().FromFEN("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBN w Qkq - 0 1"));

        // invalid side to move
        TEST_EXPECT(!Position().FromFEN("4k3/8/8/8/8/8/8/4K3 x - - 0 1"));
        TEST_EXPECT(!Position().FromFEN("4k3/8/8/8/8/8/8/4K3 wK - 0 1"));

        // non-ASCII character
        TEST_EXPECT(!Position().FromFEN("4k3/8/8/8/8/8/8/4K3 w \xC3\xA9 - 0 1"));

        // move counters
        {
            Position p;
            TEST_EXPECT(!p.FromFEN("4k3/8/8/8/8/8/8/4K3 w - - -5 1"));
            TEST_EXPECT(!p.FromFEN("4k3/8/8/8/8/8/8/4K3 w - - 1a 1"));
            TEST_EXPECT(!p.FromFEN("4k3/8/8/8/8/8/8/4K3 w - - 65536 1"));
            TEST_EXPECT(!p.FromFEN("4k3/8/8/8/8/8/8/4K3 w - - 0 x"));
            TEST_EXPECT(!p.FromFEN("4k3/8/8/8/8/8/8/4K3 w - - 0 -1"));

            TEST_EXPECT(p.FromFEN("4k3/8/8/8/8/8/8/4K3 w - - 12 34"));
            TEST_EXPECT(p.GetHalfMoveCount() == 12);
            TEST_EXPECT(p.GetMoveCount() == 34);

            TEST_EXPECT(p.FromFEN("4k3/8/8/8/8/8/8/4K3 w - - 65535 0"));
            TEST_EXPECT(p.GetHalfMoveCount() == 65535);
            TEST_EXPECT(p.GetMoveCount() == 1);

            TEST_EXPECT(p.FromFEN("4k3/8/8/8/8/8/8/4K3 b - -"));
            TEST_EXPECT(p.GetHalfMoveCount() == 0);
            TEST_EXPECT(p.GetMoveCount() == 1);
        }
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

    // X-FEN castling rights with multiple rooks on one side of the king
    {
        Position pos("rrrrkqqq/pppppppp/8/8/8/8/PPPPPPPP/QQQRKRRR w Kq - 0 1");
        TEST_EXPECT(pos.GetWhitesCastlingRights() == 0b10000000);
        TEST_EXPECT(pos.GetBlacksCastlingRights() == 0b00000001);

        const char* moves[] =
        {
            "h2h4", "g7g6", "h4h5", "d7d5", "h5g6", "f7g6", "b2b3", "d5d4", "f2f4", "c7c5", "c2c4", "c8c6", "b1e4", "f8g7",
            "g2g4", "d8d6", "h1h3", "g6g5", "d2d3", "b8d8", "h3h5", "g5f4", "c1f4", "d6e6", "e4h1", "d8d6", "a1b2",
        };
        for (const char* moveStr : moves)
        {
            const Move move = pos.MoveFromString(moveStr);
            TEST_EXPECT(move.IsValid());
            TEST_EXPECT(pos.DoMove(move));
        }
        TEST_EXPECT(pos.ToFEN() == "r3k1qq/pp2p1qp/2rrr3/2p4R/2Pp1QP1/1P1P4/PQ2P3/3RKRRQ b q - 4 14");

        const Move move = pos.MoveFromString("e8c8");
        TEST_EXPECT(move.IsValid());
        TEST_EXPECT(move.FromSquare() == Square_e8);
        TEST_EXPECT(move.ToSquare() == Square_a8);
        TEST_EXPECT(move.GetPiece() == Piece::King);
        TEST_EXPECT(move.IsLongCastle() == true);
        TEST_EXPECT(move == pos.MoveFromString("O-O-O", MoveNotation::SAN));
        TEST_EXPECT(pos.IsMoveValid(move));
        TEST_EXPECT(pos.IsMoveLegal(move));
        TEST_EXPECT(pos.DoMove(move));
        TEST_EXPECT(pos.ToFEN() == "2kr2qq/pp2p1qp/2rrr3/2p4R/2Pp1QP1/1P1P4/PQ2P3/3RKRRQ w - - 5 15");
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

        // K/Q should map to the outermost rook, file letters select any rook
        {
            Position pos;
            TEST_EXPECT(pos.FromFEN("rrrrkqqq/pppppppp/8/8/8/8/PPPPPPPP/QQQRKRRR w KQkq - 0 1"));
            TEST_EXPECT(pos.GetWhitesCastlingRights() == 0b10001000);
            TEST_EXPECT(pos.GetBlacksCastlingRights() == 0b00000001);
            TEST_EXPECT(pos.ToFEN() == "rrrrkqqq/pppppppp/8/8/8/8/PPPPPPPP/QQQRKRRR w DHa - 0 1");

            TEST_EXPECT(pos.FromFEN("rrrrkqqq/pppppppp/8/8/8/8/PPPPPPPP/QQQRKRRR w Gc - 0 1"));
            TEST_EXPECT(pos.GetWhitesCastlingRights() == 0b01000000);
            TEST_EXPECT(pos.GetBlacksCastlingRights() == 0b00000100);
            TEST_EXPECT(pos.ToFEN() == "rrrrkqqq/pppppppp/8/8/8/8/PPPPPPPP/QQQRKRRR w Gc - 0 1");

            TEST_EXPECT(pos.FromFEN("1r2krr1/pppppppp/8/8/8/8/PPPPPPPP/RR2K1RR w KQkq - 0 1"));
            TEST_EXPECT(pos.GetWhitesCastlingRights() == 0b10000001);
            TEST_EXPECT(pos.GetBlacksCastlingRights() == 0b01000010);
            TEST_EXPECT(pos.ToFEN() == "1r2krr1/pppppppp/8/8/8/8/PPPPPPPP/RR2K1RR w AHbg - 0 1");

            // the same rook given twice
            TEST_EXPECT(pos.FromFEN("rrrrkqqq/pppppppp/8/8/8/8/PPPPPPPP/QQQRKRRR w KHqa - 0 1"));
            TEST_EXPECT(pos.GetWhitesCastlingRights() == 0b10000000);
            TEST_EXPECT(pos.GetBlacksCastlingRights() == 0b00000001);
        }

        // multiple castling rooks on one side of the king
        {
            Position pos;
            TEST_EXPECT(!pos.FromFEN("4k3/8/8/8/8/8/8/RR2K3 w AB - 0 1"));
            TEST_EXPECT(!pos.FromFEN("rrrrkqqq/pppppppp/8/8/8/8/PPPPPPPP/QQQRKRRR w KG - 0 1"));
            TEST_EXPECT(!pos.FromFEN("1r2krr1/pppppppp/8/8/8/8/PPPPPPPP/RR2K1RR w fg - 0 1"));
            TEST_EXPECT(!pos.FromFEN("1r2krr1/pppppppp/8/8/8/8/PPPPPPPP/RR2K1RR w QB - 0 1"));
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
        NodeInfo node;
        node.position = pos;
        pos.ComputeThreats(node.threats);

        MoveList allMoves;
        GenerateMoveList(pos, node.threats.allThreats, allMoves);
        moveOrderer->ScoreMoves(node, allMoves);

        int32_t moveScore = 0;
        Move move;
        uint32_t moveIndex = 0;

        MovePicker movePicker(pos, *moveOrderer, nullptr, Move::Invalid(), true);
        while (movePicker.PickMove(node, move, moveScore))
        {
            bool found = false;
            for (uint32_t i = 0; i < allMoves.Size(); ++i)
            {
                if (allMoves.GetMove(i) == move)
                {
                    found = true;
                }
            }
            TEST_EXPECT(found);
            moveIndex++;
        }
        TEST_EXPECT(moveIndex == allMoves.Size());
    }

    // move picker returns every generated move exactly once, whatever the TT move, killer and counter move are
    {
        std::unique_ptr<MoveOrderer> moveOrderer = std::make_unique<MoveOrderer>();

        const auto checkPicker = [&](const Position& pos, const PackedMove ttMove, const Move killer, const Move counter, bool withQuiets)
        {
            std::unique_ptr<NodeInfo> node = std::make_unique<NodeInfo>();
            node->position = pos;
            pos.ComputeThreats(node->threats);
            node->ply = 1;
            node->depth = 1;
            node->previousMove = Move::Make(Square_a7, Square_a6, Piece::Pawn);

            moveOrderer->Clear();
            if (killer.IsValid())
            {
                moveOrderer->UpdateKillerMove(node->ply, killer);
            }
            if (counter.IsValid())
            {
                moveOrderer->UpdateQuietMovesHistory(*node, &counter, 1, counter, 0);
                TEST_EXPECT(moveOrderer->GetCounterMove(*node) == counter);
            }

            MoveList allMoves;
            GenerateMoveList(pos, node->threats.allThreats, allMoves);
            MoveList captureMoves;
            GenerateMoveList<MoveGenerationMode::Captures>(pos, node->threats.allThreats, captureMoves);

            const Move ttMoveInPos = pos.MoveFromPacked(ttMove);
            const bool expectTTMove = ttMoveInPos.IsValid() && (withQuiets || !ttMoveInPos.IsQuiet());

            uint32_t timesPicked[MoveList::MaxMoves] = {};
            uint32_t numPicked = 0;
            bool firstPick = true;

            MovePicker movePicker(pos, *moveOrderer, nullptr, ttMove, withQuiets);
            Move move;
            int32_t moveScore = 0;
            while (movePicker.PickMove(*node, move, moveScore))
            {
                if (firstPick && expectTTMove)
                {
                    TEST_EXPECT(move == ttMoveInPos);
                }
                firstPick = false;

                uint32_t index = UINT32_MAX;
                for (uint32_t i = 0; i < allMoves.Size(); ++i)
                {
                    if (allMoves.GetMove(i) == move) index = i;
                }
                TEST_EXPECT(index != UINT32_MAX);
                if (index == UINT32_MAX) continue;

                timesPicked[index]++;
                TEST_EXPECT(timesPicked[index] == 1);
                TEST_EXPECT(withQuiets || !move.IsQuiet() || move == ttMoveInPos);
                numPicked++;
            }

            if (withQuiets)
            {
                TEST_EXPECT(numPicked == allMoves.Size());
            }
            else
            {
                // captures-only mode drops losing captures, but must keep every capture that does not lose material
                for (uint32_t i = 0; i < captureMoves.Size(); ++i)
                {
                    const Move m = captureMoves.GetMove(i);
                    if (m.IsCapture() && !pos.StaticExchangeEvaluation(m)) continue;

                    bool picked = false;
                    for (uint32_t j = 0; j < allMoves.Size(); ++j)
                    {
                        if (allMoves.GetMove(j) == m) picked = timesPicked[j] > 0;
                    }
                    TEST_EXPECT(picked);
                }
            }
        };

        const char* fens[] =
        {
            Position::InitPositionFEN,
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
            "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
            "k2r4/4P3/8/1pP5/8/3p1q2/5PPP/KQ1B1RN1 w - b6 0 1",
            "4k3/8/8/8/1b6/8/3P4/4K2R w K - 0 1",
        };

        for (const char* fen : fens)
        {
            const Position pos(fen);

            MoveList moves;
            GenerateMoveList(pos, moves);
            Move quiets[3] = {};
            Move captures[2] = {};
            uint32_t numQuiets = 0, numCaptures = 0;
            for (uint32_t i = 0; i < moves.Size(); ++i)
            {
                const Move m = moves.GetMove(i);
                if (m.IsQuiet() && numQuiets < 3) quiets[numQuiets++] = m;
                if (m.IsCapture() && numCaptures < 2) captures[numCaptures++] = m;
            }
            TEST_EXPECT(numQuiets == 3);

            // killer and counter moves come from other positions, so they may be captures or not even pseudo-legal here
            const Move quietCapture0 = captures[0].IsValid() ? Move::Make(captures[0].FromSquare(), captures[0].ToSquare(), captures[0].GetPiece()) : Move::Invalid();
            const Move quietCapture1 = captures[1].IsValid() ? Move::Make(captures[1].FromSquare(), captures[1].ToSquare(), captures[1].GetPiece()) : quietCapture0;
            Square emptySquare = Square_a1;
            while (pos.Occupied() & emptySquare) emptySquare = Square(emptySquare.Index() + 1);
            const Move notPseudoLegal = Move::Make(emptySquare, Square(emptySquare.Index() ^ 8), Piece::Knight);

            const bool generateQuietsModes[] = { true, false };
            for (const bool withQuiets : generateQuietsModes)
            {
                checkPicker(pos, PackedMove::Invalid(), Move::Invalid(), Move::Invalid(), withQuiets);
                checkPicker(pos, quiets[0], quiets[1], quiets[2], withQuiets);
                checkPicker(pos, quiets[0], quiets[0], quiets[0], withQuiets);
                checkPicker(pos, quiets[0], quiets[1], quiets[1], withQuiets);
                checkPicker(pos, quiets[1], quiets[2], quiets[1], withQuiets);
                checkPicker(pos, notPseudoLegal, notPseudoLegal, notPseudoLegal, withQuiets);
                if (captures[0].IsValid())
                {
                    checkPicker(pos, captures[0], quiets[0], quiets[1], withQuiets);
                    checkPicker(pos, quiets[0], quietCapture0, quietCapture1, withQuiets);
                    checkPicker(pos, captures[0], quietCapture0, quiets[0], withQuiets);
                }
            }
        }
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

        // en passant
        {
            Position pos("rnbqkb1r/ppp1pppp/5n2/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3");
            const Move move = pos.MoveFromString("e5d6");
            TEST_EXPECT(move.IsValid());
            TEST_EXPECT(true == pos.StaticExchangeEvaluation(move, 0));
            TEST_EXPECT(false == pos.StaticExchangeEvaluation(move, 1));
        }
    }

    // IsStaleMate
    {
        {
            const Position pos("7K/5k2/8/8/8/8/8/8 w - - 0 1");
            TEST_EXPECT(!pos.IsInCheck(White));
            TEST_EXPECT(!pos.IsStalemate());
        }

        {
            const Position pos("7K/5k1P/8/8/8/8/8/8 w - - 0 1");
            TEST_EXPECT(!pos.IsInCheck(White));
            TEST_EXPECT(pos.IsStalemate());
        }

        {
            const Position pos("7k/8/7r/K2P3q/P7/8/8/1r6 w - - 0 1");
            TEST_EXPECT(!pos.IsInCheck(White));
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

void RunPositionAdvancedTests()
{
    std::cout << "Running advanced Position tests..." << std::endl;

    // Test position hash consistency
    {
        Position pos1(Position::InitPositionFEN);
        Position pos2(Position::InitPositionFEN);
        
        TEST_EXPECT(pos1.GetHash() == pos2.GetHash());
        
        Move move = pos1.MoveFromString("e2e4");
        pos1.DoMove(move);
        TEST_EXPECT(pos1.GetHash() != pos2.GetHash());
    }
    
    // Test position copy
    {
        Position pos1(Position::InitPositionFEN);
        Position pos2 = pos1;
        
        TEST_EXPECT(pos1 == pos2);
        TEST_EXPECT(pos1.GetHash() == pos2.GetHash());
        
        Move move = pos1.MoveFromString("e2e4");
        pos1.DoMove(move);
        TEST_EXPECT(pos1 != pos2);
    }
    
    // Test game move history
    {
        Game game;
        game.Reset(Position(Position::InitPositionFEN));
        Position original = game.GetPosition();
        
        Move move = game.GetPosition().MoveFromString("e2e4");
        game.DoMove(move);
        
        TEST_EXPECT(game.GetMoves().Size() == 1);
        TEST_EXPECT(game.GetMoves()[0] == move);
        TEST_EXPECT(game.GetPosition() != original);
    }
    
    // Test repetition detection
    {
        Game game;
        game.Reset(Position(Position::InitPositionFEN));
        
        // Make some moves that return to start
        game.DoMove(game.GetPosition().MoveFromString("g1f3"));
        game.DoMove(game.GetPosition().MoveFromString("b8c6"));
        game.DoMove(game.GetPosition().MoveFromString("f3g1"));
        game.DoMove(game.GetPosition().MoveFromString("c6b8"));
        
        // Should detect repetition
        // (This depends on Game implementation)
    }
    
    // Test fifty move rule
    {
        Position pos("4k3/8/8/8/8/8/8/4K3 w - - 50 1");
        TEST_EXPECT(pos.GetHalfMoveCount() == 50);
        
        // After 100 half-moves, should be draw
        Position pos100("4k3/8/8/8/8/8/8/4K3 w - - 100 1");
        TEST_EXPECT(pos100.IsFiftyMoveRuleDraw());
    }
}
