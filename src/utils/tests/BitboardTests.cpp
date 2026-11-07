#include "TestCommon.hpp"
#include "../../backend/Bitboard.hpp"
#include "../../backend/Square.hpp"

void RunBitboardTests()
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

void RunBitboardAdvancedTests()
{
    std::cout << "Running advanced Bitboard tests..." << std::endl;

    // Test all squares for rook attacks
    for (uint32_t sq = 0; sq < 64; ++sq)
    {
        const Square square(sq);
        Bitboard attacks = Bitboard::GetRookAttacks(square);
        TEST_EXPECT(attacks.Count() > 0);
        TEST_EXPECT(attacks.Count() <= 15); // Max rook attacks on empty board
    }
    
    // Test all squares for bishop attacks
    for (uint32_t sq = 0; sq < 64; ++sq)
    {
        const Square square(sq);
        Bitboard attacks = Bitboard::GetBishopAttacks(square);
        TEST_EXPECT(attacks.Count() > 0);
        TEST_EXPECT(attacks.Count() <= 13); // Max bishop attacks on empty board
    }
    
    // Test king attacks
    for (uint32_t sq = 0; sq < 64; ++sq)
    {
        const Square square(sq);
        Bitboard attacks = Bitboard::GetKingAttacks(square);
        uint32_t count = attacks.Count();
        // Corner squares have 3 attacks, edge squares have 5, center squares have 8
        TEST_EXPECT(count >= 3 && count <= 8);
    }
    
    // Test knight attacks
    for (uint32_t sq = 0; sq < 64; ++sq)
    {
        const Square square(sq);
        Bitboard attacks = Bitboard::GetKnightAttacks(square);
        uint32_t count = attacks.Count();
        // Knight attacks range from 2 (corner) to 8 (center)
        TEST_EXPECT(count >= 2 && count <= 8);
    }
    
    // Test pawn attacks
    {
        // White pawn on e4
        Bitboard whitePawnAttacks = Bitboard::GetPawnAttacks(Square_e4, White);
        TEST_EXPECT(whitePawnAttacks.Count() == 2);
        TEST_EXPECT(whitePawnAttacks & Square(Square_d5).GetBitboard());
        TEST_EXPECT(whitePawnAttacks & Square(Square_f5).GetBitboard());
        
        // Black pawn on e5
        Bitboard blackPawnAttacks = Bitboard::GetPawnAttacks(Square_e5, Black);
        TEST_EXPECT(blackPawnAttacks.Count() == 2);
        TEST_EXPECT(blackPawnAttacks & Square(Square_d4).GetBitboard());
        TEST_EXPECT(blackPawnAttacks & Square(Square_f4).GetBitboard());
    }
}

void RunBitOperationTests()
{
    std::cout << "Running Bit operation tests..." << std::endl;

    // Count
    TEST_EXPECT_EQ(Bitboard(0).Count(), 0u);
    TEST_EXPECT_EQ(Bitboard(1).Count(), 1u);
    TEST_EXPECT_EQ(Bitboard(0xFFull).Count(), 8u);
    TEST_EXPECT_EQ(Bitboard::Full().Count(), 64u);
    TEST_EXPECT_EQ(Bitboard::LightSquares().Count(), 32u);
    TEST_EXPECT_EQ(Bitboard::DarkSquares().Count(), 32u);

    // Light and dark squares are complementary
    TEST_EXPECT((Bitboard::LightSquares() & Bitboard::DarkSquares()) == 0);
    TEST_EXPECT((Bitboard::LightSquares() | Bitboard::DarkSquares()) == Bitboard::Full());

    // IsBitSet
    {
        const Bitboard bb = Square(Square_a1).GetBitboard() | Square(Square_h8).GetBitboard();
        TEST_EXPECT(bb.IsBitSet(Square(Square_a1).Index()));
        TEST_EXPECT(!bb.IsBitSet(Square(Square_b1).Index()));
        TEST_EXPECT(!bb.IsBitSet(Square(Square_e4).Index()));
        TEST_EXPECT(bb.IsBitSet(Square(Square_h8).Index()));
    }

    // FileMask
    {
        TEST_EXPECT_EQ(Bitboard(0).FileMask(), 0u);
        TEST_EXPECT_EQ(Square(Square_a1).GetBitboard().FileMask(), 1u);
        TEST_EXPECT_EQ(Square(Square_h1).GetBitboard().FileMask(), 128u);
        TEST_EXPECT_EQ(Bitboard::FileBitboard<0>().FileMask(), 1u);
        TEST_EXPECT_EQ(Bitboard::RankBitboard<0>().FileMask(), 0xFFu);
    }

    // RankBitboard and FileBitboard
    {
        for (uint32_t r = 0; r < 8; ++r)
            TEST_EXPECT_EQ(Bitboard::RankBitboard(r).Count(), 8u);
        for (uint32_t f = 0; f < 8; ++f)
            TEST_EXPECT_EQ(Bitboard::FileBitboard(f).Count(), 8u);
        // A rank and a file intersect at exactly one square
        TEST_EXPECT_EQ((Bitboard::RankBitboard<0>() & Bitboard::FileBitboard<0>()).Count(), 1u);
        // Each square sits on its own rank and file
        for (uint32_t sq = 0; sq < 64; ++sq)
        {
            const Square s(sq);
            TEST_EXPECT(Bitboard::RankBitboard(s.Rank()).IsBitSet(sq));
            TEST_EXPECT(Bitboard::FileBitboard(s.File()).IsBitSet(sq));
        }
    }

    // Iterate preserves LSB-first order
    {
        const Bitboard bb = Square(Square_a1).GetBitboard() | Square(Square_c3).GetBitboard() | Square(Square_h8).GetBitboard();
        uint32_t count = 0;
        uint32_t prev = 0;
        bb.Iterate([&](uint32_t idx)
        {
            if (count > 0) { TEST_EXPECT(idx > prev); }
            prev = idx;
            count++;
        });
        TEST_EXPECT_EQ(count, 3u);
    }

    // Bitwise operators
    {
        const Bitboard a(0xF0F0F0F0F0F0F0F0ull);
        const Bitboard b(0x0F0F0F0F0F0F0F0Full);
        TEST_EXPECT((a & b) == Bitboard(0));
        TEST_EXPECT((a | b) == Bitboard::Full());
        TEST_EXPECT((a ^ b) == Bitboard::Full());
        TEST_EXPECT(~a == b);
    }
}

void RunBitboardTransformTests()
{
    std::cout << "Running Bitboard transform tests..." << std::endl;

    // Shifts: North / South / East / West
    {
        TEST_EXPECT(Square(Square_a1).GetBitboard().North() == Square(Square_a2).GetBitboard());
        TEST_EXPECT(Square(Square_a8).GetBitboard().North() == Bitboard(0));
        TEST_EXPECT(Square(Square_h8).GetBitboard().North() == Bitboard(0));

        TEST_EXPECT(Square(Square_a2).GetBitboard().South() == Square(Square_a1).GetBitboard());
        TEST_EXPECT(Square(Square_a1).GetBitboard().South() == Bitboard(0));

        TEST_EXPECT(Square(Square_a1).GetBitboard().East() == Square(Square_b1).GetBitboard());
        TEST_EXPECT(Square(Square_h1).GetBitboard().East() == Bitboard(0));
        TEST_EXPECT(Square(Square_h4).GetBitboard().East() == Bitboard(0));

        TEST_EXPECT(Square(Square_b1).GetBitboard().West() == Square(Square_a1).GetBitboard());
        TEST_EXPECT(Square(Square_a1).GetBitboard().West() == Bitboard(0));
        TEST_EXPECT(Square(Square_a5).GetBitboard().West() == Bitboard(0));
    }

    // Diagonal shifts via Shift<Direction>
    {
        TEST_EXPECT(Square(Square_a1).GetBitboard().Shift<Direction::NorthEast>() == Square(Square_b2).GetBitboard());
        TEST_EXPECT(Square(Square_h1).GetBitboard().Shift<Direction::NorthWest>() == Square(Square_g2).GetBitboard());
        TEST_EXPECT(Square(Square_a8).GetBitboard().Shift<Direction::SouthEast>() == Square(Square_b7).GetBitboard());
        TEST_EXPECT(Square(Square_h8).GetBitboard().Shift<Direction::SouthWest>() == Square(Square_g7).GetBitboard());
        // Corner edge cases: diagonal shift off both edges = 0
        TEST_EXPECT(Square(Square_h8).GetBitboard().Shift<Direction::NorthEast>() == Bitboard(0));
        TEST_EXPECT(Square(Square_a1).GetBitboard().Shift<Direction::SouthWest>() == Bitboard(0));
    }

    // MirroredVertically (flips ranks: rank 0 ↔ rank 7)
    {
        TEST_EXPECT(Square(Square_a1).GetBitboard().MirroredVertically() == Square(Square_a8).GetBitboard());
        TEST_EXPECT(Square(Square_h1).GetBitboard().MirroredVertically() == Square(Square_h8).GetBitboard());
        TEST_EXPECT(Square(Square_e4).GetBitboard().MirroredVertically() == Square(Square_e5).GetBitboard());
        TEST_EXPECT(Bitboard::Full().MirroredVertically() == Bitboard::Full());
        // Double mirror = identity
        for (uint32_t sq = 0; sq < 64; ++sq)
        {
            const Bitboard bb = Square(sq).GetBitboard();
            TEST_EXPECT(bb.MirroredVertically().MirroredVertically() == bb);
        }
    }

    // MirroredHorizontally (flips files: file a ↔ file h)
    {
        TEST_EXPECT(Square(Square_a1).GetBitboard().MirroredHorizontally() == Square(Square_h1).GetBitboard());
        TEST_EXPECT(Square(Square_e4).GetBitboard().MirroredHorizontally() == Square(Square_d4).GetBitboard());
        TEST_EXPECT(Bitboard::Full().MirroredHorizontally() == Bitboard::Full());
        for (uint32_t sq = 0; sq < 64; ++sq)
        {
            const Bitboard bb = Square(sq).GetBitboard();
            TEST_EXPECT(bb.MirroredHorizontally().MirroredHorizontally() == bb);
        }
    }

    // Rotated180 = MirroredVertically + MirroredHorizontally
    {
        TEST_EXPECT(Square(Square_a1).GetBitboard().Rotated180() == Square(Square_h8).GetBitboard());
        TEST_EXPECT(Square(Square_h8).GetBitboard().Rotated180() == Square(Square_a1).GetBitboard());
        TEST_EXPECT(Bitboard::Full().Rotated180() == Bitboard::Full());
        for (uint32_t sq = 0; sq < 64; ++sq)
        {
            const Bitboard bb = Square(sq).GetBitboard();
            TEST_EXPECT(bb.Rotated180() == bb.MirroredVertically().MirroredHorizontally());
        }
    }

    // FlippedDiagonally (a1-h8 diagonal: transposes file ↔ rank)
    {
        TEST_EXPECT(Square(Square_a1).GetBitboard().FlippedDiagonally() == Square(Square_a1).GetBitboard());
        TEST_EXPECT(Square(Square_h8).GetBitboard().FlippedDiagonally() == Square(Square_h8).GetBitboard());
        TEST_EXPECT(Square(Square_b1).GetBitboard().FlippedDiagonally() == Square(Square_a2).GetBitboard());
        TEST_EXPECT(Square(Square_a2).GetBitboard().FlippedDiagonally() == Square(Square_b1).GetBitboard());
        // Double flip = identity
        for (uint32_t sq = 0; sq < 64; ++sq)
        {
            const Bitboard bb = Square(sq).GetBitboard();
            TEST_EXPECT(bb.FlippedDiagonally().FlippedDiagonally() == bb);
        }
    }

    // FlippedAntiDiagonally (a8-h1 diagonal)
    {
        TEST_EXPECT(Square(Square_a8).GetBitboard().FlippedAntiDiagonally() == Square(Square_a8).GetBitboard());
        TEST_EXPECT(Square(Square_h1).GetBitboard().FlippedAntiDiagonally() == Square(Square_h1).GetBitboard());
        TEST_EXPECT(Square(Square_a1).GetBitboard().FlippedAntiDiagonally() == Square(Square_h8).GetBitboard());
        for (uint32_t sq = 0; sq < 64; ++sq)
        {
            const Bitboard bb = Square(sq).GetBitboard();
            TEST_EXPECT(bb.FlippedAntiDiagonally().FlippedAntiDiagonally() == bb);
        }
    }
}

void RunSlidingAttackTests()
{
    std::cout << "Running sliding attack tests..." << std::endl;

    // Verify magic bitboard attacks match slow reference implementation
    for (uint32_t sq = 0; sq < 64; ++sq)
    {
        const Square square(sq);

        // No blockers
        TEST_EXPECT(Bitboard::GenerateRookAttacks(square, 0) == Bitboard::GenerateRookAttacks_Slow(square, 0));
        TEST_EXPECT(Bitboard::GenerateBishopAttacks(square, 0) == Bitboard::GenerateBishopAttacks_Slow(square, 0));

        // Single blocker at each square
        for (uint32_t bsq = 0; bsq < 64; ++bsq)
        {
            const Bitboard blockers = Square(bsq).GetBitboard();
            TEST_EXPECT(Bitboard::GenerateRookAttacks(square, blockers) == Bitboard::GenerateRookAttacks_Slow(square, blockers));
            TEST_EXPECT(Bitboard::GenerateBishopAttacks(square, blockers) == Bitboard::GenerateBishopAttacks_Slow(square, blockers));
        }
    }

    // Queen attacks = rook attacks | bishop attacks
    for (uint32_t sq = 0; sq < 64; ++sq)
    {
        const Square square(sq);
        TEST_EXPECT(Bitboard::GenerateQueenAttacks(square, 0) ==
            (Bitboard::GenerateRookAttacks(square, 0) | Bitboard::GenerateBishopAttacks(square, 0)));

        const Bitboard blockers = Bitboard::RankBitboard<3>() | Bitboard::FileBitboard<3>();
        TEST_EXPECT(Bitboard::GenerateQueenAttacks(square, blockers) ==
            (Bitboard::GenerateRookAttacks(square, blockers) | Bitboard::GenerateBishopAttacks(square, blockers)));
    }

    // Specific blocker scenario: rook on a1 with blocker on a4
    {
        const Bitboard blockers = Square(Square_a4).GetBitboard();
        const Bitboard attacks = Bitboard::GenerateRookAttacks(Square(Square_a1), blockers);
        TEST_EXPECT(attacks.IsBitSet(Square(Square_a2).Index()));
        TEST_EXPECT(attacks.IsBitSet(Square(Square_a3).Index()));
        TEST_EXPECT(attacks.IsBitSet(Square(Square_a4).Index()));  // can capture blocker
        TEST_EXPECT(!attacks.IsBitSet(Square(Square_a5).Index())); // blocked
        TEST_EXPECT(attacks.IsBitSet(Square(Square_b1).Index()));
        TEST_EXPECT(attacks.IsBitSet(Square(Square_h1).Index()));
    }

    // Specific blocker scenario: bishop on d4 with blocker on f6
    {
        const Bitboard blockers = Square(Square_f6).GetBitboard();
        const Bitboard attacks = Bitboard::GenerateBishopAttacks(Square(Square_d4), blockers);
        TEST_EXPECT(attacks.IsBitSet(Square(Square_e5).Index()));
        TEST_EXPECT(attacks.IsBitSet(Square(Square_f6).Index()));  // can capture
        TEST_EXPECT(!attacks.IsBitSet(Square(Square_g7).Index())); // blocked
    }

    // Sliding pieces don't attack their own square
    for (uint32_t sq = 0; sq < 64; ++sq)
    {
        TEST_EXPECT(!Bitboard::GenerateRookAttacks(Square(sq), 0).IsBitSet(sq));
        TEST_EXPECT(!Bitboard::GenerateBishopAttacks(Square(sq), 0).IsBitSet(sq));
        TEST_EXPECT(!Bitboard::GenerateQueenAttacks(Square(sq), 0).IsBitSet(sq));
    }

    // Empty-board attacks: GetXxxAttacks includes the source square, GenerateXxxAttacks excludes it
    for (uint32_t sq = 0; sq < 64; ++sq)
    {
        const Square square(sq);
        TEST_EXPECT((Bitboard::GetRookAttacks(square) & ~square.GetBitboard()) == Bitboard::GenerateRookAttacks(square, 0));
        TEST_EXPECT((Bitboard::GetBishopAttacks(square) & ~square.GetBitboard()) == Bitboard::GenerateBishopAttacks(square, 0));
        TEST_EXPECT((Bitboard::GetQueenAttacks(square) & ~square.GetBitboard()) == Bitboard::GenerateQueenAttacks(square, 0));
    }
}

void RunRayTests()
{
    std::cout << "Running ray tests..." << std::endl;

    // Rays from a1 (corner)
    {
        const Bitboard north = Bitboard::GetRay(Square(Square_a1), Direction::North);
        TEST_EXPECT_EQ(north.Count(), 7u);
        TEST_EXPECT(north.IsBitSet(Square(Square_a2).Index()));
        TEST_EXPECT(north.IsBitSet(Square(Square_a8).Index()));
        TEST_EXPECT(!north.IsBitSet(Square(Square_a1).Index()));

        const Bitboard east = Bitboard::GetRay(Square(Square_a1), Direction::East);
        TEST_EXPECT_EQ(east.Count(), 7u);
        TEST_EXPECT(east.IsBitSet(Square(Square_b1).Index()));
        TEST_EXPECT(east.IsBitSet(Square(Square_h1).Index()));

        const Bitboard ne = Bitboard::GetRay(Square(Square_a1), Direction::NorthEast);
        TEST_EXPECT_EQ(ne.Count(), 7u);
        TEST_EXPECT(ne.IsBitSet(Square(Square_b2).Index()));
        TEST_EXPECT(ne.IsBitSet(Square(Square_h8).Index()));

        // No squares south, west, or diagonal-toward-corner from a1
        TEST_EXPECT(Bitboard::GetRay(Square(Square_a1), Direction::South) == Bitboard(0));
        TEST_EXPECT(Bitboard::GetRay(Square(Square_a1), Direction::West) == Bitboard(0));
        TEST_EXPECT(Bitboard::GetRay(Square(Square_a1), Direction::SouthWest) == Bitboard(0));
        TEST_EXPECT(Bitboard::GetRay(Square(Square_a1), Direction::SouthEast) == Bitboard(0));
        TEST_EXPECT(Bitboard::GetRay(Square(Square_a1), Direction::NorthWest) == Bitboard(0));
    }

    // Rays from h8 (opposite corner)
    {
        TEST_EXPECT_EQ(Bitboard::GetRay(Square(Square_h8), Direction::South).Count(), 7u);
        TEST_EXPECT_EQ(Bitboard::GetRay(Square(Square_h8), Direction::West).Count(), 7u);
        TEST_EXPECT_EQ(Bitboard::GetRay(Square(Square_h8), Direction::SouthWest).Count(), 7u);
        TEST_EXPECT(Bitboard::GetRay(Square(Square_h8), Direction::North) == Bitboard(0));
        TEST_EXPECT(Bitboard::GetRay(Square(Square_h8), Direction::East) == Bitboard(0));
        TEST_EXPECT(Bitboard::GetRay(Square(Square_h8), Direction::NorthEast) == Bitboard(0));
    }

    // Rays from center (e4)
    {
        TEST_EXPECT_EQ(Bitboard::GetRay(Square(Square_e4), Direction::North).Count(), 4u);     // e5-e8
        TEST_EXPECT_EQ(Bitboard::GetRay(Square(Square_e4), Direction::South).Count(), 3u);     // e3-e1
        TEST_EXPECT_EQ(Bitboard::GetRay(Square(Square_e4), Direction::East).Count(), 3u);      // f4-h4
        TEST_EXPECT_EQ(Bitboard::GetRay(Square(Square_e4), Direction::West).Count(), 4u);      // d4-a4
        TEST_EXPECT_EQ(Bitboard::GetRay(Square(Square_e4), Direction::NorthEast).Count(), 3u); // f5,g6,h7
        TEST_EXPECT_EQ(Bitboard::GetRay(Square(Square_e4), Direction::NorthWest).Count(), 4u); // d5,c6,b7,a8
        TEST_EXPECT_EQ(Bitboard::GetRay(Square(Square_e4), Direction::SouthEast).Count(), 3u); // f3,g2,h1
        TEST_EXPECT_EQ(Bitboard::GetRay(Square(Square_e4), Direction::SouthWest).Count(), 3u); // d3,c2,b1
    }

    // Opposite rays on the same line cover the full file/rank (minus the source)
    for (uint32_t sq = 0; sq < 64; ++sq)
    {
        const Square s(sq);
        const Bitboard file = Bitboard::GetRay(s, Direction::North) | Bitboard::GetRay(s, Direction::South);
        TEST_EXPECT_EQ(file.Count(), 7u);
        const Bitboard rank = Bitboard::GetRay(s, Direction::East) | Bitboard::GetRay(s, Direction::West);
        TEST_EXPECT_EQ(rank.Count(), 7u);
    }
}

void RunPieceAttackDetailTests()
{
    std::cout << "Running piece attack detail tests..." << std::endl;

    // King attacks: specific squares
    {
        // Corner a1: 3 attacks (a2, b1, b2)
        const Bitboard a1Atk = Bitboard::GetKingAttacks(Square(Square_a1));
        TEST_EXPECT_EQ(a1Atk.Count(), 3u);
        TEST_EXPECT(a1Atk.IsBitSet(Square(Square_a2).Index()));
        TEST_EXPECT(a1Atk.IsBitSet(Square(Square_b1).Index()));
        TEST_EXPECT(a1Atk.IsBitSet(Square(Square_b2).Index()));

        // Corner h8: 3 attacks (g8, h7, g7)
        const Bitboard h8Atk = Bitboard::GetKingAttacks(Square(Square_h8));
        TEST_EXPECT_EQ(h8Atk.Count(), 3u);
        TEST_EXPECT(h8Atk.IsBitSet(Square(Square_g8).Index()));
        TEST_EXPECT(h8Atk.IsBitSet(Square(Square_h7).Index()));
        TEST_EXPECT(h8Atk.IsBitSet(Square(Square_g7).Index()));

        // Edge a4: 5 attacks
        TEST_EXPECT_EQ(Bitboard::GetKingAttacks(Square(Square_a4)).Count(), 5u);

        // Center e4: 8 attacks
        const Bitboard e4Atk = Bitboard::GetKingAttacks(Square(Square_e4));
        TEST_EXPECT_EQ(e4Atk.Count(), 8u);
        TEST_EXPECT(e4Atk.IsBitSet(Square(Square_d3).Index()));
        TEST_EXPECT(e4Atk.IsBitSet(Square(Square_d4).Index()));
        TEST_EXPECT(e4Atk.IsBitSet(Square(Square_d5).Index()));
        TEST_EXPECT(e4Atk.IsBitSet(Square(Square_e3).Index()));
        TEST_EXPECT(e4Atk.IsBitSet(Square(Square_e5).Index()));
        TEST_EXPECT(e4Atk.IsBitSet(Square(Square_f3).Index()));
        TEST_EXPECT(e4Atk.IsBitSet(Square(Square_f4).Index()));
        TEST_EXPECT(e4Atk.IsBitSet(Square(Square_f5).Index()));

        // King never attacks own square
        for (uint32_t sq = 0; sq < 64; ++sq)
            TEST_EXPECT(!Bitboard::GetKingAttacks(Square(sq)).IsBitSet(sq));
    }

    // Knight attacks: specific squares
    {
        // Corner a1: 2 attacks (b3, c2)
        const Bitboard a1Atk = Bitboard::GetKnightAttacks(Square(Square_a1));
        TEST_EXPECT_EQ(a1Atk.Count(), 2u);
        TEST_EXPECT(a1Atk.IsBitSet(Square(Square_b3).Index()));
        TEST_EXPECT(a1Atk.IsBitSet(Square(Square_c2).Index()));

        // Corner h1: 2 attacks (f2, g3)
        const Bitboard h1Atk = Bitboard::GetKnightAttacks(Square(Square_h1));
        TEST_EXPECT_EQ(h1Atk.Count(), 2u);
        TEST_EXPECT(h1Atk.IsBitSet(Square(Square_f2).Index()));
        TEST_EXPECT(h1Atk.IsBitSet(Square(Square_g3).Index()));

        // Near-corner b1: 3 attacks (a3, c3, d2)
        const Bitboard b1Atk = Bitboard::GetKnightAttacks(Square(Square_b1));
        TEST_EXPECT_EQ(b1Atk.Count(), 3u);
        TEST_EXPECT(b1Atk.IsBitSet(Square(Square_a3).Index()));
        TEST_EXPECT(b1Atk.IsBitSet(Square(Square_c3).Index()));
        TEST_EXPECT(b1Atk.IsBitSet(Square(Square_d2).Index()));

        // Center e4: 8 attacks
        const Bitboard e4Atk = Bitboard::GetKnightAttacks(Square(Square_e4));
        TEST_EXPECT_EQ(e4Atk.Count(), 8u);
        TEST_EXPECT(e4Atk.IsBitSet(Square(Square_c3).Index()));
        TEST_EXPECT(e4Atk.IsBitSet(Square(Square_c5).Index()));
        TEST_EXPECT(e4Atk.IsBitSet(Square(Square_d2).Index()));
        TEST_EXPECT(e4Atk.IsBitSet(Square(Square_d6).Index()));
        TEST_EXPECT(e4Atk.IsBitSet(Square(Square_f2).Index()));
        TEST_EXPECT(e4Atk.IsBitSet(Square(Square_f6).Index()));
        TEST_EXPECT(e4Atk.IsBitSet(Square(Square_g3).Index()));
        TEST_EXPECT(e4Atk.IsBitSet(Square(Square_g5).Index()));

        // Knight never attacks own square
        for (uint32_t sq = 0; sq < 64; ++sq)
            TEST_EXPECT(!Bitboard::GetKnightAttacks(Square(sq)).IsBitSet(sq));
    }

    // Multi-piece knight attacks: GetKnightAttacks(Bitboard)
    {
        // Two knights: Bitboard version = union of individual attacks
        const Bitboard knights = Square(Square_a1).GetBitboard() | Square(Square_h8).GetBitboard();
        TEST_EXPECT(Bitboard::GetKnightAttacks(knights) ==
            (Bitboard::GetKnightAttacks(Square(Square_a1)) | Bitboard::GetKnightAttacks(Square(Square_h8))));

        // Single knight: both versions match
        for (uint32_t sq = 0; sq < 64; ++sq)
            TEST_EXPECT(Bitboard::GetKnightAttacks(Square(sq).GetBitboard()) == Bitboard::GetKnightAttacks(Square(sq)));
    }

    // Pawn attacks: edge files
    {
        // White pawn on a4: only attacks b5
        const Bitboard wa4 = Bitboard::GetPawnAttacks(Square(Square_a4), White);
        TEST_EXPECT_EQ(wa4.Count(), 1u);
        TEST_EXPECT(wa4.IsBitSet(Square(Square_b5).Index()));

        // White pawn on h4: only attacks g5
        const Bitboard wh4 = Bitboard::GetPawnAttacks(Square(Square_h4), White);
        TEST_EXPECT_EQ(wh4.Count(), 1u);
        TEST_EXPECT(wh4.IsBitSet(Square(Square_g5).Index()));

        // Black pawn on a5: only attacks b4
        const Bitboard ba5 = Bitboard::GetPawnAttacks(Square(Square_a5), Black);
        TEST_EXPECT_EQ(ba5.Count(), 1u);
        TEST_EXPECT(ba5.IsBitSet(Square(Square_b4).Index()));

        // Black pawn on h5: only attacks g4
        const Bitboard bh5 = Bitboard::GetPawnAttacks(Square(Square_h5), Black);
        TEST_EXPECT_EQ(bh5.Count(), 1u);
        TEST_EXPECT(bh5.IsBitSet(Square(Square_g4).Index()));

        // Center pawn: 2 attacks
        const Bitboard wd4 = Bitboard::GetPawnAttacks(Square(Square_d4), White);
        TEST_EXPECT_EQ(wd4.Count(), 2u);
        TEST_EXPECT(wd4.IsBitSet(Square(Square_c5).Index()));
        TEST_EXPECT(wd4.IsBitSet(Square(Square_e5).Index()));
    }

    // Multi-pawn attacks: GetPawnsAttacks<Color>(Bitboard)
    {
        // White pawns on b2 and d2: attacks a3, c3, e3 (c3 overlaps)
        const Bitboard wPawns = Square(Square_b2).GetBitboard() | Square(Square_d2).GetBitboard();
        const Bitboard wAtk = Bitboard::GetPawnsAttacks<White>(wPawns);
        TEST_EXPECT(wAtk.IsBitSet(Square(Square_a3).Index()));
        TEST_EXPECT(wAtk.IsBitSet(Square(Square_c3).Index()));
        TEST_EXPECT(wAtk.IsBitSet(Square(Square_e3).Index()));
        TEST_EXPECT_EQ(wAtk.Count(), 3u);

        // Black pawns on b7 and d7: attacks a6, c6, e6
        const Bitboard bPawns = Square(Square_b7).GetBitboard() | Square(Square_d7).GetBitboard();
        const Bitboard bAtk = Bitboard::GetPawnsAttacks<Black>(bPawns);
        TEST_EXPECT(bAtk.IsBitSet(Square(Square_a6).Index()));
        TEST_EXPECT(bAtk.IsBitSet(Square(Square_c6).Index()));
        TEST_EXPECT(bAtk.IsBitSet(Square(Square_e6).Index()));
        TEST_EXPECT_EQ(bAtk.Count(), 3u);
    }
}

void RunSquareUtilityTests()
{
    std::cout << "Running Square utility tests..." << std::endl;

    // Construction and File / Rank
    {
        TEST_EXPECT_EQ(Square(Square_a1).File(), 0u);
        TEST_EXPECT_EQ(Square(Square_a1).Rank(), 0u);
        TEST_EXPECT_EQ(Square(Square_h8).File(), 7u);
        TEST_EXPECT_EQ(Square(Square_h8).Rank(), 7u);
        TEST_EXPECT_EQ(Square(Square_e4).File(), 4u);
        TEST_EXPECT_EQ(Square(Square_e4).Rank(), 3u);
        TEST_EXPECT(Square(0, 0) == Square(Square_a1));
        TEST_EXPECT(Square(7, 7) == Square(Square_h8));
        TEST_EXPECT(Square(4, 3) == Square(Square_e4));
    }

    // Distance (Chebyshev)
    {
        TEST_EXPECT_EQ(Square::Distance(Square(Square_a1), Square(Square_a1)), 0);
        TEST_EXPECT_EQ(Square::Distance(Square(Square_a1), Square(Square_a2)), 1);
        TEST_EXPECT_EQ(Square::Distance(Square(Square_a1), Square(Square_b2)), 1);
        TEST_EXPECT_EQ(Square::Distance(Square(Square_a1), Square(Square_h8)), 7);
        TEST_EXPECT_EQ(Square::Distance(Square(Square_a1), Square(Square_a8)), 7);
        TEST_EXPECT_EQ(Square::Distance(Square(Square_a1), Square(Square_h1)), 7);
        // Symmetric
        TEST_EXPECT_EQ(
            Square::Distance(Square(Square_a1), Square(Square_e4)),
            Square::Distance(Square(Square_e4), Square(Square_a1)));
        // Lookup matches compute
        for (uint32_t a = 0; a < 64; ++a)
            for (uint32_t b = 0; b < 64; ++b)
                TEST_EXPECT_EQ(Square::Distance(Square(a), Square(b)), Square::ComputeDistance(Square(a), Square(b)));
    }

    // FlippedFile (a↔h, b↔g, c↔f, d↔e)
    {
        TEST_EXPECT(Square(Square_a1).FlippedFile() == Square(Square_h1));
        TEST_EXPECT(Square(Square_h1).FlippedFile() == Square(Square_a1));
        TEST_EXPECT(Square(Square_e4).FlippedFile() == Square(Square_d4));
        TEST_EXPECT_EQ(Square(Square_a1).FlippedFile().Rank(), 0u); // rank unchanged
        for (uint32_t sq = 0; sq < 64; ++sq)
            TEST_EXPECT(Square(sq).FlippedFile().FlippedFile() == Square(sq));
    }

    // FlippedRank (1↔8, 2↔7, 3↔6, 4↔5)
    {
        TEST_EXPECT(Square(Square_a1).FlippedRank() == Square(Square_a8));
        TEST_EXPECT(Square(Square_a8).FlippedRank() == Square(Square_a1));
        TEST_EXPECT(Square(Square_e4).FlippedRank() == Square(Square_e5));
        TEST_EXPECT_EQ(Square(Square_a1).FlippedRank().File(), 0u); // file unchanged
        for (uint32_t sq = 0; sq < 64; ++sq)
            TEST_EXPECT(Square(sq).FlippedRank().FlippedRank() == Square(sq));
    }

    // FlippedFileAndRank
    {
        TEST_EXPECT(Square(Square_a1).FlippedFileAndRank() == Square(Square_h8));
        TEST_EXPECT(Square(Square_h8).FlippedFileAndRank() == Square(Square_a1));
        for (uint32_t sq = 0; sq < 64; ++sq)
        {
            TEST_EXPECT(Square(sq).FlippedFileAndRank().FlippedFileAndRank() == Square(sq));
            TEST_EXPECT(Square(sq).FlippedFileAndRank() == Square(sq).FlippedFile().FlippedRank());
        }
    }

    // IsCorner
    {
        TEST_EXPECT(Square(Square_a1).IsCorner());
        TEST_EXPECT(Square(Square_h1).IsCorner());
        TEST_EXPECT(Square(Square_a8).IsCorner());
        TEST_EXPECT(Square(Square_h8).IsCorner());
        TEST_EXPECT(!Square(Square_e4).IsCorner());
        TEST_EXPECT(!Square(Square_a2).IsCorner());
        TEST_EXPECT(!Square(Square_b1).IsCorner());
    }

    // EdgeDistance
    {
        TEST_EXPECT_EQ(Square(Square_a1).EdgeDistance(), 0);
        TEST_EXPECT_EQ(Square(Square_h8).EdgeDistance(), 0);
        TEST_EXPECT_EQ(Square(Square_a4).EdgeDistance(), 0);
        TEST_EXPECT_EQ(Square(Square_d1).EdgeDistance(), 0);
        TEST_EXPECT_EQ(Square(Square_b2).EdgeDistance(), 1);
        TEST_EXPECT_EQ(Square(Square_g7).EdgeDistance(), 1);
        TEST_EXPECT_EQ(Square(Square_d4).EdgeDistance(), 3);
        TEST_EXPECT_EQ(Square(Square_e4).EdgeDistance(), 3);
    }

    // FromString / ToString round-trip
    {
        TEST_EXPECT(Square::FromString("a1") == Square(Square_a1));
        TEST_EXPECT(Square::FromString("h8") == Square(Square_h8));
        TEST_EXPECT(Square::FromString("e4") == Square(Square_e4));
        TEST_EXPECT(Square(Square_a1).ToString() == "a1");
        TEST_EXPECT(Square(Square_h8).ToString() == "h8");
        TEST_EXPECT(Square(Square_e4).ToString() == "e4");
        for (uint32_t sq = 0; sq < 64; ++sq)
            TEST_EXPECT(Square::FromString(Square(sq).ToString()) == Square(sq));
    }

    // Safe movement (returns Invalid() at edges)
    {
        TEST_EXPECT(Square(Square_a1).South() == Square::Invalid());
        TEST_EXPECT(Square(Square_a1).West() == Square::Invalid());
        TEST_EXPECT(Square(Square_h8).North() == Square::Invalid());
        TEST_EXPECT(Square(Square_h8).East() == Square::Invalid());
        TEST_EXPECT(Square(Square_a1).North() == Square(Square_a2));
        TEST_EXPECT(Square(Square_a1).East() == Square(Square_b1));
        TEST_EXPECT(Square(Square_e4).North() == Square(Square_e5));
        TEST_EXPECT(Square(Square_e4).South() == Square(Square_e3));
        TEST_EXPECT(Square(Square_e4).East() == Square(Square_f4));
        TEST_EXPECT(Square(Square_e4).West() == Square(Square_d4));
    }

    // RelativeRank
    {
        TEST_EXPECT_EQ(Square(Square_a1).RelativeRank(White), 0u);
        TEST_EXPECT_EQ(Square(Square_a8).RelativeRank(White), 7u);
        TEST_EXPECT_EQ(Square(Square_a1).RelativeRank(Black), 7u);
        TEST_EXPECT_EQ(Square(Square_a8).RelativeRank(Black), 0u);
    }

    // Diagonal / AntiDiagonal
    {
        // Squares on same diagonal share Diagonal() value
        TEST_EXPECT_EQ(Square(Square_a1).Diagonal(), Square(Square_b2).Diagonal());
        TEST_EXPECT_EQ(Square(Square_b2).Diagonal(), Square(Square_c3).Diagonal());
        TEST_EXPECT_EQ(Square(Square_a1).Diagonal(), Square(Square_h8).Diagonal());
        TEST_EXPECT(Square(Square_a1).Diagonal() != Square(Square_a2).Diagonal());

        // Squares on same anti-diagonal share AntiDiagonal() value
        TEST_EXPECT_EQ(Square(Square_a8).AntiDiagonal(), Square(Square_b7).AntiDiagonal());
        TEST_EXPECT_EQ(Square(Square_b7).AntiDiagonal(), Square(Square_c6).AntiDiagonal());
        TEST_EXPECT_EQ(Square(Square_a8).AntiDiagonal(), Square(Square_h1).AntiDiagonal());
        TEST_EXPECT(Square(Square_a1).AntiDiagonal() != Square(Square_a2).AntiDiagonal());
    }
}
