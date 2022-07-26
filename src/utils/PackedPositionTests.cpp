#include "../backend/Position.hpp"
#include "../backend/PositionUtils.hpp"
#include "../backend/Material.hpp"

#include <iostream>

#define TEST_EXPECT(x) \
    if (!(x)) { std::cout << "Test failed: " << #x << std::endl; DEBUG_BREAK(); }

void RunPackedPositionTests()
{
    std::cout << "Running PackedPosition tests..." << std::endl;

    {
        const Position originalPos(Position::InitPositionFEN);
        PackedPosition packedPos;
        Position unpackedPos;

        TEST_EXPECT(PackPosition(originalPos, packedPos));
        TEST_EXPECT(UnpackPosition(packedPos, unpackedPos));
        TEST_EXPECT(originalPos == unpackedPos);
    }

    // en passant, white to move
    {
        const Position originalPos("r1bqkbnr/pppp1ppp/2n5/3Pp3/8/8/PPP1PPPP/RNBQKBNR w KQkq e6 0 3");
        PackedPosition packedPos;
        Position unpackedPos;

        TEST_EXPECT(PackPosition(originalPos, packedPos));
        TEST_EXPECT(UnpackPosition(packedPos, unpackedPos));
        TEST_EXPECT(originalPos == unpackedPos);
    }

    // en passant, black to move
    {
        const Position originalPos("rnbqkbnr/pppp1ppp/8/8/3PpP2/2P5/PP2P1PP/RNBQKBNR b KQkq f3 0 3");
        PackedPosition packedPos;
        Position unpackedPos;

        TEST_EXPECT(PackPosition(originalPos, packedPos));
        TEST_EXPECT(UnpackPosition(packedPos, unpackedPos));
        TEST_EXPECT(originalPos == unpackedPos);
    }
    
    // castling rights test
    for (uint32_t i = 0; i < 16; ++i)
    {
        std::string fen = "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w ";
        if (i == 0) fen += '-';
        if (i & 1) fen += 'K';
        if (i & 2) fen += 'Q';
        if (i & 4) fen += 'k';
        if (i & 8) fen += 'q';
        fen += " - 0 1";

        const Position originalPos(fen);
        PackedPosition packedPos;
        Position unpackedPos;

        TEST_EXPECT(PackPosition(originalPos, packedPos));
        TEST_EXPECT(UnpackPosition(packedPos, unpackedPos));
        TEST_EXPECT(originalPos == unpackedPos);
    }

    // random positions
    {
        using Distr = std::uniform_int_distribution<uint32_t>;

        std::mt19937 mt;

        for (uint32_t i = 0; i < 1000; ++i)
        {
            MaterialKey key;
            key.numWhitePawns   = Distr(0, 8)(mt);
            key.numWhiteKnights = Distr(0, 2)(mt);
            key.numWhiteBishops = Distr(0, 2)(mt);
            key.numWhiteRooks   = Distr(0, 2)(mt);
            key.numWhiteQueens  = Distr(0, 1)(mt);
            key.numBlackPawns   = Distr(0, 8)(mt);
            key.numBlackKnights = Distr(0, 2)(mt);
            key.numBlackBishops = Distr(0, 2)(mt);
            key.numBlackRooks   = Distr(0, 2)(mt);
            key.numBlackQueens  = Distr(0, 1)(mt);

            Position originalPos;
            GenerateRandomPosition(mt, key, originalPos);

            PackedPosition packedPos;
            TEST_EXPECT(PackPosition(originalPos, packedPos));

            Position unpackedPos;
            TEST_EXPECT(UnpackPosition(packedPos, unpackedPos));
            TEST_EXPECT(originalPos == unpackedPos);
        }
    }
}
