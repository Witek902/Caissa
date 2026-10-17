#include "Material.hpp"
#include "Position.hpp"

static_assert(sizeof(MaterialKey) == sizeof(uint64_t), "Invalid material key size");

static void AppendRepeated(String& str, char c, uint32_t count)
{
    for (uint32_t i = 0; i < count; ++i)
    {
        str.Append(c);
    }
}

String MaterialKey::ToString() const
{
    String str = "K";

    AppendRepeated(str, 'Q', (uint32_t)numWhiteQueens);
    AppendRepeated(str, 'R', (uint32_t)numWhiteRooks);
    AppendRepeated(str, 'B', (uint32_t)numWhiteBishops);
    AppendRepeated(str, 'N', (uint32_t)numWhiteKnights);
    AppendRepeated(str, 'P', (uint32_t)numWhitePawns);

    str.Append("vK");

    AppendRepeated(str, 'Q', (uint32_t)numBlackQueens);
    AppendRepeated(str, 'R', (uint32_t)numBlackRooks);
    AppendRepeated(str, 'B', (uint32_t)numBlackBishops);
    AppendRepeated(str, 'N', (uint32_t)numBlackKnights);
    AppendRepeated(str, 'P', (uint32_t)numBlackPawns);

    return str;
}

void MaterialKey::FromString(const char* str)
{
    value = 0;

    Color side = White;

    for (;;)
    {
        const char c = *(str++);

        if (c == 0)
        {
            break;
        }

        if (c == 'v')
        {
            side = Black;
            continue;
        }

        if (side == White)
        {
            if (c == 'P') numWhitePawns++;
            if (c == 'N') numWhiteKnights++;
            if (c == 'B') numWhiteBishops++;
            if (c == 'R') numWhiteRooks++;
            if (c == 'Q') numWhiteQueens++;
        }
        else
        {
            if (c == 'P') numBlackPawns++;
            if (c == 'N') numBlackKnights++;
            if (c == 'B') numBlackBishops++;
            if (c == 'R') numBlackRooks++;
            if (c == 'Q') numBlackQueens++;
        }
    }
}

uint32_t MaterialKey::GetNeuralNetworkInputsNumber() const
{
    uint32_t inputs = 0;

    if (numWhitePawns > 0 || numBlackPawns > 0)
    {
        // has pawns, so can't exploit vertical/diagonal symmetry
        inputs += 32; // white king on left files
        inputs += 64; // black king on any file
    }
    else
    {
        // pawnless position, can exploit vertical/horizonal/diagonal symmetry
        inputs += 16; // white king on files A-D, ranks 1-4
        inputs += 64; // black king on any file

        // TODO diagonal symmetry
        //inputs += 36; // black king on bottom-right triangle (a1, b1, b2, c1, c2, c3, ...)
    }

    // knights/bishops/rooks/queens on any square
    if (numWhiteQueens)     inputs += 64;
    if (numBlackQueens)     inputs += 64;
    if (numWhiteRooks)      inputs += 64;
    if (numBlackRooks)      inputs += 64;
    if (numWhiteBishops)    inputs += 64;
    if (numBlackBishops)    inputs += 64;
    if (numWhiteKnights)    inputs += 64;
    if (numBlackKnights)    inputs += 64;

    // pawns on ranks 2-7
    if (numWhitePawns)      inputs += 48;
    if (numBlackPawns)      inputs += 48;

    return inputs;
}

MaterialMask BuildMaterialMask(const Position& pos)
{
    MaterialMask mask = (MaterialMask)0;

    if (pos.Whites().pawns)     mask = mask | MaterialMask_WhitePawn;
    if (pos.Whites().knights)   mask = mask | MaterialMask_WhiteKnight;
    if (pos.Whites().bishops)   mask = mask | MaterialMask_WhiteBishop;
    if (pos.Whites().rooks)     mask = mask | MaterialMask_WhiteRook;
    if (pos.Whites().queens)    mask = mask | MaterialMask_WhiteQueen;

    if (pos.Blacks().pawns)     mask = mask | MaterialMask_BlackPawn;
    if (pos.Blacks().knights)   mask = mask | MaterialMask_BlackKnight;
    if (pos.Blacks().bishops)   mask = mask | MaterialMask_BlackBishop;
    if (pos.Blacks().rooks)     mask = mask | MaterialMask_BlackRook;
    if (pos.Blacks().queens)    mask = mask | MaterialMask_BlackQueen;

    return mask;
}
