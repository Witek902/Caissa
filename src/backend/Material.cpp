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
