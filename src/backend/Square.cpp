#include "Square.hpp"

alignas(CACHELINE_SIZE) uint8_t Square::sDistances[Square::NumSquares * Square::NumSquares];

Square Square::FromString(const char* str)
{
    if (str[0] < 'a' || str[0] > 'h' || str[1] < '1' || str[1] > '8' || str[2] != 0)
    {
        return {};
    }

    return Square(str[0] - 'a', str[1] - '1');
}

SquareString Square::ToString() const
{
    SquareString str;
    str.Append(char('a' + File()));
    str.Append(char('1' + Rank()));
    return str;
}

void Square::Init()
{
    for (uint32_t i = 0; i < 64; ++i)
    {
        for (uint32_t j = 0; j < 64; ++j)
        {
            sDistances[64u * i + j] = (uint8_t)ComputeDistance(Square(i), Square(j));
        }
    }
}
