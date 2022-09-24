#include "Square.hpp"

alignas(CACHELINE_SIZE) uint8_t Square::sDistances[Square::NumSquares * Square::NumSquares];

Square Square::FromString(const std::string& str)
{
    if (str.length() != 2 || str[0] < 'a' || str[0] > 'h' || str[1] < '1' || str[1] > '8')
    {
        return {};
    }

    return Square(str[0] - 'a', str[1] - '1');
}

std::string Square::ToString() const
{
    std::string str;
    str += 'a' + File();
    str += '1' + Rank();
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
