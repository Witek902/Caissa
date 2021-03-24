#include "Square.hpp"

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