#pragma once

#include <string>

enum class Color : uint8_t
{
    White,
    Black,
};

INLINE Color GetOppositeColor(Color color)
{
    return Color((uint8_t)color ^ 1);
}

INLINE ScoreType ColorMultiplier(Color color)
{
    return color == Color::White ? 1 : -1;
}
