#pragma once

#include <string>

enum class Color : uint8_t
{
    White,
    Black,
};

inline Color GetOppositeColor(Color color)
{
    return Color((uint32_t)color ^ 1);
}
