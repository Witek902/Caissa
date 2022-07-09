#pragma once

#include "Common.hpp"

#include <sstream>
#include <string>
#include <iomanip>

inline bool IsMate(const ScoreType score)
{
	return score > CheckmateValue - (int32_t)MaxSearchDepth || score < -CheckmateValue + (int32_t)MaxSearchDepth;
}

inline std::string ScoreToStr(const ScoreType score)
{
    std::stringstream ss;

    if (score > CheckmateValue - MaxSearchDepth)
    {
        ss << "+M" << ((CheckmateValue - score + 1) / 2);
    }
    else if (score < -CheckmateValue + MaxSearchDepth)
    {
        ss << "-M" << ((CheckmateValue + score + 1) / 2);
    }
    else
    {
        if (score < 0)
        {
            ss << std::fixed << std::setprecision(2) << (float(score) / 100.0f);
        }
        else
        {
            ss << '+' << std::fixed << std::setprecision(2) << (float(score) / 100.0f);
        }
    }

    return ss.str();
}