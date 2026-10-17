#pragma once

#include "containers/FixedString.hpp"

template<typename T>
struct TPieceScore
{
    T mg;
    T eg;

    TPieceScore() = default;
    INLINE constexpr TPieceScore(const T _mg, const T _eg) : mg(_mg), eg(_eg) { }
    INLINE TPieceScore(const T* ptr) : mg(ptr[0]), eg(ptr[1]) { }
    INLINE bool operator == (const TPieceScore<T>& rhs) const { return mg == rhs.mg && eg == rhs.eg; }

    template<typename T2>
    INLINE constexpr TPieceScore& operator += (const TPieceScore<T2>& rhs)
    {
        mg += rhs.mg;
        eg += rhs.eg;
        return *this;
    }

    template<typename T2>
    INLINE constexpr TPieceScore& operator -= (const TPieceScore<T2>& rhs)
    {
        mg -= rhs.mg;
        eg -= rhs.eg;
        return *this;
    }

    INLINE constexpr TPieceScore operator - () const
    {
        return { static_cast<T>(-mg), static_cast<T>(-eg) };
    }

    INLINE constexpr TPieceScore operator + (const TPieceScore rhs) const
    {
        return { mg + rhs.mg, eg + rhs.eg };
    }

    INLINE constexpr TPieceScore operator - (const TPieceScore rhs) const
    {
        return { mg - rhs.mg, eg - rhs.eg };
    }

    INLINE constexpr TPieceScore<int32_t> operator * (const int32_t rhs) const
    {
        return { mg * rhs, eg * rhs };
    }

    INLINE constexpr TPieceScore<int32_t> operator / (const int32_t rhs) const
    {
        return { mg / rhs, eg / rhs };
    }

    INLINE constexpr T Average() const { return (mg + eg) / 2; }
};

using PieceScore = TPieceScore<int16_t>;

inline bool IsMate(const ScoreType score)
{
    return score > CheckmateValue - (int32_t)MaxSearchDepth || score < -CheckmateValue + (int32_t)MaxSearchDepth;
}

using ScoreString = FixedString<16>;

inline ScoreString ScoreToStr(const ScoreType score)
{
    ScoreString str;

    if (score > CheckmateValue - MaxSearchDepth)
    {
        str.AppendFormat("+M%d", (CheckmateValue - score + 1) / 2);
    }
    else if (score < -CheckmateValue + MaxSearchDepth)
    {
        str.AppendFormat("-M%d", (CheckmateValue + score + 1) / 2);
    }
    else
    {
        if (score < 0)
        {
            str.AppendFormat("%.2f", float(score) / 100.0f);
        }
        else
        {
            str.AppendFormat("+%.2f", float(score) / 100.0f);
        }
    }

    return str;
}