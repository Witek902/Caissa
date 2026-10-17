#pragma once

#include "Common.hpp"

class TimePoint
{
public:

    TimePoint() = default;

#if defined(PLATFORM_WINDOWS)
    TimePoint(int64_t value) : mValue(value) { }
#elif defined(PLATFORM_LINUX)
    TimePoint(uint64_t value) : mValue(value) { }
#endif // PLATFORM

    float ToSeconds() const;

    bool IsValid() const;

    static TimePoint Invalid();
    static TimePoint GetCurrent();
    static TimePoint FromSeconds(float t);

    TimePoint operator - (const TimePoint& rhs) const;
    TimePoint operator + (const TimePoint& rhs) const;

    TimePoint& operator *= (const double rhs);

    bool operator >= (const TimePoint& rhs) const;
    bool operator != (const TimePoint& rhs) const;

private:

#ifdef PLATFORM_WINDOWS
    int64_t mValue; // performance counter ticks
    static const int64_t sFreq;
    static const float sPeriod;
#elif defined(PLATFORM_LINUX)
    uint64_t mValue; // nanoseconds
#endif // PLATFORM

};
