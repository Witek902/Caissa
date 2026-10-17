#include "Time.hpp"

#if defined(PLATFORM_WINDOWS)

#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif // NOMINMAX
#include <Windows.h>

namespace
{
    static int64_t GetCounterFrequency()
    {
        LARGE_INTEGER freq;
        QueryPerformanceFrequency(&freq);
        return freq.QuadPart;
    }

    static float GetCounterPeriod()
    {
        LARGE_INTEGER freq;
        QueryPerformanceFrequency(&freq);
        return 1.0f / static_cast<float>(freq.QuadPart);
    }
}

const int64_t TimePoint::sFreq = GetCounterFrequency();
const float TimePoint::sPeriod = GetCounterPeriod();

float TimePoint::ToSeconds() const
{
    return static_cast<float>(mValue) * sPeriod;
}

bool TimePoint::IsValid() const
{
    return mValue >= 0;
}

TimePoint TimePoint::Invalid()
{
    return { -1 };
}

TimePoint TimePoint::GetCurrent()
{
    LARGE_INTEGER value = {};
    QueryPerformanceCounter(&value);
    return { value.QuadPart };
}

TimePoint TimePoint::FromSeconds(float t)
{
    return { static_cast<int64_t>(sFreq * t) };
}

TimePoint TimePoint::operator - (const TimePoint& rhs) const
{
    return { mValue - rhs.mValue };
}

TimePoint TimePoint::operator + (const TimePoint& rhs) const
{
    return { mValue + rhs.mValue };
}

TimePoint& TimePoint::operator *= (const double rhs)
{
    mValue = (int64_t)(mValue * rhs);
    return *this;
}

bool TimePoint::operator >= (const TimePoint& rhs) const
{
    return mValue >= rhs.mValue;
}

bool TimePoint::operator != (const TimePoint& rhs) const
{
    return mValue != rhs.mValue;
}

#elif defined(PLATFORM_LINUX)

#include <time.h>

float TimePoint::ToSeconds() const
{
    return static_cast<float>(mValue) * 1.0e-9f;
}

bool TimePoint::IsValid() const
{
    return mValue < UINT64_MAX;
}

TimePoint TimePoint::Invalid()
{
    return { UINT64_MAX };
}

TimePoint TimePoint::GetCurrent()
{
    struct timespec value;
    clock_gettime(CLOCK_MONOTONIC, &value);

    return (uint64_t)value.tv_sec * 1000000000ull + value.tv_nsec;
}

TimePoint TimePoint::FromSeconds(float t)
{
    return { static_cast<uint64_t>(t * 1.0e+9f) };
}

TimePoint TimePoint::operator - (const TimePoint& rhs) const
{
    return { mValue - rhs.mValue };
}

TimePoint TimePoint::operator + (const TimePoint& rhs) const
{
    return { mValue + rhs.mValue };
}

TimePoint& TimePoint::operator *= (const double rhs)
{
    mValue = (uint64_t)(mValue * rhs);
    return *this;
}

bool TimePoint::operator >= (const TimePoint& rhs) const
{
    return mValue >= rhs.mValue;
}

bool TimePoint::operator != (const TimePoint& rhs) const
{
    return mValue != rhs.mValue;
}

#endif // PLATFORM
