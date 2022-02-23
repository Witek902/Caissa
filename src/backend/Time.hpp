#pragma once

#include "Common.hpp"

#if defined(PLATFORM_WINDOWS)
	#define WIN32_LEAN_AND_MEAN
	#define NOMINMAX
	#include <Windows.h>
#elif defined(PLATFORM_LINUX)
	#include <time.h>
#endif

class TimePoint
{
public:

	TimePoint() = default;

#if defined(PLATFORM_WINDOWS)
	TimePoint(LARGE_INTEGER value) : mValue(value) { }
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

	bool operator >= (const TimePoint& rhs) const;

private:

#ifdef PLATFORM_WINDOWS
	LARGE_INTEGER mValue;
	static const LARGE_INTEGER sFreq;
	static const float sPeriod;
#elif defined(PLATFORM_LINUX)
	uint64_t mValue; // nanoseconds
#endif // PLATFORM

};
