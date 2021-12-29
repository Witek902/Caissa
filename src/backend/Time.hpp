#pragma once

#include "Common.hpp"

#if defined(_MSC_VER)
#define NOMINMAX
#include <Windows.h>
#endif

class TimePoint
{
public:

	TimePoint() = default;
	TimePoint(LARGE_INTEGER value) : mValue(value) { }

	float ToSeconds() const;

	bool IsValid() const;

	static TimePoint Invalid();
	static TimePoint GetCurrent();
	static TimePoint FromSeconds(float t);

	TimePoint operator - (const TimePoint& rhs) const;
	TimePoint operator + (const TimePoint& rhs) const;

	bool operator >= (const TimePoint& rhs) const;

private:
	LARGE_INTEGER mValue;

	static const LARGE_INTEGER sFreq;
	static const float sPeriod;
};