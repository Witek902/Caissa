#include "Time.hpp"

namespace
{
	static LARGE_INTEGER GetCounterFrequency()
	{
		LARGE_INTEGER freq;
		QueryPerformanceFrequency(&freq);
		return freq;
	}

	static float GetCounterPeriod()
	{
		LARGE_INTEGER freq;
		QueryPerformanceFrequency(&freq);
		return 1.0f / static_cast<float>(freq.QuadPart);
	}
}

const LARGE_INTEGER TimePoint::sFreq = GetCounterFrequency();
const float TimePoint::sPeriod = GetCounterPeriod();

float TimePoint::ToSeconds() const
{
	return static_cast<float>(mValue.QuadPart) * sPeriod;
}

bool TimePoint::IsValid() const
{
	return mValue.QuadPart >= 0;
}

TimePoint TimePoint::Invalid()
{
	LARGE_INTEGER value = {};
	value.QuadPart = -1;
	return { value };
}

TimePoint TimePoint::GetCurrent()
{
	LARGE_INTEGER value = {};
	QueryPerformanceCounter(&value);
	return { value };
}

TimePoint TimePoint::FromSeconds(float t)
{
	LARGE_INTEGER value = {};
	value.QuadPart = static_cast<LONGLONG>(sFreq.QuadPart * t);
	return { value };
}

TimePoint TimePoint::operator - (const TimePoint& rhs) const
{
	LARGE_INTEGER p = {};
	p.QuadPart = mValue.QuadPart - rhs.mValue.QuadPart;
	return { p };
}

TimePoint TimePoint::operator + (const TimePoint& rhs) const
{
	LARGE_INTEGER p = {};
	p.QuadPart = mValue.QuadPart + rhs.mValue.QuadPart;
	return { p };
}

bool TimePoint::operator >= (const TimePoint& rhs) const
{
	return mValue.QuadPart >= rhs.mValue.QuadPart;
}