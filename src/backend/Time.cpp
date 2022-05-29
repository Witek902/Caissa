#include "Time.hpp"

#if defined(PLATFORM_WINDOWS)

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

TimePoint& TimePoint::operator *= (const double rhs)
{
	mValue.QuadPart = (LONGLONG)(mValue.QuadPart * rhs);
	return *this;
}


bool TimePoint::operator >= (const TimePoint& rhs) const
{
	return mValue.QuadPart >= rhs.mValue.QuadPart;
}

bool TimePoint::operator != (const TimePoint& rhs) const
{
	return mValue.QuadPart != rhs.mValue.QuadPart;
}

#elif defined(PLATFORM_LINUX)

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

bool TimePoint::operator >= (const TimePoint& rhs) const
{
	return mValue >= rhs.mValue;
}

#endif // PLATFORM

