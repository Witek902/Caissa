#pragma once

#include "../Common.hpp"

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

// Null-terminated string stored in place, for short text such as FEN or move notation.
// Text that does not fit is truncated.
template<uint32_t Capacity>
class FixedString
{
public:
    FixedString() { mData[0] = 0; }

    FixedString(const char* str) { mData[0] = 0; Append(str); }

    INLINE const char* CStr() const { return mData; }
    INLINE operator const char* () const { return mData; }

    INLINE uint32_t Length() const { return mLength; }
    INLINE bool Empty() const { return mLength == 0; }

    INLINE char operator[](uint32_t index) const { ASSERT(index < mLength); return mData[index]; }

    void Clear()
    {
        mLength = 0;
        mData[0] = 0;
    }

    void Append(char c)
    {
        ASSERT(mLength + 1 < Capacity);
        if (mLength + 1 < Capacity)
        {
            mData[mLength++] = c;
            mData[mLength] = 0;
        }
    }

    void Append(const char* str)
    {
        while (*str)
        {
            Append(*str++);
        }
    }

    void AppendFormat(const char* format, ...)
    {
        va_list args;
        va_start(args, format);
        const int written = vsnprintf(mData + mLength, Capacity - mLength, format, args);
        va_end(args);
        ASSERT(written >= 0 && mLength + written < Capacity);
        if (written > 0)
        {
            mLength = Min(mLength + (uint32_t)written, Capacity - 1);
        }
    }

    bool operator == (const char* str) const { return strcmp(mData, str) == 0; }
    bool operator != (const char* str) const { return strcmp(mData, str) != 0; }

    template<uint32_t OtherCapacity>
    bool operator == (const FixedString<OtherCapacity>& other) const { return strcmp(mData, other.CStr()) == 0; }
    template<uint32_t OtherCapacity>
    bool operator != (const FixedString<OtherCapacity>& other) const { return strcmp(mData, other.CStr()) != 0; }

private:
    uint32_t mLength = 0;
    char mData[Capacity];
};
