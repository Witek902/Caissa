#pragma once

#include "DynArray.hpp"

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

// Growable null-terminated string, for text of unbounded length such as PGN.
class String
{
public:
    String() = default;

    String(const char* str) { Append(str); }

    INLINE const char* CStr() const { return mData.Empty() ? "" : mData.Data(); }
    INLINE operator const char* () const { return CStr(); }

    INLINE uint32_t Length() const { return mData.Empty() ? 0 : mData.Size() - 1; }
    INLINE bool Empty() const { return Length() == 0; }

    INLINE char operator[](uint32_t index) const { ASSERT(index < Length()); return mData[index]; }

    void Clear()
    {
        mData.Clear();
    }

    void Append(char c)
    {
        if (mData.Empty())
        {
            mData.PushBack(c);
        }
        else
        {
            mData.Back() = c;
        }
        mData.PushBack(0);
    }

    void Append(const char* str)
    {
        Append(str, (uint32_t)strlen(str));
    }

    void Append(const char* str, uint32_t length)
    {
        const uint32_t oldLength = Length();
        mData.Resize(oldLength + length + 1);
        memcpy(mData.Data() + oldLength, str, length);
        mData[oldLength + length] = 0;
    }

    void AppendFormat(const char* format, ...)
    {
        va_list args;
        va_start(args, format);
        va_list argsCopy;
        va_copy(argsCopy, args);
        const int length = vsnprintf(nullptr, 0, format, argsCopy);
        va_end(argsCopy);
        if (length > 0)
        {
            const uint32_t oldLength = Length();
            mData.Resize(oldLength + (uint32_t)length + 1);
            vsnprintf(mData.Data() + oldLength, (size_t)length + 1, format, args);
        }
        va_end(args);
    }

    bool operator == (const char* str) const { return strcmp(CStr(), str) == 0; }
    bool operator != (const char* str) const { return strcmp(CStr(), str) != 0; }
    bool operator == (const String& other) const { return strcmp(CStr(), other.CStr()) == 0; }
    bool operator != (const String& other) const { return strcmp(CStr(), other.CStr()) != 0; }

private:
    // includes the null terminator when not empty
    DynArray<char> mData;
};
