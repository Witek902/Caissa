#pragma once

#include "../Common.hpp"

// Unstable in-place sort (quicksort with insertion sort for short ranges).
template<typename T, typename Less>
void Sort(T* begin, T* end, const Less& less)
{
    while (end - begin > 16)
    {
        // median of three as the pivot
        T* mid = begin + (end - begin) / 2;
        if (less(*mid, *begin)) Swap(*mid, *begin);
        if (less(*(end - 1), *mid)) Swap(*(end - 1), *mid);
        if (less(*mid, *begin)) Swap(*mid, *begin);
        const T pivot = *mid;

        T* i = begin;
        T* j = end - 1;
        for (;;)
        {
            while (less(*i, pivot)) ++i;
            while (less(pivot, *j)) --j;
            if (i >= j) break;
            Swap(*i, *j);
            ++i;
            --j;
        }

        // recurse into the smaller part, loop on the larger one
        T* split = j + 1;
        if (split - begin < end - split)
        {
            Sort(begin, split, less);
            begin = split;
        }
        else
        {
            Sort(split, end, less);
            end = split;
        }
    }

    for (T* i = begin + 1; i < end; ++i)
    {
        T value = MoveTemp(*i);
        T* j = i;
        for (; j > begin && less(value, *(j - 1)); --j)
        {
            *j = MoveTemp(*(j - 1));
        }
        *j = MoveTemp(value);
    }
}
