#pragma once

#include "../Memory.hpp"

// Dynamic array (like std::vector), based on nfengine's DynArray.
// begin()/end() are raw pointers, so it works with range-based for and STL algorithms.
template<typename T>
class DynArray
{
public:
    DynArray() = default;

    // value-initializes the elements (zeroes for fundamental types)
    explicit DynArray(uint32_t size) { Resize(size); }

    DynArray(uint32_t size, const T& value) { Resize(size, value); }

    DynArray(const T* elements, uint32_t count) { Assign(elements, count); }

    DynArray(const DynArray& other) { Assign(other.mElements, other.mSize); }

    DynArray(DynArray&& other) noexcept
        : mElements(other.mElements), mSize(other.mSize), mCapacity(other.mCapacity)
    {
        other.mElements = nullptr;
        other.mSize = 0;
        other.mCapacity = 0;
    }

    ~DynArray()
    {
        Clear();
        AlignedFree(mElements);
    }

    DynArray& operator = (const DynArray& other)
    {
        if (this != &other)
        {
            Assign(other.mElements, other.mSize);
        }
        return *this;
    }

    DynArray& operator = (DynArray&& other) noexcept
    {
        if (this != &other)
        {
            Clear();
            AlignedFree(mElements);
            mElements = other.mElements;
            mSize = other.mSize;
            mCapacity = other.mCapacity;
            other.mElements = nullptr;
            other.mSize = 0;
            other.mCapacity = 0;
        }
        return *this;
    }

    INLINE uint32_t Size() const { return mSize; }
    INLINE uint32_t Capacity() const { return mCapacity; }
    INLINE bool Empty() const { return mSize == 0; }

    INLINE T* Data() { return mElements; }
    INLINE const T* Data() const { return mElements; }

    INLINE T& operator[](size_t index) { ASSERT(index < mSize); return mElements[index]; }
    INLINE const T& operator[](size_t index) const { ASSERT(index < mSize); return mElements[index]; }

    INLINE T& Front() { ASSERT(mSize > 0); return mElements[0]; }
    INLINE const T& Front() const { ASSERT(mSize > 0); return mElements[0]; }
    INLINE T& Back() { ASSERT(mSize > 0); return mElements[mSize - 1]; }
    INLINE const T& Back() const { ASSERT(mSize > 0); return mElements[mSize - 1]; }

    INLINE T* begin() { return mElements; }
    INLINE T* end() { return mElements + mSize; }
    INLINE const T* begin() const { return mElements; }
    INLINE const T* end() const { return mElements + mSize; }

    // destroys all elements, keeps the allocated memory
    void Clear()
    {
        for (uint32_t i = 0; i < mSize; ++i)
        {
            mElements[i].~T();
        }
        mSize = 0;
    }

    void Reserve(uint32_t capacity)
    {
        if (capacity > mCapacity)
        {
            Reallocate(capacity);
        }
    }

    void Resize(uint32_t size)
    {
        ShrinkTo(size);
        Reserve(size);
        for (uint32_t i = mSize; i < size; ++i)
        {
            PLACEMENT_NEW(mElements + i) T();
        }
        mSize = size;
    }

    void Resize(uint32_t size, const T& value)
    {
        ShrinkTo(size);
        Reserve(size);
        for (uint32_t i = mSize; i < size; ++i)
        {
            PLACEMENT_NEW(mElements + i) T(value);
        }
        mSize = size;
    }

    void Assign(const T* elements, uint32_t count)
    {
        Clear();
        Reserve(count);
        for (uint32_t i = 0; i < count; ++i)
        {
            PLACEMENT_NEW(mElements + i) T(elements[i]);
        }
        mSize = count;
    }

    void PushBack(const T& element)
    {
        EmplaceBack(element);
    }

    void PushBack(T&& element)
    {
        EmplaceBack(MoveTemp(element));
    }

    template<typename... Args>
    T& EmplaceBack(Args&&... args)
    {
        if (mSize == mCapacity)
        {
            return EmplaceBackWithGrowth(Forward<Args>(args)...);
        }
        PLACEMENT_NEW(mElements + mSize) T(Forward<Args>(args)...);
        return mElements[mSize++];
    }

    void PopBack()
    {
        ASSERT(mSize > 0);
        mElements[--mSize].~T();
    }

    // removes elements in range [first, last), keeps the order of the remaining ones
    void Erase(uint32_t first, uint32_t last)
    {
        ASSERT(first <= last && last <= mSize);
        const uint32_t count = last - first;
        for (uint32_t i = first; i + count < mSize; ++i)
        {
            mElements[i] = MoveTemp(mElements[i + count]);
        }
        ShrinkTo(mSize - count);
    }

    void Erase(uint32_t index)
    {
        Erase(index, index + 1);
    }

    void Insert(uint32_t index, const T& element)
    {
        ASSERT(index <= mSize);
        T copy(element);
        EmplaceBack();
        for (uint32_t i = mSize - 1; i > index; --i)
        {
            mElements[i] = MoveTemp(mElements[i - 1]);
        }
        mElements[index] = MoveTemp(copy);
    }

    // returns index of the first element equal to the value, or UINT32_MAX
    uint32_t Find(const T& value) const
    {
        for (uint32_t i = 0; i < mSize; ++i)
        {
            if (mElements[i] == value) return i;
        }
        return UINT32_MAX;
    }

    bool Contains(const T& value) const
    {
        return Find(value) != UINT32_MAX;
    }

    bool operator == (const DynArray& other) const
    {
        if (mSize != other.mSize) return false;
        for (uint32_t i = 0; i < mSize; ++i)
        {
            if (!(mElements[i] == other.mElements[i])) return false;
        }
        return true;
    }

    bool operator != (const DynArray& other) const
    {
        return !(*this == other);
    }

private:

    // kept out of line like std::vector's reallocation, so callers only inline the fast path
    template<typename... Args>
    NO_INLINE T& EmplaceBackWithGrowth(Args&&... args)
    {
        // construct the new element before moving the old ones, the arguments may refer to them
        const uint32_t newCapacity = NextCapacity(mSize + 1);
        T* newElements = Allocate(newCapacity);
        PLACEMENT_NEW(newElements + mSize) T(Forward<Args>(args)...);
        MoveElements(newElements, newCapacity);
        return mElements[mSize++];
    }

    static T* Allocate(uint32_t capacity)
    {
        constexpr size_t alignment = alignof(T) > sizeof(void*) ? alignof(T) : sizeof(void*);
        T* ptr = static_cast<T*>(AlignedMalloc(sizeof(T) * capacity, alignment));
        ASSERT(ptr != nullptr);
        return ptr;
    }

    uint32_t NextCapacity(uint32_t minCapacity) const
    {
        // grow by 50%
        const uint32_t grown = mCapacity + mCapacity / 2;
        return grown > minCapacity ? grown : minCapacity;
    }

    // moves the elements into a new buffer and frees the old one
    void MoveElements(T* newElements, uint32_t newCapacity)
    {
        for (uint32_t i = 0; i < mSize; ++i)
        {
            PLACEMENT_NEW(newElements + i) T(MoveTemp(mElements[i]));
            mElements[i].~T();
        }
        AlignedFree(mElements);
        mElements = newElements;
        mCapacity = newCapacity;
    }

    NO_INLINE void Reallocate(uint32_t capacity)
    {
        MoveElements(Allocate(capacity), capacity);
    }

    void ShrinkTo(uint32_t size)
    {
        for (uint32_t i = size; i < mSize; ++i)
        {
            mElements[i].~T();
        }
        if (size < mSize)
        {
            mSize = size;
        }
    }

    T* mElements = nullptr;
    uint32_t mSize = 0;
    uint32_t mCapacity = 0;
};
