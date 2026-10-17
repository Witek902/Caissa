#pragma once

#include "../../backend/Memory.hpp"

#include <vector>
#include <memory>

// https://stackoverflow.com/a/8545389
template <typename T, std::size_t N = 16>
class AlignmentAllocator
{
public:
    typedef T value_type;
    typedef std::size_t size_type;
    typedef std::ptrdiff_t difference_type;
    typedef T* pointer;
    typedef const T* const_pointer;
    typedef T& reference;
    typedef const T& const_reference;

public:
    AlignmentAllocator() throw () { }

    template <typename T2>
    AlignmentAllocator(const AlignmentAllocator<T2, N>&) throw () { }

    ~AlignmentAllocator() throw () { }

    pointer address(reference r) { return &r; }
    const_pointer address(const_reference r) const { return &r; }
    pointer allocate(size_type n) { return (pointer)AlignedMalloc(n * sizeof(value_type), N); }

    void deallocate(pointer p, size_type) { AlignedFree(p); }
    void construct(pointer p, const value_type& wert) { new (p) value_type(wert); }
    void destroy(pointer p) { p->~value_type(); }
    size_type max_size() const throw () { return size_type(-1) / sizeof(value_type); }

    template <typename T2>
    struct rebind { typedef AlignmentAllocator<T2, N> other; };

    bool operator != (const AlignmentAllocator<T, N>& other) const { return !(*this == other); }
    bool operator == (const AlignmentAllocator<T, N>&) const { return true; }
};

namespace nn {

using Values = std::vector<float, AlignmentAllocator<float, 32>>;

struct Gradients;

struct WeightsStorage
{
public:
    WeightsStorage(uint32_t inputSize, uint32_t outputSize, uint32_t numVariants);

    void Init(uint32_t numActiveInputs, float bias = 0.0f);

    struct Stats
    {
        float minWeight, maxWeight, weightAvg, weightStdDev;
        float minBias, maxBias, biasAvg, biasStdDev;
    };

    Stats ComputeStats() const;

    uint32_t m_inputSize = 0;
    uint32_t m_outputSize = 0;
    bool m_isSparse = false;

    bool m_updateWeights = true;
    Values m_weightsMask;

    float m_weightsRange = 10.0f;
    float m_biasRange = 10.0f;

    struct Variant
    {
        Values m_weights;

        // used for learning
        Values m_gradientMoment1;
        Values m_gradientMoment2;
    };

    std::vector<Variant> m_variants;
};

using WeightsStoragePtr = std::shared_ptr<WeightsStorage>;

} // namespace nn
