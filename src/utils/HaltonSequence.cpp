#include "HaltonSequence.hpp"

#include <random>

HaltonSequence::HaltonSequence()
{
    mDimensions = 0;
    ppm = nullptr;
}

HaltonSequence::~HaltonSequence()
{
    ClearPermutation();
}

void HaltonSequence::ClearPermutation()
{
    if (ppm)
    {
        for (uint32_t i = 0; i < mDimensions; i++)
        {
            delete[] * (ppm + i);
            *(ppm + i) = nullptr;
        }
        delete[] ppm;
        ppm = nullptr;
    }
}

void HaltonSequence::InitPowerBuffer()
{
    for (uint32_t d = 0; d < mDimensions; d++)
    {
        for (uint8_t j = 0; j < Width; j++)
        {
            if (j == 0)
            {
                mPowerBuffer[d][j] = mBase[d];
            }
            else
            {
                mPowerBuffer[d][j] = mPowerBuffer[d][j - 1] * mBase[d];
            }
        }
    }

    for (auto &v : rnd)
    {
        std::fill(v.begin(), v.end(), 0.0);
    }

    for (auto &v : digit)
    {
        std::fill(v.begin(), v.end(), 0);
    }
}

void HaltonSequence::InitExpansion()
{
    for (uint32_t i = 0; i < mDimensions; i++)
    {
        uint64_t n = mStarts[i] - 1;
        int8_t j = 0;
        while (n > 0)
        {
            digit[i][j] = n % mBase[i];
            n = n / mBase[i];
            j++;
        }
        j--;
        while (j >= 0)
        {
            uint64_t d = digit[i][j];
            d = Permute(i, j);
            rnd[i][j] = rnd[i][j + 1] + d * 1.0 / mPowerBuffer[i][j];
            j--;
        }
    }
}

void HaltonSequence::NextSample()
{
    for (uint32_t i = 0; i < mDimensions; i++)
    {
        int8_t j = 0;
        while (digit[i][j] + 1 >= mBase[i])
        {
            j++;
        }
        digit[i][j]++;
        uint64_t d = digit[i][j];
        d = Permute(i, j);
        rnd[i][j] = rnd[i][j + 1] + d * 1.0 / mPowerBuffer[i][j];

        for (j = j - 1; j >= 0; j--)
        {
            digit[i][j] = 0;
            d = 0;
            d = Permute(i, j);
            rnd[i][j] = rnd[i][j + 1] + d * 1.0 / mPowerBuffer[i][j];
        }
    }
}

void HaltonSequence::NextSampleLeap()
{
    const uint32_t leapSize = 727; // 129th prime (not used as any base)

    for (uint32_t i = 0; i < leapSize; ++i)
    {
        NextSample();
    }
}

uint64_t HaltonSequence::Permute(uint32_t i, uint8_t j)
{
    return *(*(ppm + i) + digit[i][j]);
}

void HaltonSequence::InitPermutation()
{
    std::random_device rd;

    ppm = new uint64_t*[mDimensions];

    for (uint32_t i = 0; i < mDimensions; i++)
    {
        *(ppm + i) = new uint64_t[mBase[i]];
        for (uint64_t j = 0; j < mBase[i]; j++)
        {
            *(*(ppm + i) + j) = j;
        }

        for (uint64_t j = 1; j < mBase[i]; j++)
        {
            const double seed = std::uniform_real_distribution<double>(0.0, 1.0)(rd);
            uint64_t tmp = (uint64_t)floor(seed * mBase[i]);
            if (tmp != 0)
            {
                uint64_t k = *(*(ppm + i) + j);
                *(*(ppm + i) + j) = *(*(ppm + i) + tmp);
                *(*(ppm + i) + tmp) = k;
            }
        }
    }
}

void HaltonSequence::InitPrimes()
{
    int64_t n = mDimensions;
    uint32_t prime = 1;
    uint32_t m = 0;
    do
    {
        prime++;
        mBase[m++] = prime;
        n--;
        for (uint64_t i = 2; i <= sqrt(prime); i++)
        {
            if (prime % i == 0)
            {
                n++;
                m--;
                break;
            }
        }
    } while (n > 0);
}

void HaltonSequence::InitStart()
{
    std::random_device rd;

    for (uint32_t i = 0; i < mDimensions; i++)
    {
        double r = std::uniform_real_distribution<double>(0.0, 1.0)(rd);
        const uint64_t base = mBase[i];

        uint64_t z = 0;
        uint64_t b = base;
        while (r > 1.0e-16)
        {
            uint64_t cnt = 0;
            if (r >= 1.0 / b)
            {
                cnt = (uint64_t)floor(r * b);
                r = r - cnt * 1.0 / b;
                z += cnt * b / base;
            }
            b *= base;
        }

        mStarts[i] = z;
    }
}

void HaltonSequence::Initialize(uint32_t dim)
{
    ClearPermutation();

    mDimensions = dim;

    rnd.resize(mDimensions, std::vector<double>(Width));
    digit.resize(mDimensions, std::vector<uint64_t>(Width));
    mPowerBuffer.resize(mDimensions, std::vector<uint64_t>(Width));
    mStarts.resize(mDimensions);
    mBase.resize(mDimensions);

    if (mDimensions > 0)
    {
        InitPrimes();
        InitStart();
        InitPowerBuffer();
        InitPermutation();
        InitExpansion();
    }
}