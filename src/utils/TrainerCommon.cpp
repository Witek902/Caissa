#include "TrainerCommon.hpp"
#include "../backend/Math.hpp"
#include "../backend/Evaluate.hpp"
#include "../backend/Endgame.hpp"
#include "../backend/NeuralNetworkEvaluator.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>

static_assert(sizeof(PositionEntry) == 32, "Invalid PositionEntry size");

// target share (%) of training positions per pair of piece counts, halfway between the selfplay data
// and the piece counts where LTC games are decided; pairs smooth out the odd/even pattern left by exchanges
static constexpr float c_PiecePairTargetShare[] =
{
    0.0f, 0.0f, 1.6f, 5.6f, 7.6f, 9.3f, 9.9f, 9.9f, 9.6f, 9.1f, 8.6f, 8.3f, 7.4f, 6.3f, 4.3f, 2.1f, 0.5f,
};

bool TrainingDataLoader::Init(std::mt19937& gen, const Options& options, const std::string& trainingDataPath)
{
    uint64_t totalDataSize = 0;

    mOptions = options;
    mPiecePairKeepProb.fill(1.0f);

    mDataPath = std::filesystem::absolute(trainingDataPath).string();
    mCDF.push_back(0.0);

    for (const auto& path : std::filesystem::directory_iterator(trainingDataPath))
    {
        const std::string& fileName = path.path().string();
        auto fileStream = std::make_unique<FileInputStream>(fileName.c_str());

        uint64_t fileSize = fileStream->GetSize();
        totalDataSize += fileSize;

        if (fileStream->IsOpen() && fileSize > sizeof(PositionEntry))
        {
            InputFileContext& ctx = mContexts.emplace_back();
            ctx.mutex = std::make_unique<std::mutex>();
            ctx.fileStream = std::move(fileStream);
            ctx.fileName = fileName;
            ctx.fileSize = fileSize;

            // Seek to random location so that each stream starts at different position.
            {
                const uint64_t numEntries = fileSize / sizeof(PositionEntry);
                std::uniform_int_distribution<uint64_t> distr(0, numEntries - 1);
                const uint64_t entryIndex = distr(gen);
                ctx.fileStream->SetPosition(entryIndex * sizeof(PositionEntry));
            }

            // Set a small, random skipping probability.
            // The idea is to have each stream running at different rates
            // so there's lower chance of generating similar batches from different streams.
            // Basically, it's another layer of data shuffling.
            {
                std::uniform_real_distribution<float> distr(0.0f, 0.01f);
                ctx.skippingProbability = distr(gen);
            }

            mCDF.push_back((double)totalDataSize);
        }
        else
        {
            std::cout << "ERROR: Failed to load selfplay data file: " << fileName << std::endl;
        }
    }

    if (totalDataSize > 0)
    {
        // normalize
        for (double& v : mCDF)
        {
            v /= static_cast<double>(totalDataSize);
        }
    }

    if (mContexts.empty())
        return false;

    mTotalDataSize = totalDataSize;

    if (mOptions.pieceCountTarget)
        InitPiecePairKeepProb(gen);

    return true;
}

void TrainingDataLoader::InitPiecePairKeepProb(std::mt19937& gen)
{
    static_assert(std::size(c_PiecePairTargetShare) == NumPiecePairs);

    const uint32_t numSamples = 1000000;
    uint32_t counts[NumPiecePairs] = {};
    for (uint32_t i = 0; i < numSamples; ++i)
    {
        PositionEntry entry;
        Position pos;
        if (!FetchNextPosition(gen, entry, pos, UINT64_MAX))
            return;
        counts[entry.pos.occupied.Count() / 2]++;
    }

    float ratios[NumPiecePairs];
    float maxRatio = 0.0f;
    for (uint32_t i = 0; i < NumPiecePairs; ++i)
    {
        ratios[i] = counts[i] > 0 ? c_PiecePairTargetShare[i] / static_cast<float>(counts[i]) : 0.0f;
        maxRatio = std::max(maxRatio, ratios[i]);
    }

    std::cout << "Piece count keep probability:";
    for (uint32_t i = 0; i < NumPiecePairs; ++i)
    {
        if (counts[i] > 0)
            mPiecePairKeepProb[i] = ratios[i] / maxRatio;
        if (c_PiecePairTargetShare[i] > 0.0f)
            std::cout << " " << 2 * i << "-" << 2 * i + 1 << ": " << mPiecePairKeepProb[i];
    }
    std::cout << std::endl;
}

uint32_t TrainingDataLoader::SampleInputFileIndex(double u) const
{
    uint32_t low = 0u;
    uint32_t high = static_cast<uint32_t>(mContexts.size());

    // binary search
    while (low < high)
    {
        uint32_t mid = (low + high) / 2u;
        if (u >= mCDF[mid])
        {
            low = mid + 1u;
        }
        else
        {
            high = mid;
        }
    }

    return low - 1u;
}

bool TrainingDataLoader::FetchNextPosition(std::mt19937& gen, PositionEntry& outEntry, Position& outPosition, uint64_t kingBucketMask) const
{
    std::uniform_real_distribution<double> distr;
    const double u = distr(gen);
    const uint32_t fileIndex = SampleInputFileIndex(u);
    ASSERT(fileIndex < mContexts.size());

    if (fileIndex >= mContexts.size())
        return false;

    return mContexts[fileIndex].FetchNextPosition(gen, mOptions, mPiecePairKeepProb, outEntry, outPosition, kingBucketMask);
}

bool TrainingDataLoader::InputFileContext::FetchNextPosition(std::mt19937& gen, const Options& options, const PiecePairKeepProb& piecePairKeepProb, PositionEntry& outEntry, Position& outPosition, uint64_t kingBucketMask) const
{
    for (;;)
    {
        {
            std::scoped_lock lock(*mutex);
            if (!fileStream->Read(&outEntry, sizeof(PositionEntry)))
            {
                // if read failed, reset to the file beginning and try again

                if (fileStream->GetPosition() > 0)
                {
                    fileStream->SetPosition(0);
                }
                else
                {
                    std::cout << "ERROR: Failed to read from stream " << fileName << std::endl;
                    return false;
                }

                if (!fileStream->Read(&outEntry, sizeof(PositionEntry)))
                {
                    std::cout << "ERROR: Failed to read from stream " << fileName << " after reset" << std::endl;
                    return false;
                }
            }
        }

        // skip invalid scores
        if (outEntry.score >= CheckmateValue || outEntry.score <= -CheckmateValue)
            continue;

        // skip positions with very high score and matching WDL score
        const int32_t WdlSkippingThreshold = 2000;
        if ((outEntry.score > WdlSkippingThreshold && outEntry.wdlScore == 1) ||
            (outEntry.score < -WdlSkippingThreshold && outEntry.wdlScore == 2))
            continue;

        // partially skip decisive positions whose game result agrees with the score, the net learns little from them
        if (options.decisiveSkip &&
            ((outEntry.score > 0 && outEntry.wdlScore == 1) || (outEntry.score < 0 && outEntry.wdlScore == 2)))
        {
            const float decisiveSkipProb = std::clamp(static_cast<float>(std::abs(outEntry.score) - 400) / 800.0f, 0.0f, 0.75f);
            if (decisiveSkipProb > 0.0f && std::bernoulli_distribution(decisiveSkipProb)(gen))
                continue;
        }

        // constant skipping
        {
            std::bernoulli_distribution skippingDistr(skippingProbability);
            if (skippingDistr(gen))
                continue;
        }

        VERIFY(UnpackPosition(outEntry.pos, outPosition, false));
        ASSERT(outPosition.IsValid());

        // filter by king bucket
        if (kingBucketMask != UINT64_MAX)
        {
            uint32_t whiteKingSide, blackKingSide;
            uint32_t whiteKingBucket, blackKingBucket;
            GetKingSideAndBucket(outPosition.Whites().GetKingSquare(), whiteKingSide, whiteKingBucket);
            GetKingSideAndBucket(outPosition.Blacks().GetKingSquare().FlippedRank(), blackKingSide, blackKingBucket);

            if ((((1ull << whiteKingBucket) & kingBucketMask) == 0ull) && (((1ull << blackKingBucket) & kingBucketMask) == 0ull))
                continue;
        }
        else
        {
            // results close to the 50-move rule depend on the half-move counter, which the net doesn't see
            {
                const float hmcSkipProb = options.hmcSkipFrom20 ?
                    std::clamp(static_cast<float>(outEntry.pos.halfMoveCount - 20) / 80.0f, 0.0f, 1.0f) :
                    sqrtf(static_cast<float>(outEntry.pos.halfMoveCount) / 100.0f);
                if (hmcSkipProb > 0.0f && std::bernoulli_distribution(hmcSkipProb)(gen))
                    continue;
            }

            const int32_t numPieces = outEntry.pos.occupied.Count();

            // skip early moves
            if (outEntry.pos.moveCount <= 12 && numPieces > 24)
                continue;

            // skip based on piece count
            {
                if (numPieces <= 3)
                    continue;

                if (CheckInsufficientMaterial(outPosition))
                    continue;

                // skip recognized endgames
                int32_t endgameScore = 0;
                if (EvaluateEndgame(outPosition, endgameScore))
                    continue;

                const float keepProb = options.pieceCountTarget ?
                    piecePairKeepProb[numPieces / 2] :
                    1.0f - Sqr(static_cast<float>(numPieces - 22) / 30.0f);
                if (keepProb < 1.0f && !std::bernoulli_distribution(keepProb)(gen))
                    continue;
            }
        }

        return true;
    }
}
