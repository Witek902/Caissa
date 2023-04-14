#include "Common.hpp"
#include "ThreadPool.hpp"
#include "TrainerCommon.hpp"

#include "../backend/Math.hpp"
#include "../backend/Material.hpp"
#include "../backend/Waitable.hpp"
#include "../backend/Evaluate.hpp"
#include "../backend/Endgame.hpp"
#include "../backend/Tablebase.hpp"

#include <filesystem>

static_assert(sizeof(PositionEntry) == 32, "Invalid PositionEntry size");

bool TrainingDataLoader::Init(const std::string& trainingDataPath)
{
    uint64_t totalDataSize = 0;

    mCDF.push_back(0.0);

    for (const auto& path : std::filesystem::directory_iterator(trainingDataPath))
    {
        const std::string& fileName = path.path().string();
        auto fileStream = std::make_unique<FileInputStream>(fileName.c_str());

        uint64_t fileSize = fileStream->GetSize();
        totalDataSize += fileSize;

        if (fileStream->IsOpen() && fileSize > 0)
        {
            std::cout << "Using " << fileName << std::endl;

            InputFileContext& ctx = mContexts.emplace_back();
            ctx.fileStream = std::move(fileStream);
            ctx.fileName = fileName;
            ctx.fileSize = fileSize;

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

    return !mContexts.empty();
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

bool TrainingDataLoader::FetchNextPosition(std::mt19937& gen, PositionEntry& outEntry, Position& outPosition)
{
    std::uniform_real_distribution<double> distr;
    const double u = distr(gen);
    const uint32_t fileIndex = SampleInputFileIndex(u);
    ASSERT(fileIndex < mContexts.size());

    if (fileIndex >= mContexts.size())
        return false;

    return mContexts[fileIndex].FetchNextPosition(gen, outEntry, outPosition);
}

bool TrainingDataLoader::InputFileContext::FetchNextPosition(std::mt19937& gen, PositionEntry& outEntry, Position& outPosition)
{
    for (;;)
    {
        if (!fileStream->Read(&outEntry, sizeof(PositionEntry)))
        {
            // if read failed, reset to the file beginning and try again

            if (fileStream->GetPosition() > 0)
            {
                std::cout << "Resetting stream " << fileName << std::endl;
                fileStream->SetPosition(0);
            }
            else
            {
                return false;
            }

            if (!fileStream->Read(&outEntry, sizeof(PositionEntry)))
            {
                return false;
            }
        }

        if (outEntry.score == (uint32_t)Game::Score::Unknown)
        {
            continue;
        }

        // skip based half-move counter
        {
            const float hmcSkipProb = 0.25f + 0.75f * (float)outEntry.pos.halfMoveCount / 100.0f;
            std::bernoulli_distribution skippingDistr(hmcSkipProb);
            if (skippingDistr(gen))
                continue;
        }

        // skip based on piece count
        {
            const int32_t numPieces = outEntry.pos.occupied.Count();
            const float pieceCountSkipProb = Sqr(static_cast<float>(numPieces - 20) / 40.0f);
            std::bernoulli_distribution skippingDistr(pieceCountSkipProb);
            if (skippingDistr(gen))
                continue;
        }

        VERIFY(UnpackPosition(outEntry.pos, outPosition, false));
        ASSERT(outPosition.IsValid());

        // skip based on PSQT score
        {
            const ScoreType psqtScore = Evaluate(outPosition, nullptr, false);

            const int32_t minRange = 512;
            const int32_t maxRange = 1024;

            if (std::abs(psqtScore) > maxRange)
            {
                continue;
            }
            if (std::abs(psqtScore) > minRange)
            {
                const float psqtScoreSkipProb = static_cast<float>(std::abs(psqtScore) - minRange) / static_cast<float>(maxRange - minRange);
                std::bernoulli_distribution skippingDistr(psqtScoreSkipProb);
                if (skippingDistr(gen))
                    continue;
            }
        }

        // skip based on kings placement (prefer king on further ranks)
        {
            const float whiteKingProb = 0.5f + 0.5f * (float)outPosition.Whites().GetKingSquare().Rank() / 7.0f;
            const float blackKingProb = 0.5f + 0.5f * (7 - (float)outPosition.Blacks().GetKingSquare().Rank()) / 7.0f;
            std::bernoulli_distribution skippingDistr(std::min(whiteKingProb, blackKingProb));
            if (skippingDistr(gen))
                continue;
        }

        // TODO more skipping techniques:
        // - eval / game outcome match

        return true;
    }
}
