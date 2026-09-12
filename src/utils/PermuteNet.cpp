#include "Common.hpp"
#include "TrainerCommon.hpp"

#include "../backend/Endgame.hpp"
#include "../backend/NeuralNetworkEvaluator.hpp"
#include "../backend/PackedNeuralNetwork.hpp"
#include "../backend/Position.hpp"
#include "../backend/PositionUtils.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

// Reorders the accumulator so that pair slots which are rarely active end up in the same 4-input
// group of the sparse L1 layer, which makes more groups entirely zero and lets the sparse path skip
// them. Purely a layout change: the network computes exactly the same values before and after.

namespace {

// A pair slot is the product of accumulator neurons k and k + PairCount, so the two must always move
// together. Both perspectives read the same accumulator, so one permutation covers all of L1.
static constexpr uint32_t PairCount = nn::AccumulatorSize / 2;
static constexpr uint32_t GroupSize = 4;
static constexpr uint32_t NumGroups = nn::L1InputSize / GroupSize;
static constexpr uint32_t RowWords = PairCount / 64;

// One sample is one position seen from one perspective: a bit per pair slot that is non-zero.
using SampleRows = std::vector<uint64_t>;

INLINE bool TestBit(const uint64_t* row, uint32_t slot)
{
    return (row[slot / 64] >> (slot % 64)) & 1ull;
}

// Matches what the engine actually evaluates: the network is never asked about positions in check,
// with fewer than four pieces, or with a recognized endgame.
bool IsEvaluatedAtRuntime(const Position& pos)
{
    if (pos.IsInCheck())
        return false;

    if (pos.GetNumPieces() < 4)
        return false;

    int32_t endgameScore = 0;
    if (EvaluateEndgame(pos, endgameScore))
        return false;

    return true;
}

void AppendSample(SampleRows& rows, const nn::Accumulator& accum)
{
    const size_t base = rows.size();
    rows.resize(base + RowWords, 0ull);

    for (uint32_t k = 0; k < PairCount; ++k)
    {
        const int32_t a = std::clamp<int32_t>(accum.values[k], 0, nn::ActivationRangeScaling);
        const int32_t b = std::clamp<int32_t>(accum.values[k + PairCount], 0, nn::ActivationRangeScaling);
        if (((a * b) >> nn::PairwiseShift) != 0)
        {
            rows[base + k / 64] |= 1ull << (k % 64);
        }
    }
}

// Reads PositionEntry records and records the activation pattern of every position the engine would
// actually evaluate. Returns the number of samples appended.
uint64_t CollectSamples(const char* path, const nn::PackedNeuralNetwork& net, uint64_t maxPositions, SampleRows& rows)
{
    FILE* file = fopen(path, "rb");
    if (!file)
    {
        std::cerr << "Failed to open " << path << std::endl;
        return 0;
    }

    constexpr uint32_t cEntriesPerRead = 4096;
    std::vector<PositionEntry> buffer(cEntriesPerRead);

    uint64_t numAccepted = 0;
    uint64_t numRead = 0;
    for (;;)
    {
        const size_t count = fread(buffer.data(), sizeof(PositionEntry), cEntriesPerRead, file);
        if (count == 0)
            break;

        for (size_t i = 0; i < count && numAccepted < maxPositions; ++i)
        {
            numRead++;

            Position pos;
            if (!UnpackPosition(buffer[i].pos, pos, false) || !pos.IsValid())
                continue;

            if (!IsEvaluatedAtRuntime(pos))
                continue;

            numAccepted++;

            // the side to move fills the low half of the L1 input, the other side the high half
            for (uint32_t side = 0; side < 2; ++side)
            {
                const Color perspective = (Color)(pos.GetSideToMove() ^ side);

                uint16_t features[64];
                const uint32_t numFeatures = PositionToFeaturesVector(pos, features, perspective);

                nn::Accumulator accum;
                accum.Refresh(net.accumulatorWeights, net.accumulatorBiases, numFeatures, features);
                AppendSample(rows, accum);
            }
        }

        if (numAccepted >= maxPositions)
            break;
    }

    fclose(file);

    std::cout << path << ": " << numRead << " entries read, " << numAccepted << " usable" << std::endl;
    return numAccepted;
}

// Fraction of 4-input groups that contain at least one non-zero, averaged over the samples.
double LiveGroupRate(const SampleRows& rows, const std::vector<uint32_t>& perm)
{
    const size_t numSamples = rows.size() / RowWords;
    if (numSamples == 0)
        return 0.0;

    uint64_t live = 0;
    for (size_t s = 0; s < numSamples; ++s)
    {
        const uint64_t* row = rows.data() + s * RowWords;
        for (uint32_t g = 0; g < PairCount / GroupSize; ++g)
        {
            for (uint32_t j = 0; j < GroupSize; ++j)
            {
                if (TestBit(row, perm[g * GroupSize + j]))
                {
                    live++;
                    break;
                }
            }
        }
    }

    // both halves of the L1 input use the same permutation, so the per-perspective rate is the answer
    return (double)live / (double)(numSamples * (PairCount / GroupSize));
}

std::vector<uint64_t> CountActivations(const SampleRows& rows)
{
    std::vector<uint64_t> counts(PairCount, 0);
    const size_t numSamples = rows.size() / RowWords;
    for (size_t s = 0; s < numSamples; ++s)
    {
        const uint64_t* row = rows.data() + s * RowWords;
        for (uint32_t w = 0; w < RowWords; ++w)
        {
            uint64_t bits = row[w];
            while (bits)
            {
                counts[w * 64 + FirstBitSet(bits)]++;
                bits &= bits - 1;
            }
        }
    }
    return counts;
}

// Integral's heuristic: busiest slots first, so the quiet ones collect at the end and their groups
// are almost always dead.
std::vector<uint32_t> PermutationBySortedActivation(const SampleRows& rows)
{
    const std::vector<uint64_t> counts = CountActivations(rows);

    std::vector<uint32_t> perm(PairCount);
    std::iota(perm.begin(), perm.end(), 0u);
    std::stable_sort(perm.begin(), perm.end(), [&counts](uint32_t a, uint32_t b)
    {
        return counts[a] > counts[b];
    });
    return perm;
}

// Stockfish's heuristic: fill the output slots in order, each time taking the
// remaining pair slot that turns the fewest additional samples' groups live.
std::vector<uint32_t> PermutationByGreedy(const SampleRows& rows)
{
    const size_t numSamples = rows.size() / RowWords;
    const std::vector<uint64_t> globalCounts = CountActivations(rows);

    std::vector<uint32_t> perm(PairCount);
    std::vector<bool> remaining(PairCount, true);
    std::vector<uint8_t> groupLive(numSamples);
    std::vector<uint64_t> counts(PairCount);

    for (uint32_t g = 0; g < PairCount / GroupSize; ++g)
    {
        // a fresh group starts dead in every sample, so every slot is back to its global count
        counts = globalCounts;
        std::fill(groupLive.begin(), groupLive.end(), (uint8_t)0);

        for (uint32_t j = 0; j < GroupSize; ++j)
        {
            uint32_t best = UINT32_MAX;
            uint64_t bestCount = UINT64_MAX;
            for (uint32_t slot = 0; slot < PairCount; ++slot)
            {
                if (remaining[slot] && counts[slot] < bestCount)
                {
                    bestCount = counts[slot];
                    best = slot;
                }
            }

            perm[g * GroupSize + j] = best;
            remaining[best] = false;

            // samples this slot just brought to life no longer contribute to any other slot's count
            for (size_t s = 0; s < numSamples; ++s)
            {
                if (groupLive[s])
                    continue;

                const uint64_t* row = rows.data() + s * RowWords;
                if (!TestBit(row, best))
                    continue;

                groupLive[s] = 1;
                for (uint32_t w = 0; w < RowWords; ++w)
                {
                    uint64_t bits = row[w];
                    while (bits)
                    {
                        counts[w * 64 + FirstBitSet(bits)]--;
                        bits &= bits - 1;
                    }
                }
            }
        }
    }

    return perm;
}

// perm[i] is the pair slot that moves to position i
void ApplyPermutation(const nn::PackedNeuralNetwork& src, nn::PackedNeuralNetwork& dst, const std::vector<uint32_t>& perm)
{
    memcpy(&dst, &src, sizeof(nn::PackedNeuralNetwork));

    for (uint32_t feature = 0; feature < nn::NumNetworkInputs; ++feature)
    {
        const nn::FirstLayerWeightType* srcRow = src.accumulatorWeights + feature * nn::AccumulatorSize;
        nn::FirstLayerWeightType* dstRow = dst.accumulatorWeights + feature * nn::AccumulatorSize;

        for (uint32_t i = 0; i < PairCount; ++i)
        {
            dstRow[i] = srcRow[perm[i]];
            dstRow[i + PairCount] = srcRow[perm[i] + PairCount];
        }
    }

    for (uint32_t i = 0; i < PairCount; ++i)
    {
        dst.accumulatorBiases[i] = src.accumulatorBiases[perm[i]];
        dst.accumulatorBiases[i + PairCount] = src.accumulatorBiases[perm[i] + PairCount];
    }

    for (uint32_t variant = 0; variant < nn::NumVariants; ++variant)
    {
        const auto& srcSubnet = src.outputSubnetVariants[variant];
        auto& dstSubnet = dst.outputSubnetVariants[variant];

        for (uint32_t output = 0; output < nn::L1Size; ++output)
        {
            for (uint32_t i = 0; i < PairCount; ++i)
            {
                // side to move occupies the low half of the L1 input, the other side the high half
                dstSubnet.l1Weights[nn::HiddenWeightIndex(i, output, nn::L1Size)] = srcSubnet.l1Weights[nn::HiddenWeightIndex(perm[i], output, nn::L1Size)];
                dstSubnet.l1Weights[nn::HiddenWeightIndex(i + PairCount, output, nn::L1Size)] = srcSubnet.l1Weights[nn::HiddenWeightIndex(perm[i] + PairCount, output, nn::L1Size)];
            }
        }
    }
}

// Re-evaluates positions with both nets; the permutation is only correct if every output matches.
bool VerifyIdentical(const char* path, const nn::PackedNeuralNetwork& a, const nn::PackedNeuralNetwork& b, uint64_t maxPositions)
{
    FILE* file = fopen(path, "rb");
    if (!file)
        return false;

    constexpr uint32_t cEntriesPerRead = 4096;
    std::vector<PositionEntry> buffer(cEntriesPerRead);

    uint64_t numChecked = 0;
    uint64_t numMismatches = 0;
    for (;;)
    {
        const size_t count = fread(buffer.data(), sizeof(PositionEntry), cEntriesPerRead, file);
        if (count == 0)
            break;

        for (size_t i = 0; i < count && numChecked < maxPositions; ++i)
        {
            Position pos;
            if (!UnpackPosition(buffer[i].pos, pos, false) || !pos.IsValid())
                continue;
            if (!IsEvaluatedAtRuntime(pos))
                continue;

            uint16_t stmFeatures[64];
            uint16_t nstmFeatures[64];
            const Color stm = pos.GetSideToMove();
            const uint32_t numStm = PositionToFeaturesVector(pos, stmFeatures, stm);
            const uint32_t numNstm = PositionToFeaturesVector(pos, nstmFeatures, stm ^ 1);
            const uint32_t variant = GetNetworkVariant(pos);

            numChecked++;
            if (a.Run(stmFeatures, numStm, nstmFeatures, numNstm, variant) !=
                b.Run(stmFeatures, numStm, nstmFeatures, numNstm, variant))
            {
                numMismatches++;
            }
        }

        if (numChecked >= maxPositions)
            break;
    }

    fclose(file);

    std::cout << "Verified " << numChecked << " positions, " << numMismatches << " mismatches" << std::endl;
    return numMismatches == 0;
}

} // namespace

bool PermuteNet(const std::vector<std::string>& args)
{
    std::string netPath;
    std::string outPath;
    std::string method = "greedy";
    uint64_t maxPositions = 200000;
    std::vector<std::string> dataPaths;

    for (size_t i = 0; i < args.size(); ++i)
    {
        const bool hasValue = i + 1 < args.size();
        if (args[i] == "--net" && hasValue)
            netPath = args[++i];
        else if (args[i] == "--out" && hasValue)
            outPath = args[++i];
        else if (args[i] == "--method" && hasValue)
            method = args[++i];
        else if (args[i] == "--maxPositions" && hasValue)
            maxPositions = std::stoull(args[++i]);
        else
            dataPaths.push_back(args[i]);
    }

    if (netPath.empty())
    {
        std::cerr << "Missing --net" << std::endl;
        return false;
    }

    if (dataPaths.empty())
    {
        std::cerr << "No training data files given; pass one or more .dat files to sample activations from" << std::endl;
        return false;
    }

    if (method != "greedy" && method != "sort")
    {
        std::cerr << "Unknown --method " << method << ", expected greedy or sort" << std::endl;
        return false;
    }

    if (outPath.empty())
    {
        const size_t dot = netPath.find_last_of('.');
        outPath = (dot == std::string::npos ? netPath : netPath.substr(0, dot)) + "-permuted.pnn";
    }

    auto net = std::make_unique<nn::PackedNeuralNetwork>();
    if (!net->LoadFromFile(netPath.c_str()))
        return false;

    std::cout << "Loaded " << netPath << std::endl;

    SampleRows rows;
    rows.reserve(2ull * maxPositions * RowWords);
    {
        const uint64_t perFile = std::max<uint64_t>(1, maxPositions / dataPaths.size());
        uint64_t total = 0;
        for (const std::string& path : dataPaths)
            total += CollectSamples(path.c_str(), *net, perFile, rows);

        if (total == 0)
        {
            std::cerr << "No usable positions found" << std::endl;
            return false;
        }
        std::cout << "Collected " << total << " positions (" << rows.size() / RowWords << " samples)" << std::endl;
    }

    std::vector<uint32_t> identity(PairCount);
    std::iota(identity.begin(), identity.end(), 0u);

    const std::vector<uint32_t> perm = (method == "greedy") ? PermutationByGreedy(rows) : PermutationBySortedActivation(rows);

    const double before = LiveGroupRate(rows, identity);
    const double after = LiveGroupRate(rows, perm);
    std::cout << "Live 4-input groups: " << 100.0 * before << "% -> " << 100.0 * after << "%"
        << "  (" << (uint32_t)(before * NumGroups + 0.5f) << " -> " << (uint32_t)(after * NumGroups + 0.5f)
        << " of " << NumGroups << " per evaluation)" << std::endl;

    auto permuted = std::make_unique<nn::PackedNeuralNetwork>();
    ApplyPermutation(*net, *permuted, perm);

    if (!VerifyIdentical(dataPaths[0].c_str(), *net, *permuted, 100000))
    {
        std::cerr << "ERROR: permuted network does not match the original" << std::endl;
        return false;
    }

    if (!permuted->SaveToFile(outPath.c_str()))
        return false;

    std::cout << "Saved " << outPath << std::endl;
    return true;
}
