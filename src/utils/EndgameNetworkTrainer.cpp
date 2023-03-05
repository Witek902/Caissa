#include "Common.hpp"
#include "ThreadPool.hpp"
#include "NeuralNetwork.hpp"

#include "../backend/Position.hpp"
#include "../backend/PositionUtils.hpp"
#include "../backend/Game.hpp"
#include "../backend/Move.hpp"
#include "../backend/Search.hpp"
#include "../backend/TranspositionTable.hpp"
#include "../backend/Evaluate.hpp"
#include "../backend/Material.hpp"
#include "../backend/Endgame.hpp"
#include "../backend/Tablebase.hpp"
#include "../backend/PackedNeuralNetwork.hpp"
#include "../backend/Waitable.hpp"

#include <iostream>
#include <iomanip>
#include <random>
#include <mutex>
#include <fstream>
#include <limits.h>

// #define USE_DTZ
// #define USE_DTM

using namespace threadpool;

static const uint32_t cMaxIterations = 10000000;
static const uint32_t cNumTrainingVectorsPerIteration = 128 * 1024;
static const uint32_t cNumValidationVectorsPerIteration = 16 * 1024;
static const uint32_t cMinBatchSize = 32;
static const uint32_t cMaxBatchSize = 16 * 1024;

static void PositionToPackedVector(const Position& pos, nn::TrainingVector& outVector)
{
    const uint32_t maxFeatures = 64;

    uint16_t features[maxFeatures];
    //uint32_t numFeatures = pos.ToFeaturesVector(features, NetworkInputMapping::MaterialPacked_Symmetrical);
    //uint32_t numFeatures = pos.ToFeaturesVector(features, NetworkInputMapping::KingPiece_Symmetrical);
    uint32_t numFeatures = pos.ToFeaturesVector(features, NetworkInputMapping::Full_Symmetrical);
    ASSERT(numFeatures <= maxFeatures);

    outVector.inputMode = nn::InputMode::SparseBinary;
    outVector.sparseBinaryInputs.clear();
    outVector.sparseBinaryInputs.reserve(numFeatures);

    for (uint32_t i = 0; i < numFeatures; ++i)
    {
        outVector.sparseBinaryInputs.emplace_back(features[i]);
    }
}

static float ScoreFromNN(float score)
{
    return std::clamp(score, 0.0f, 1.0f);
}

bool TrainEndgame()
{
    struct TrainingEntry
    {
        Position pos;
        nn::TrainingVector trainingVector;
    };

    const auto generateTrainingSet = [&](TaskBuilder& taskBuilder, std::vector<TrainingEntry>& outSet)
    {
        taskBuilder.ParallelFor("", (uint32_t)outSet.size(), [&](const TaskContext&, const uint32_t i)
        {
            std::random_device randomDevice;
            std::mt19937 gen(randomDevice());
            std::uniform_int_distribution<uint32_t> pieceIndexDistr(0, 9);
            std::uniform_int_distribution<uint32_t> scoreDistr(0, 18);
            std::uniform_int_distribution<uint32_t> materialDistr(0, 15);

            for (;;)
            {
                MaterialKey materialKey;

                uint32_t seed = materialDistr(gen);

                     if (seed ==  0) materialKey.FromString("KRPvKRP");
                else if (seed ==  1) materialKey.FromString("KRPvKR");
                else if (seed ==  2) materialKey.FromString("KRPPvKR");
                else if (seed ==  3) materialKey.FromString("KPPvKPP");
                else if (seed ==  4) materialKey.FromString("KPPPvKP");
                else if (seed ==  5) materialKey.FromString("KRPvKBP");
                else if (seed ==  6) materialKey.FromString("KRPvKNP");
                else if (seed ==  7) materialKey.FromString("KBPvKBP");
                else if (seed ==  8) materialKey.FromString("KBPvKPP");
                else if (seed ==  9) materialKey.FromString("KNPPvKN");
                else if (seed == 10) materialKey.FromString("KQPPvKQ");
                else if (seed == 11) materialKey.FromString("KQPvKQP");
                else if (seed == 12) materialKey.FromString("KBPvKNP");
                else
                {
                    seed = materialDistr(gen);

                    uint32_t numPieces = 6;
                    if (seed < 4) numPieces = 5;
                    if (seed == 0) numPieces = 4;

                    for (uint32_t j = 0; j < numPieces - 2; ++j)
                    {
                        const uint32_t pieceIndex = pieceIndexDistr(gen);
                        ASSERT(pieceIndex < 10);
                        switch (pieceIndex)
                        {
                        case 0: materialKey.numWhitePawns++; break;
                        case 1: materialKey.numWhiteKnights++; break;
                        case 2: materialKey.numWhiteBishops++; break;
                        case 3: materialKey.numWhiteRooks++; break;
                        case 4: materialKey.numWhiteQueens++; break;
                        case 5: materialKey.numBlackPawns++; break;
                        case 6: materialKey.numBlackKnights++; break;
                        case 7: materialKey.numBlackBishops++; break;
                        case 8: materialKey.numBlackRooks++; break;
                        case 9: materialKey.numBlackQueens++; break;
                        }
                    }
                }

                // generate unbalanced positions with lower probability
                const int64_t whitesScore = materialKey.numWhitePawns + 3 * materialKey.numWhiteKnights + 3 * materialKey.numWhiteBishops + 5 * materialKey.numWhiteRooks + 9 * materialKey.numWhiteQueens;
                const int64_t blacksScore = materialKey.numBlackPawns + 3 * materialKey.numBlackKnights + 3 * materialKey.numBlackBishops + 5 * materialKey.numBlackRooks + 9 * materialKey.numBlackQueens;
                const int64_t scoreDiff = std::abs(whitesScore - blacksScore);
                if (whitesScore == 0 || blacksScore == 0) continue;
                if (scoreDiff > 15) continue;
                if (scoreDistr(gen) < scoreDiff) continue;

                // randomize side
                if (std::uniform_int_distribution<>{0, 1}(gen))
                {
                    materialKey = materialKey.SwappedColors();
                }

                RandomPosDesc desc;
                desc.materialKey = materialKey;

                Position pos;
                GenerateRandomPosition(gen, desc, pos);

                // generate only quiet position
                if (!pos.IsValid() || !pos.IsQuiet())
                {
                    continue;
                }

                int32_t wdl = 0;
                if (!ProbeSyzygy_WDL(pos, &wdl))
                {
                    continue;
                }

                const float bias = 0.0f;

                float score = 0.5f;
                if (wdl < 0) score = 0.0f + bias;
                if (wdl > 0) score = 1.0f - bias;

                if (wdl != 0)
                {
#ifdef USE_DTM
                    const uint32_t longestDTM = 253; // for 5 pieces
                    uint32_t gaviotaDTM = 0;
                    int32_t gaviotaWDL = 0;
                    if (ProbeGaviota(pos, &gaviotaDTM, &gaviotaWDL))
                    {
                        ASSERT(gaviotaDTM > 0);
                        ASSERT(wdl == gaviotaWDL);

                        const float power = 1.5f;
                        const float scale = 0.45f;
                        const float offset = 0.001f;

                        if (wdl < 0) score = offset + scale * powf((float)(gaviotaDTM - 1) / (float)longestDTM, power);
                        if (wdl > 0) score = 1.0f - offset - scale * powf((float)(gaviotaDTM - 1) / (float)longestDTM, power);
                    }
#endif // USE_DTM
#ifdef USE_DTZ
                    Move tempMove;
                    uint32_t dtz = 0;
                    if (ProbeSyzygy_Root(pos, tempMove, &dtz))
                    {
                        if (dtz <= 1) continue;

                        const float power = 1.5f;
                        const float scale = 0.45f;
                        const float offset = 0.0f;

                        if (wdl < 0) score = offset + scale * powf((float)dtz / 100.0f, power);
                        if (wdl > 0) score = 1.0f - offset - scale * powf((float)dtz / 100.0f, power);
                    }
#endif // USE_DTZ
                }

                PositionToPackedVector(pos, outSet[i].trainingVector);
                outSet[i].trainingVector.singleOutput = score;
                outSet[i].pos = pos;

                break;
            }
        });
    };

    //const uint32_t numNetworkInputs = 32 + 64 * 3 + 48 * 2;
    const uint32_t numNetworkInputs = 704;
    //const uint32_t numNetworkInputs = materialKey.GetNeuralNetworkInputsNumber();
    //const uint32_t numNetworkInputs = 2 * 3 * 32 * 64;
    //const uint32_t numNetworkInputs = 2 * 10 * 32 * 64;

    std::string name = "endgame";

    nn::NeuralNetwork network;
    network.Init(numNetworkInputs, { 1024, 1 });
    //network.Load("checkpoint.nn");

    nn::NeuralNetworkRunContext networkRunCtx;
    networkRunCtx.Init(network);

    nn::NeuralNetworkTrainer trainer;
    std::unique_ptr<nn::PackedNeuralNetwork> packedNetwork = std::make_unique<nn::PackedNeuralNetwork>();

    std::vector<TrainingEntry> trainingSet(cNumTrainingVectorsPerIteration);
    std::vector<nn::TrainingVector> batch(cNumTrainingVectorsPerIteration);

    std::vector<int32_t> packedNetworkOutputs(cNumTrainingVectorsPerIteration);

    uint32_t numTrainingVectorsPassed = 0;

    {
        Waitable waitable;
        {
            TaskBuilder childTaskBuilder(waitable);
            generateTrainingSet(childTaskBuilder, trainingSet);
        }
        waitable.Wait();
    }

    TimePoint prevIterationStartTime = TimePoint::GetCurrent();

    for (uint32_t iteration = 0; iteration < cMaxIterations; ++iteration)
    {
        float learningRate = std::max(0.1f, 1.0f / (1.0f + 0.00001f * iteration));

        TimePoint iterationStartTime = TimePoint::GetCurrent();
        float iterationTime = (iterationStartTime - prevIterationStartTime).ToSeconds();
        prevIterationStartTime = iterationStartTime;

        for (size_t i = 0; i < cNumTrainingVectorsPerIteration; ++i)
        {
            batch[i] = trainingSet[i].trainingVector;
        }

        // validation vectors generation can be done in parallel with training
        Waitable waitable;
        {
            TaskBuilder taskBuilder(waitable);

            taskBuilder.Task("Train", [&](const TaskContext& ctx)
            {
                nn::TrainParams params;
                params.batchSize = std::min(cMinBatchSize + iteration * cMinBatchSize, cMaxBatchSize);
                params.learningRate = learningRate;

                TaskBuilder taskBuilder{ ctx };
                trainer.Train(network, batch, params, &taskBuilder);
            });

            taskBuilder.Task("GenerateSet", [&](const TaskContext& ctx)
            {
                TaskBuilder childTaskBuilder(ctx);
                generateTrainingSet(childTaskBuilder, trainingSet);
            });

        }
        waitable.Wait();

        numTrainingVectorsPassed += cNumTrainingVectorsPerIteration;

        float nnMinError = std::numeric_limits<float>::max();
        float nnMaxError = 0.0f, nnErrorSum = 0.0f;

        float nnPackedQuantizationErrorSum = 0.0f;
        float nnPackedMinError = std::numeric_limits<float>::max();
        float nnPackedMaxError = 0.0f, nnPackedErrorSum = 0.0f;

        float evalMinError = std::numeric_limits<float>::max();
        float evalMaxError = 0.0f, evalErrorSum = 0.0f;

        uint32_t correctPredictions = 0;

        float packedNetworkRunTime = 0.0f;

        if (packedNetwork)
        {
            network.ToPackedNetwork(*packedNetwork);

            TimePoint startTime = TimePoint::GetCurrent();
            for (uint32_t i = 0; i < cNumValidationVectorsPerIteration; ++i)
            {
                const std::vector<uint16_t>& features = trainingSet[i].trainingVector.sparseBinaryInputs;
                packedNetworkOutputs[i] = packedNetwork->Run(features.data(), (uint32_t)features.size(), 0u);
            }
            packedNetworkRunTime = (TimePoint::GetCurrent() - startTime).ToSeconds();
        }

        for (uint32_t i = 0; i < cNumValidationVectorsPerIteration; ++i)
        {
            const nn::NeuralNetwork::InputDesc networkInput(trainingSet[i].trainingVector.sparseBinaryInputs);
            const nn::Values& networkOutput = network.Run(networkInput, networkRunCtx);
            const int32_t packedNetworkOutput = packedNetworkOutputs[i];

            const float expectedValue = ScoreFromNN(trainingSet[i].trainingVector.singleOutput);
            const float nnValue = ScoreFromNN(networkOutput[0]);
            const float nnPackedValue = ScoreFromNN(nn::Sigmoid((float)packedNetworkOutput / (float)nn::OutputScale));
            const float evalValue = PawnToWinProbability((float)Evaluate(trainingSet[i].pos) / 100.0f);

            nnPackedQuantizationErrorSum += (nnValue - nnPackedValue) * (nnValue - nnPackedValue);

            if ((expectedValue >= 2.0f/3.0f && nnValue >= 2.0f/3.0f) ||
                (expectedValue <= 1.0f/3.0f && nnValue <= 1.0f/3.0f) ||
                (expectedValue > 1.0f/3.0f && expectedValue < 2.0f/3.0f && nnValue > 1.0f/3.0f && nnValue < 2.0f/3.0f))
            {
                correctPredictions++;
            }

            if (i + 1 == cNumValidationVectorsPerIteration)
            {
                std::cout
                    << trainingSet[i].pos.ToFEN() << std::endl << trainingSet[i].pos.Print() << std::endl
                    << "True Score:     " << expectedValue << " (" << WinProbabilityToCentiPawns(expectedValue) << ")" << std::endl
                    << "NN eval:        " << nnValue << " (" << WinProbabilityToCentiPawns(nnValue) << ")" << std::endl
                    << "Packed NN eval: " << nnPackedValue << " (" << WinProbabilityToCentiPawns(nnPackedValue) << ")" << std::endl
                    << "Static eval:    " << evalValue << " (" << WinProbabilityToCentiPawns(evalValue) << ")" << std::endl
                    << std::endl;
            }

            {
                const float error = expectedValue - nnValue;
                const float errorDiff = std::abs(error);
                nnErrorSum += error * error;
                nnMinError = std::min(nnMinError, errorDiff);
                nnMaxError = std::max(nnMaxError, errorDiff);
            }

            {
                const float error = expectedValue - nnPackedValue;
                const float errorDiff = std::abs(error);
                nnPackedErrorSum += error * error;
                nnPackedMinError = std::min(nnPackedMinError, errorDiff);
                nnPackedMaxError = std::max(nnPackedMaxError, errorDiff);
            }

            {
                const float error = expectedValue - evalValue;
                const float errorDiff = std::abs(error);
                evalErrorSum += error * error;
                evalMinError = std::min(evalMinError, errorDiff);
                evalMaxError = std::max(evalMaxError, errorDiff);
            }
        }

        const float accuracy = (float)correctPredictions / (float)cNumValidationVectorsPerIteration;
        nnErrorSum = sqrtf(nnErrorSum / cNumValidationVectorsPerIteration);
        nnPackedErrorSum = sqrtf(nnPackedErrorSum / cNumValidationVectorsPerIteration);
        evalErrorSum = sqrtf(evalErrorSum / cNumValidationVectorsPerIteration);
        nnPackedQuantizationErrorSum = sqrtf(nnPackedQuantizationErrorSum / cNumValidationVectorsPerIteration);

        std::cout
            << "Epoch:                  " << iteration << std::endl
            << "Num training vectors:   " << numTrainingVectorsPassed << std::endl
            << "Learning rate:          " << learningRate << std::endl
            << "Accuracy:               " << (100.0f * accuracy) << "%" << std::endl
            << "NN avg/min/max error:   " << std::setprecision(5) << nnErrorSum << " " << std::setprecision(4) << nnMinError << " " << std::setprecision(4) << nnMaxError << std::endl
            << "PNN avg/min/max error:  " << std::setprecision(5) << nnPackedErrorSum << " " << std::setprecision(4) << nnPackedMinError << " " << std::setprecision(4) << nnPackedMaxError << std::endl
            << "Quantization error:     " << std::setprecision(5) << nnPackedQuantizationErrorSum << std::endl
            << "Eval avg/min/max error: " << std::setprecision(5) << evalErrorSum << " " << std::setprecision(4) << evalMinError << " " << std::setprecision(4) << evalMaxError << std::endl;

        network.PrintStats();

        std::cout << "Iteration time:   " << 1000.0f * iterationTime << " ms" << std::endl;
        std::cout << "Network run time: " << 1000.0f * packedNetworkRunTime << " ms" << std::endl << std::endl;

        if (iteration % 10 == 0)
        {
            network.Save((name + ".nn").c_str());

            if (packedNetwork)
            {
                packedNetwork->Save((name + ".pnn").c_str());
                packedNetwork->SaveAsImage((name + ".raw").c_str());
            }
        }
    }

    return true;
}