#include "Common.hpp"
#include "ThreadPool.hpp"
#include "TrainerCommon.hpp"

#include "../backend/Position.hpp"
#include "../backend/PositionUtils.hpp"
#include "../backend/Game.hpp"
#include "../backend/Move.hpp"
#include "../backend/Search.hpp"
#include "../backend/TranspositionTable.hpp"
#include "../backend/Evaluate.hpp"
#include "../backend/Material.hpp"
#include "../backend/Tablebase.hpp"
#include "../backend/PackedNeuralNetwork.hpp"
#include "../backend/Waitable.hpp"

#include <iostream>
#include <iomanip>
#include <random>
#include <mutex>
#include <fstream>
#include <limits.h>

void GenerateEndgamePositions()
{
    const uint32_t numPieces = 6;
    const uint32_t maxPositions = 5'000'000;

    const std::string outputPath = "endgame.bin";
    const std::string outputPathTxt = "endgame.epd";

    std::ofstream outputFileBin(outputPath, std::ios::binary);
    if (!outputFileBin.is_open())
    {
        std::cout << "Failed to open output file: " << outputPath << std::endl;
        return;
    }

    std::ofstream outputFileTxt(outputPathTxt);
    if (!outputFileTxt.is_open())
    {
        std::cout << "Failed to open output file: " << outputPathTxt << std::endl;
        return;
    }

    std::mutex mutex;
    //std::mutex tbMutex;
    uint32_t numPositions = 0;

    const auto generate = [&]()
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<uint32_t> pieceIndexDistr(0, 9);
        std::uniform_int_distribution<uint32_t> scoreDistr(0, 18);

        for (;;)
        {
            MaterialKey matKey;

            for (uint32_t j = 0; j < numPieces - 2; ++j)
            {
                const uint32_t pieceIndex = pieceIndexDistr(gen);
                ASSERT(pieceIndex < 10);
                switch (pieceIndex)
                {
                case 0: matKey.numWhitePawns++; break;
                case 1: matKey.numWhiteKnights++; break;
                case 2: matKey.numWhiteBishops++; break;
                case 3: matKey.numWhiteRooks++; break;
                case 4: matKey.numWhiteQueens++; break;
                case 5: matKey.numBlackPawns++; break;
                case 6: matKey.numBlackKnights++; break;
                case 7: matKey.numBlackBishops++; break;
                case 8: matKey.numBlackRooks++; break;
                case 9: matKey.numBlackQueens++; break;
                }
            }

            if (matKey.numWhiteKnights > 2 || matKey.numBlackKnights > 2 ||
                matKey.numWhiteBishops > 2 || matKey.numBlackBishops > 2 ||
                matKey.numWhiteRooks > 2 || matKey.numBlackRooks > 2 ||
                matKey.numWhiteQueens > 1 || matKey.numBlackQueens > 1)
                continue;

            // generate unbalanced positions with lower probability
            const int64_t whitesScore = matKey.numWhitePawns + 3 * matKey.numWhiteKnights + 3 * matKey.numWhiteBishops + 5 * matKey.numWhiteRooks + 9 * matKey.numWhiteQueens;
            const int64_t blacksScore = matKey.numBlackPawns + 3 * matKey.numBlackKnights + 3 * matKey.numBlackBishops + 5 * matKey.numBlackRooks + 9 * matKey.numBlackQueens;
            const int64_t scoreDiff = std::abs(whitesScore - blacksScore);
            if (whitesScore == 0 || blacksScore == 0) continue;
            if (scoreDiff > 10) continue;
            //if (scoreDistr(gen) < scoreDiff) continue;

            // randomize side
            if (std::uniform_int_distribution<>{0, 1}(gen))
            {
                matKey = matKey.SwappedColors();
            }

            Position pos;

            const RandomPosDesc desc{ matKey };
            GenerateRandomPosition(gen, desc, pos);

            // skip positions with more than 1 bishop on the same color square
            if ((pos.Whites().bishops & Bitboard::LightSquares()).Count() > 1 ||
                (pos.Whites().bishops & Bitboard::DarkSquares()).Count() > 1 ||
                (pos.Blacks().bishops & Bitboard::LightSquares()).Count() > 1 ||
                (pos.Blacks().bishops & Bitboard::DarkSquares()).Count() > 1)
                continue;

            // generate only quiet position
            if (!pos.IsValid() || !pos.IsQuiet())
                continue;

            // skip positions not present in tablebase
            int32_t wdl = 0;
            if (!ProbeSyzygy_WDL(pos, &wdl))
                continue;

            const ScoreType eval = Evaluate(pos);

            // skip positions which evaluations matches WDL
            if ((eval > 800 && wdl > 0) || (eval < -800 && wdl < 0))
                continue;

            /*
            if (wdl != 0)
            {
                Move bestMove;
                uint32_t dtz = 0;
                {
                    std::lock_guard<std::mutex> lock(tbMutex);
                    if (!ProbeSyzygy_Root(pos, bestMove, &dtz, nullptr))
                        continue;
                }
                if (dtz < 4)
                    continue;
            }
            */

            // skip positions which evaluations matches WDL (probabilistic)
            {
                const uint32_t ply = 64;
                const float w = EvalToWinProbability(eval / 100.0f, ply);
                const float l = EvalToWinProbability(-eval / 100.0f, ply);
                const float d = 1.0f - w - l;

                float prob = d;
                if (wdl > 0) prob = w;
                if (wdl < 0) prob = l;

                std::bernoulli_distribution skippingDistr(prob);
                if (skippingDistr(gen))
                    continue;
            }

            {
                std::lock_guard<std::mutex> lock(mutex);

                // write position to file
                {
                    PositionEntry entry{};
                    VERIFY(PackPosition(pos, entry.pos));
                    if (wdl > 0)
                        entry.wdlScore = static_cast<uint8_t>(Game::Score::WhiteWins);
                    else if (wdl < 0)
                        entry.wdlScore = static_cast<uint8_t>(Game::Score::BlackWins);
                    else
                        entry.wdlScore = static_cast<uint8_t>(Game::Score::Draw);
                    entry.tbScore = entry.wdlScore;
                    entry.score = static_cast<ScoreType>(eval);

                    outputFileBin.write(reinterpret_cast<const char*>(&entry), sizeof(PositionEntry));
                }

                outputFileTxt << pos.ToFEN() << " eval=" << eval << " wdl=" << wdl << "\n";

                numPositions++;
                if (numPositions % 10000 == 0)
                    std::cout << "Generated " << numPositions << " positions" << std::endl;

                if (numPositions >= maxPositions)
                    return;
            }
        }
    };

    std::vector<std::thread> threads;
    for (uint32_t i = 0; i < std::thread::hardware_concurrency(); ++i)
    {
        threads.emplace_back(generate);
    }

    for (auto& thread : threads)
    {
        thread.join();
    }
}