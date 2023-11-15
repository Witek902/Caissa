#include "Common.hpp"
#include "ThreadPool.hpp"
#include "TrainerCommon.hpp"
#include "GameCollection.hpp"

#include "../backend/Math.hpp"
#include "../backend/Material.hpp"
#include "../backend/Waitable.hpp"
#include "../backend/Evaluate.hpp"
#include "../backend/Endgame.hpp"
#include "../backend/Tablebase.hpp"

#include <filesystem>
#include <fstream>

// converts games in plain text format <FEN> [game result] <eval>
// to binary format
void PlainTextToTrainingData(const std::vector<std::string>& args)
{
    if (args.size() < 1)
    {
        std::cout << "Usage: PrepareTrainingData <input files>" << std::endl;
        return;
    }

    for (const std::string& inputPath : args)
    {
        std::cout << "Processing " << inputPath << std::endl;

        // read input file
        std::ifstream inputFile(inputPath);
        if (!inputFile.is_open())
        {
            std::cout << "Failed to open input file: " << inputPath << std::endl;
            return;
        }

        // write output file
        const std::string outputPath = inputPath + ".bin";
        std::ofstream outputFile(outputPath, std::ios::binary);
        if (!outputFile.is_open())
        {
            std::cout << "Failed to open output file: " << outputPath << std::endl;
            return;
        }

        std::vector<PositionEntry> entries;

        // read line by line
        std::string line;
        while (std::getline(inputFile, line))
        {
            // remove trailing spaces
            line.erase(line.find_last_not_of(" \n\r\t") + 1);

            // parse score (last number in the line)
            const size_t scoreStart = line.find_last_of(' ');
            const std::string scoreStr = line.substr(scoreStart + 1);
            const int32_t moveScore = static_cast<ScoreType>(std::stoi(scoreStr.c_str()));
            if (moveScore > INT16_MAX || moveScore < INT16_MIN)
            {
                std::cout << "Score out of range: " << moveScore << std::endl;
                continue;
            }

            // parse game result
            const size_t resultStart = line.find_last_of(' ', scoreStart - 1);
            const std::string resultStr = line.substr(resultStart + 1, scoreStart - resultStart - 1);

            Game::Score gameScore = Game::Score::Unknown;
            if (resultStr == "[1.0]" || resultStr == "[1-0]") gameScore = Game::Score::WhiteWins;
            else if (resultStr == "[0.5]" || resultStr == "[1/2-1/2]") gameScore = Game::Score::Draw;
            else if (resultStr == "[0.0]" || resultStr == "[0-1]") gameScore = Game::Score::BlackWins;
            else
            {
                std::cout << "Failed to parse game result: " << resultStr << std::endl;
                continue;
            }

            // parse FEN
            const std::string fen = line.substr(0, resultStart);
            Position pos;
            if (!pos.FromFEN(fen))
            {
                std::cout << "Failed to parse FEN: " << fen << std::endl;
                continue;
            }

            if (pos.GetNumPieces() >= 4 &&
                (std::abs(moveScore) < 2000 || std::abs(Evaluate(pos)) < 2000) &&   // skip unbalanced positions
                !pos.IsInCheck())
            {
                PositionEntry entry{};

                entry.wdlScore = static_cast<uint8_t>(gameScore);
                entry.tbScore = static_cast<uint8_t>(Game::Score::Unknown);
                entry.score = static_cast<ScoreType>(moveScore);

                Position normalizedPos = pos;
                if (pos.GetSideToMove() == Color::Black)
                {
                    // make whites side to move
                    normalizedPos = normalizedPos.SwappedColors();

                    // flip score
                    entry.score = -entry.score;
                    if (gameScore == Game::Score::WhiteWins) entry.wdlScore = static_cast<uint8_t>(Game::Score::BlackWins);
                    if (gameScore == Game::Score::BlackWins) entry.wdlScore = static_cast<uint8_t>(Game::Score::WhiteWins);
                }

                // tweak score with the help of endgame tablebases
                int32_t wdl = 0;
                if (pos.GetNumPieces() <= 7 && ProbeSyzygy_WDL(pos, &wdl))
                {
                    if (wdl > 0)        entry.tbScore = static_cast<uint8_t>(Game::Score::WhiteWins);
                    else if (wdl < 0)   entry.tbScore = static_cast<uint8_t>(Game::Score::BlackWins);
                    else                entry.tbScore = static_cast<uint8_t>(Game::Score::Draw);
                }

                ASSERT(normalizedPos.IsValid());
                VERIFY(PackPosition(normalizedPos, entry.pos));
                entries.push_back(entry);
            }
        }

        std::cout << "Extracted " << entries.size() << " positions" << std::endl;

        // shuffle the training data
        {
            std::mt19937 randomGenerator;
            std::shuffle(entries.begin(), entries.end(), randomGenerator);
        }

        // write entries
        outputFile.write(reinterpret_cast<const char*>(entries.data()), entries.size() * sizeof(PositionEntry));
    }
}