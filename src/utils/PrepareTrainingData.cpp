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

using namespace threadpool;

// #define OUTPUT_TEXT_FILE

static bool ConvertGamesToTrainingData(const std::string& inputPath, const std::string& outputPath)
{
    std::vector<PositionEntry> entries;
    std::vector<Move> moves;

    FileInputStream gamesFile(inputPath.c_str());
    if (!gamesFile.IsOpen())
    {
        std::cout << "ERROR: Failed to load selfplay data file: " << inputPath << std::endl;
        return false;
    }

    if (std::filesystem::exists(outputPath))
    {
        std::cout << "INFO: Output training data file " << outputPath << " already exists. Skipping" << std::endl;
        return true;
    }

#ifndef OUTPUT_TEXT_FILE
    FileOutputStream trainingDataFile(outputPath.c_str());
    if (!trainingDataFile.IsOpen())
    {
        std::cout << "ERROR: Failed to load output training data file: " << outputPath << std::endl;
        return false;
    }
#endif // OUTPUT_TEXT_FILE

    uint32_t numGames = 0;
    uint32_t numPositions = 0;

    Game game;
    while (GameCollection::ReadGame(gamesFile, game, moves))
    {
        Game::Score gameScore = game.GetScore();

        ASSERT(game.GetMoves().size() == game.GetMoveScores().size());

        if (game.GetScore() == Game::Score::Unknown)
        {
            continue;
        }

        Position pos = game.GetInitialPosition();

        // replay the game
        for (size_t i = 0; i < game.GetMoves().size(); ++i)
        {
            const Move move = moves[i];
            const ScoreType moveScore = game.GetMoveScores()[i];

            if (move.IsQuiet() &&                                   // best move must be quiet
                (i >= 2 || pos.GetNumPieces() < 32) &&
                pos.GetNumPieces() >= 4 &&
                (std::abs(moveScore) < 1200 || std::abs(Evaluate(pos)) < 2000) &&   // skip unbalanced positions
                !pos.IsInCheck())
            {
                PositionEntry entry{};

                entry.wdlScore = static_cast<uint8_t>(gameScore);
                entry.tbScore = static_cast<uint8_t>(Game::Score::Unknown);
                entry.score = moveScore;

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
                         if (wdl > 0)   entry.tbScore = static_cast<uint8_t>(Game::Score::WhiteWins);
                    else if (wdl < 0)   entry.tbScore = static_cast<uint8_t>(Game::Score::BlackWins);
                    else                entry.tbScore = static_cast<uint8_t>(Game::Score::Draw);
                }

                ASSERT(normalizedPos.IsValid());
                VERIFY(PackPosition(normalizedPos, entry.pos));
                entries.push_back(entry);
                numPositions++;
            }

            if (!pos.DoMove(move))
            {
                break;
            }
        }

        numGames++;
    }

    std::cout << "Parsed " << numGames << " games from " << inputPath << ", extracted " << numPositions << " positions" << std::endl;

    // shuffle the training data
    {
        std::random_device randomDevice;
        std::mt19937 randomGenerator;
        std::shuffle(entries.begin(), entries.end(), randomGenerator);
    }

#ifdef OUTPUT_TEXT_FILE
    {
        FILE* outputTextFile = fopen(outputPath.c_str(), "w");

        Position pos;
        for (const PositionEntry& entry : entries)
        {
            VERIFY(UnpackPosition(entry.pos, pos, false));
            ASSERT(pos.GetSideToMove() == Color::White);

            const char* scoreStr = "0.5";
            if (entry.wdlScore == static_cast<uint8_t>(Game::Score::WhiteWins)) scoreStr = "1";
            if (entry.wdlScore == static_cast<uint8_t>(Game::Score::BlackWins)) scoreStr = "0";

            fprintf(outputTextFile, "%s | %d | %s\n", pos.ToFEN().c_str(), (int32_t)entry.score, scoreStr);
        }

        fclose(outputTextFile);
    }
#else // !OUTPUT_TEXT_FILE

    if (!trainingDataFile.Write(entries.data(), entries.size() * sizeof(PositionEntry)))
    {
        std::cout << "ERROR: Failed to write training data file: " << outputPath << std::endl;
        return false;
    }

#endif // OUTPUT_TEXT_FILE

    return true;
}

void PrepareTrainingData(const std::vector<std::string>& args)
{
    (void)args;

    const std::string gamesPath = DATA_PATH "selfplayGames/";
    const std::string trainingDataPath = DATA_PATH "trainingData/";

    Waitable waitable;
    {
        TaskBuilder taskBuilder(waitable);

        for (const auto& path : std::filesystem::directory_iterator(gamesPath))
        {
            std::cout << "Loading " << path.path().string() << "..." << std::endl;

            taskBuilder.Task("LoadPositions", [path, &trainingDataPath](const TaskContext&)
            {
                const std::string outputPath = trainingDataPath + path.path().stem().string() + ".dat";
                ConvertGamesToTrainingData(path.path().string().c_str(), outputPath);
            });
        }
    }

    waitable.Wait();
}