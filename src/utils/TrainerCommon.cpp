#include "Common.hpp"
#include "ThreadPool.hpp"
#include "TrainerCommon.hpp"
#include "GameCollection.hpp"

#include "../backend/Material.hpp"
#include "../backend/Waitable.hpp"
#include "../backend/Evaluate.hpp"
#include "../backend/Endgame.hpp"

#include <filesystem>

using namespace threadpool;

static bool LoadPositions(const char* fileName, std::vector<PositionEntry>& entries)
{
    FileInputStream gamesFile(fileName);
    if (!gamesFile.IsOpen())
    {
        std::cout << "ERROR: Failed to load selfplay data file!" << std::endl;
        return false;
    }

    GameCollection::Reader reader(gamesFile);

    uint32_t numGames = 0;
    uint32_t numPositions = 0;

    Game game;
    while (reader.ReadGame(game))
    {
        Game::Score gameScore = game.GetScore();

        ASSERT(game.GetMoves().size() == game.GetMoveScores().size());

        if (game.GetScore() == Game::Score::Unknown)
        {
            continue;
        }

        float score = 0.5f;
        if (gameScore == Game::Score::WhiteWins) score = 1.0f;
        if (gameScore == Game::Score::BlackWins) score = 0.0f;

        Position pos = game.GetInitialPosition();

        // replay the game
        for (size_t i = 0; i < game.GetMoves().size(); ++i)
        {
            const float gamePhase = powf((float)i / (float)game.GetMoves().size(), 2.0f);
            const Move move = pos.MoveFromPacked(game.GetMoves()[i]);
            const ScoreType moveScore = game.GetMoveScores()[i];
            const MaterialKey matKey = pos.GetMaterialKey();

            // skip boring equal positions
            const bool equalPosition =
                i > 20 &&
                matKey.numBlackPawns == matKey.numWhitePawns &&
                matKey.numBlackKnights == matKey.numWhiteKnights &&
                matKey.numBlackBishops == matKey.numWhiteBishops &&
                matKey.numBlackRooks == matKey.numWhiteRooks &&
                matKey.numBlackQueens == matKey.numWhiteQueens &&
                gameScore == Game::Score::Draw &&
                std::abs(moveScore) < 5;

            // skip positions that will be using simplified evaluation at the runtime
            const int32_t staticEval = Evaluate(pos, nullptr, false);
            const bool simplifiedEvaluation = std::abs(staticEval) > 2 * c_nnTresholdMax;

            // skip recognized endgame positions
            int32_t endgameScore;
            const bool knownEndgamePosition = EvaluateEndgame(pos, endgameScore);

            if (!simplifiedEvaluation &&
                !knownEndgamePosition &&
                !equalPosition &&
                !pos.IsInCheck() && !move.IsCapture() && !move.IsPromotion() &&
                pos.GetNumPieces() >= 6 &&
                pos.GetHalfMoveCount() < 60)
            {
                PositionEntry entry{};

                // blend in future scores into current move score
                float scoreSum = 0.0f;
                float weightSum = 0.0f;
                const size_t maxLookahead = 8;
                for (size_t j = 0; j < maxLookahead; ++j)
                {
                    if (i + j >= game.GetMoves().size()) break;
                    const float weight = 1.0f / (j + 1);
                    scoreSum += weight * CentiPawnToWinProbability(game.GetMoveScores()[i + j]);
                    weightSum += weight;
                }
                ASSERT(weightSum > 0.0f);
                scoreSum /= weightSum;

                // scale position that approach fifty-move rule
                if (gameScore == Game::Score::Draw && pos.GetHalfMoveCount() > 2)
                {
                    scoreSum = std::lerp(scoreSum, 0.5f, pos.GetHalfMoveCount() / 100.0f);
                }

                // blend between eval score and actual game score
                const float lambda = std::lerp(0.9f, 0.6f, gamePhase);
                entry.score = std::lerp(score, scoreSum, lambda);

                const float offset = 0.001f;
                entry.score = offset + entry.score * (1.0f - 2.0f * offset);

                Position normalizedPos = pos;
                if (pos.GetSideToMove() == Color::Black)
                {
                    // make whites side to move
                    normalizedPos = normalizedPos.SwappedColors();
                    entry.score = 1.0f - entry.score;
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

    std::cout << "Parsed " << numGames << " games from " << fileName << ", extracted " << numPositions << " positions" << std::endl;
    return true;
}

void LoadAllPositions(std::vector<PositionEntry>& outEntries)
{
    std::mutex mutex;
    std::vector<std::string> paths;

    const std::string gamesPath = "../../data/selfplayGames";
    for (const auto& path : std::filesystem::directory_iterator(gamesPath))
    {
        std::cout << "Loading " << path.path().string() << "..." << std::endl;
        paths.push_back(path.path().string());
    }

    Waitable waitable;
    {
        TaskBuilder taskBuilder(waitable);
        taskBuilder.ParallelFor("LoadPositions", uint32_t(paths.size()), [&](const TaskContext&, uint32_t i)
        {
            std::vector<PositionEntry> tempEntries;
            LoadPositions(paths[i].c_str(), tempEntries);

            {
                std::unique_lock<std::mutex> lock(mutex);
                outEntries.insert(outEntries.end(), tempEntries.begin(), tempEntries.end());
            }
        });
    }

    waitable.Wait();
}