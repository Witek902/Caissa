#include "Common.hpp"
#include "ThreadPool.hpp"
#include "TrainerCommon.hpp"
#include "GameCollection.hpp"

#include "../backend/Material.hpp"
#include "../backend/Waitable.hpp"
#include "../backend/Evaluate.hpp"
#include "../backend/Endgame.hpp"
#include "../backend/Tablebase.hpp"

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
            const float gamePhase = (float)i / (float)game.GetMoves().size();
            const Move move = pos.MoveFromPacked(game.GetMoves()[i]);
            const ScoreType moveScore = game.GetMoveScores()[i];

            const bool whitePawnsMoved = (pos.Whites().pawns & Bitboard::RankBitboard(1)) != Bitboard::RankBitboard(1);
            const bool blackPawnsMoved = (pos.Blacks().pawns & Bitboard::RankBitboard(6)) != Bitboard::RankBitboard(6);

            if (move.IsQuiet() &&
                pos.GetNumPieces() < 32 &&
                pos.GetNumPieces() >= 5 &&
                pos.GetHalfMoveCount() < 60 &&
                whitePawnsMoved && blackPawnsMoved &&
                !pos.IsInCheck() && pos.GetNumLegalMoves() &&
                std::abs(Evaluate(pos, nullptr, false)) < 1024)
            {
                PositionEntry entry{};

                int32_t wdl = 0;
                if (ProbeSyzygy_WDL(pos, &wdl))
                {
                    if (wdl > 0)        entry.score = std::lerp(entry.score, 1.0f, 0.8f);
                    else if (wdl < 0)   entry.score = std::lerp(entry.score, 0.0f, 0.8f);
                    else                entry.score = 0.5f;
                }
                else
                {
                    // blend in future scores into current move score
                    float scoreSum = 0.0f;
                    float weightSum = 0.0f;
                    const size_t maxLookahead = 10;
                    for (size_t j = 0; j < maxLookahead; ++j)
                    {
                        if (i + j >= game.GetMoves().size()) break;
                        const float weight = expf(-(float)j * 0.5f);
                        scoreSum += weight * CentiPawnToWinProbability(game.GetMoveScores()[i + j]);
                        weightSum += weight;
                    }
                    ASSERT(weightSum > 0.0f);
                    scoreSum /= weightSum;

                    // scale position that approach fifty-move rule
                    if (gameScore == Game::Score::Draw)
                    {
                        scoreSum = std::lerp(0.5f, scoreSum, Sqr(1.0f - pos.GetHalfMoveCount() / 100.0f));

                        if (pos.GetHalfMoveCount() > 40 && std::abs(moveScore) < 5)
                        {
                            break;
                        }
                    }

                    // blend between eval score and actual game score
                    const float lambda = std::lerp(0.95f, 0.8f, gamePhase);
                    entry.score = std::lerp(score, scoreSum, lambda);
                }

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

                if (std::abs(moveScore) >= KnownWinValue)
                {
                    break;
                }
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

    const std::string gamesPath = "../../data/selfplayGames";

    Waitable waitable;
    {
        TaskBuilder taskBuilder(waitable);

        for (const auto& path : std::filesystem::directory_iterator(gamesPath))
        {
            std::cout << "Loading " << path.path().string() << "..." << std::endl;

            taskBuilder.Task("LoadPositions", [path, &mutex, &outEntries](const TaskContext&)
            {
                std::vector<PositionEntry> tempEntries;
                LoadPositions(path.path().string().c_str(), tempEntries);

                {
                    std::unique_lock<std::mutex> lock(mutex);
                    outEntries.insert(outEntries.end(), tempEntries.begin(), tempEntries.end());
                }
            });
        }
    }

    waitable.Wait();
}