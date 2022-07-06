#include "Common.hpp"
#include "ThreadPool.hpp"
#include "GameCollection.hpp"

#include "../backend/Position.hpp"
#include "../backend/Game.hpp"
#include "../backend/Move.hpp"
#include "../backend/Search.hpp"
#include "../backend/TranspositionTable.hpp"
#include "../backend/Evaluate.hpp"
#include "../backend/Endgame.hpp"
#include "../backend/Tablebase.hpp"
#include "../backend/NeuralNetwork.hpp"
#include "../backend/Waitable.hpp"

#include <chrono>
#include <random>
#include <mutex>
#include <fstream>
#include <sstream>
#include <string>
#include <limits.h>

using namespace threadpool;

bool LoadOpeningPositions(const std::string& path, std::vector<PackedPosition>& outPositions)
{
    std::ifstream file(path);
    if (!file.good())
    {
        std::cout << "Failed to load opening positions file " << path << std::endl;
        return false;
    }

    std::string line;
    while (std::getline(file, line))
    {
        Position pos;
        if (!pos.FromFEN(line)) continue;

        PackedPosition packedPos;
        PackPosition(pos, packedPos);
        outPositions.push_back(packedPos);
    }

    std::cout << "Loaded " << outPositions.size() << " opening positions" << std::endl;

    return true;
}

void SelfPlay(const std::vector<std::string>& args)
{
    FileOutputStream gamesFile("selfplay.dat");
    GameCollection::Writer writer(gamesFile);

    const size_t numThreads = ThreadPool::GetInstance().GetNumThreads();

    std::vector<Search> searchArray{ numThreads };
    std::vector<TranspositionTable> ttArray;

    std::cout << "Allocating transposition table..." << std::endl;
    ttArray.resize(numThreads);
    for (size_t i = 0; i < numThreads; ++i)
    {
        ttArray[i].Resize(32ull * 1024ull * 1024ull);
    }

    std::cout << "Loading opening positions..." << std::endl;
    std::vector<PackedPosition> openingPositions;
    if (args.size() > 0)
    {
        LoadOpeningPositions(args[0], openingPositions);
    }
    
    std::mutex mutex;
    uint32_t games = 0;

    std::cout << "Starting games..." << std::endl;

    Waitable waitable;
    {
        TaskBuilder taskBuilder(waitable);

        taskBuilder.ParallelFor("SelfPlay", 1000000, [&](const TaskContext& context, uint32_t)
        {
            std::random_device rd;
            std::mt19937 gen(rd());

            Search& search = searchArray[context.threadId];
            TranspositionTable& tt = ttArray[context.threadId];

            // start new game
            Game game;
            tt.Clear();
            search.Clear();

            // generate opening position
            Position openingPos(Position::InitPositionFEN);
            if (!openingPositions.empty())
            {
                std::uniform_int_distribution<size_t> distrib(0, openingPositions.size() - 1);
                UnpackPosition(openingPositions[distrib(gen)], openingPos);
            }
            game.Reset(openingPos);

            SearchResult searchResult;

            int32_t scoreDiffTreshold = 20;

            uint32_t halfMoveNumber = 0;
            for (;; ++halfMoveNumber)
            {
                const TimePoint startTimePoint = TimePoint::GetCurrent();

                SearchParam searchParam{ tt };
                searchParam.debugLog = false;
                searchParam.limits.maxNodes = 100000 + std::uniform_int_distribution<int32_t>(0, 10000)(gen);
                //searchParam.limits.maxTime = startTimePoint + TimePoint::FromSeconds(0.2f);
                //searchParam.limits.idealTime = startTimePoint + TimePoint::FromSeconds(0.06f);
                //searchParam.limits.rootSingularityTime = startTimePoint + TimePoint::FromSeconds(0.02f);

                searchResult.clear();
                tt.NextGeneration();
                search.DoSearch(game, searchParam, searchResult);

                if (searchResult.empty())
                {
                    DEBUG_BREAK();
                    break;
                }

                // sort moves by score
                std::sort(searchResult.begin(), searchResult.end(), [](const PvLine& a, const PvLine& b)
                {
                    return a.score > b.score;
                });

                // if one of the move is much worse than the best candidate, ignore it and the rest
                for (size_t i = 1; i < searchResult.size(); ++i)
                {
                    ASSERT(searchResult[i].score <= searchResult[0].score);
                    int32_t diff = searchResult[i].score - searchResult[0].score;
                    if (diff > scoreDiffTreshold || diff < -scoreDiffTreshold)
                    {
                        searchResult.erase(searchResult.begin() + i, searchResult.end());
                        break;
                    }
                }

                // select random move
                // TODO prefer moves with higher score
                std::uniform_int_distribution<size_t> distrib(0, searchResult.size() - 1);
                const size_t moveIndex = distrib(gen);
                ASSERT(!searchResult[moveIndex].moves.empty());
                const Move move = searchResult[moveIndex].moves.front();

                ScoreType moveScore = searchResult[moveIndex].score;
                if (game.GetSideToMove() == Color::Black) moveScore = -moveScore;

                // don't play forced mate sequences
                if (moveScore > CheckmateValue - MaxSearchDepth)
                {
                    game.SetScore(Game::Score::WhiteWins);
                    break;
                }
                else if (moveScore < -CheckmateValue + MaxSearchDepth)
                {
                    game.SetScore(Game::Score::BlackWins);
                    break;
                }

                // reduce treshold of picking worse move
                // this way the game will be more random at the beginning and there will be less blunders later in the game
                scoreDiffTreshold = std::max(5, scoreDiffTreshold - 1);

                const bool moveSuccess = game.DoMove(move, moveScore);
                ASSERT(moveSuccess);
                (void)moveSuccess;

                if (game.GetScore() != Game::Score::Unknown)
                {
                    break;
                }
            }

            writer.WriteGame(game);

            {
                std::unique_lock<std::mutex> lock(mutex);

                const uint32_t gameNumber = ++games;

                GameMetadata metadata;
                metadata.roundNumber = gameNumber;
                game.SetMetadata(metadata);

                const std::string pgn = game.ToPGN();

                std::cout << std::endl << pgn << std::endl;
            }
        });

        //taskBuilder.Fence();
    }

    waitable.Wait();

#ifdef COLLECT_ENDGAME_STATISTICS
    PrintEndgameStatistics();
#endif // COLLECT_ENDGAME_STATISTICS
}
