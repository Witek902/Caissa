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
#include <limits.h>

using namespace threadpool;

void SelfPlay()
{
    FileOutputStream gamesFile("selfplay.dat");
    GameCollection::Writer writer(gamesFile);

    TranspositionTable tt(2ull * 1024ull * 1024ull * 1024ull);

    std::vector<Search> searchArray{ std::thread::hardware_concurrency() };
    
    std::mutex mutex;
    uint32_t games = 0;
    uint32_t whiteWins = 0;
    uint32_t blackWins = 0;
    uint32_t draws = 0;

    Waitable waitable;
    {
        TaskBuilder taskBuilder(waitable);

        taskBuilder.ParallelFor("SelfPlay", 200, [&](const TaskContext& context, uint32_t)
        {
            std::random_device rd;
            std::mt19937 gen(rd());

            Search& search = searchArray[context.threadId];

            Game game;
            game.Reset(Position(Position::InitPositionFEN));

            SearchResult searchResult;

            int32_t score = 0;

            int32_t scoreDiffTreshold = 10;
            uint32_t maxMoves = 500;

            uint32_t halfMoveNumber = 0;
            for (;; ++halfMoveNumber)
            {
                const TimePoint startTimePoint = TimePoint::GetCurrent();

                SearchParam searchParam{ tt };
                searchParam.limits.maxDepth = 20;
                searchParam.numPvLines = halfMoveNumber > 20 ? 1 : 2;
                searchParam.debugLog = false;
                searchParam.limits.maxNodes = 10000;
                //searchParam.limits.maxTime = startTimePoint + TimePoint::FromSeconds(0.4f);
                //searchParam.limits.maxTimeSoft = startTimePoint + TimePoint::FromSeconds(0.1f);
                //searchParam.limits.rootSingularityTime = startTimePoint + TimePoint::FromSeconds(0.03f);

                searchResult.clear();

                search.DoSearch(game, searchParam, searchResult);

                if (searchResult.empty())
                {
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
                score = searchResult[moveIndex].score;
                if (game.GetSideToMove() == Color::Black) score = -score;

                // if didn't picked best move, reduce treshold of picking worse move
                // this way the game will be more random at the beginning and there will be less blunders later in the game
                if (moveIndex > 0)
                {
                    scoreDiffTreshold = std::max(10, scoreDiffTreshold - 5);
                }

                const bool moveSuccess = game.DoMove(move);
                ASSERT(moveSuccess);
                (void)moveSuccess;

                // check for draw
                if (game.IsDrawn() || halfMoveNumber > maxMoves)
                {
                    score = 0;
                    break;
                }
            }

            writer.WriteGame(game);

            {
                const std::string pgn = game.ToPGN();

                std::unique_lock<std::mutex> lock(mutex);

                const uint32_t gameNumber = games++;

                std::cout << "Game #" << gameNumber << " " << pgn;

                if (score > 0)
                {
                    std::cout << " (white won)";
                    whiteWins++;
                }
                else if (score < 0)
                {
                    std::cout << " (black won)";
                    blackWins++;
                }
                else
                {
                    if (game.GetRepetitionCount(game.GetPosition()) >= 2) std::cout << "(draw by repetition)";
                    else if (game.GetPosition().GetHalfMoveCount() >= 100) std::cout << "(draw by 50 move rule)";
                    else if (CheckInsufficientMaterial(game.GetPosition())) std::cout << "(draw by insufficient material)";
                    else std::cout << "(draw by too long game)";

                    draws++;
                }

                std::cout << " W:" << whiteWins << " B:" << blackWins << " D:" << draws << std::endl;
            }
        });

        //taskBuilder.Fence();
    }

    waitable.Wait();

#ifdef COLLECT_ENDGAME_STATISTICS
    PrintEndgameStatistics();
#endif // COLLECT_ENDGAME_STATISTICS
}
