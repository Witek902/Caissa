#include "Common.hpp"
#include "ThreadPool.hpp"
#include "GameCollection.hpp"

#include "../backend/Position.hpp"
#include "../backend/Material.hpp"
#include "../backend/Game.hpp"
#include "../backend/Score.hpp"
#include "../backend/Move.hpp"
#include "../backend/Search.hpp"
#include "../backend/TranspositionTable.hpp"
#include "../backend/Evaluate.hpp"
#include "../backend/Endgame.hpp"
#include "../backend/Tablebase.hpp"
#include "../backend/Waitable.hpp"

#include <chrono>
#include <random>
#include <mutex>
#include <fstream>
#include <sstream>
#include <string>
#include <limits.h>

static const bool randomizeOrder = true;
static const uint32_t c_printPgnFrequency = 1;
static const uint32_t c_minNodes = 80000;
static const uint32_t c_maxNodes = 80000;
static const uint32_t c_maxDepth = 32;
static const int32_t c_maxEval = 3000;
static const int32_t c_openingMaxEval = 1000;
static const int32_t c_multiPv = 1;
static const int32_t c_multiPvMaxPly = 0;
static const int32_t c_multiPvScoreTreshold = 50;
static const uint32_t c_minRandomMoves = 2;
static const uint32_t c_maxRandomMoves = 2;

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
        if (!pos.FromFEN(line))
        {
            std::cout << "Invalid FEN string: " << line << std::endl;
            continue;
        }

        if (pos.GetNumPieces() > 32)
        {
            std::cout << "Too many pieces: " << line << std::endl;
            continue;
        }

        PackedPosition packedPos;
        PackPosition(pos, packedPos);
        outPositions.push_back(packedPos);
    }

    std::cout << "Loaded " << outPositions.size() << " opening positions" << std::endl;

    return true;
}

static Move GetRandomMove(std::mt19937& randomGenerator, const Position& pos)
{
    std::vector<Move> moves;
    pos.GetNumLegalMoves(&moves);

    // don't play losing moves (according to SEE)
    moves.erase(std::remove_if(moves.begin(),
        moves.end(),
        [&](const Move& move) { return !pos.StaticExchangeEvaluation(move); }),
        moves.end());

    if (moves.empty())
        return Move::Invalid();

    Move move = moves.front();
    if (moves.size() > 1)
    {
        std::uniform_int_distribution<size_t> distr(0, moves.size() - 1);
        move = moves[distr(randomGenerator)];
    }

    return move;
}

struct SelfPlayStats
{
    std::atomic<uint32_t> numWhiteWins = 0;
    std::atomic<uint32_t> numBlackWins = 0;
    std::atomic<uint32_t> numDraws = 0;
};

static bool SelfPlayThreadFunc(
    uint32_t nameSeed,
    uint32_t threadIndex,
    const std::vector<PackedPosition>& openingPositions,
    SelfPlayStats& stats)
{
    const size_t c_transpositionTableSize = 2ull * 1024ull * 1024ull;

    std::random_device rd;
    std::mt19937 gen(rd());

    Search search;
    TranspositionTable tt{ c_transpositionTableSize };

    const std::string outputFileName = DATA_PATH "selfplayGames/selfplay_" +
        std::to_string(nameSeed) + "_" +
        std::to_string(c_maxNodes / 1000) + "kn_" +
        "t" + std::to_string(threadIndex) + ".dat";

    FileOutputStream gamesFile(outputFileName.c_str());
    GameCollection::Writer writer(gamesFile);
    if (!writer.IsOK())
    {
        std::cerr << "Failed to open output file (games)!" << std::endl;
        return false;
    }

    uint32_t gameIndex = 0;

    for (;;)
    {
        SearchResult searchResult;

        // generate opening position
        Position openingPos;

        const uint32_t index = gameIndex++;

        if (!openingPositions.empty())
        {
            uint32_t openingIndex = index;
            if (randomizeOrder)
            {
                std::uniform_int_distribution<size_t> distrib(0, openingPositions.size() - 1);
                openingIndex = uint32_t(distrib(gen));
            }
            UnpackPosition(openingPositions[openingIndex], openingPos);
        }

        if constexpr (c_maxRandomMoves > 0)
        {
            // play few random moves in the opening
            const uint32_t numRandomMoves = std::uniform_int_distribution<uint32_t>(c_minRandomMoves, c_maxRandomMoves)(gen);
            for (uint32_t i = 0; i < numRandomMoves; ++i)
            {
                Move move = GetRandomMove(gen, openingPos);
                if (!move.IsValid())
                    break;

                const bool moveSuccess = openingPos.DoMove(move);
                ASSERT(moveSuccess);
                (void)moveSuccess;
            }
        }

        if (openingPos.IsMate() || openingPos.IsStalemate())
            continue;

        // start new game
        Game game;
        tt.Clear();
        search.Clear();
        game.Reset(openingPos);

        int32_t multiPvScoreTreshold = c_multiPvScoreTreshold;
        int32_t halfMoveNumber = 0;
        uint32_t drawScoreCounter = 0;
        uint32_t whiteWinsCounter = 0;
        uint32_t blackWinsCounter = 0;

        const uint32_t searchSeed = gen();

        for (;; ++halfMoveNumber)
        {
            SearchParam searchParam{ tt };
            searchParam.debugLog = false;
            searchParam.useRootTablebase = false;
            searchParam.evalRandomization = 1;
            searchParam.seed = searchSeed;
            searchParam.numPvLines = (halfMoveNumber < c_multiPvMaxPly) ? c_multiPv : 1;
            searchParam.limits.maxDepth = c_maxDepth;
            searchParam.limits.maxNodesSoft = c_minNodes + (c_maxNodes - c_minNodes) * std::max(0, 80 - halfMoveNumber) / 80;
            searchParam.limits.maxNodes = 5 * searchParam.limits.maxNodesSoft;

            searchResult.clear();
            tt.NextGeneration();
            search.DoSearch(game, searchParam, searchResult);

            ASSERT(!searchResult.empty());

            // skip game if starting position is unbalanced
            if (halfMoveNumber == 0 && std::abs(searchResult.begin()->score) > c_openingMaxEval)
                break;

            // sort moves by score
            std::sort(searchResult.begin(), searchResult.end(), [](const PvLine& a, const PvLine& b)
            {
                return a.score > b.score;
            });

            // if one of the move is much worse than the best candidate, ignore it and the rest
            for (size_t i = 1; i < searchResult.size(); ++i)
            {
                ASSERT(searchResult[i].score <= searchResult[0].score);
                const int32_t diff = std::abs((int32_t)searchResult[i].score - (int32_t)searchResult[0].score);
                if (diff > multiPvScoreTreshold)
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
            Move move = searchResult[moveIndex].moves.front();

            // reduce threshold of picking worse move
            // this way the game will be more random at the beginning and there will be less blunders later in the game
            multiPvScoreTreshold = std::max(10, multiPvScoreTreshold - 2);

            ScoreType moveScore = searchResult[moveIndex].score;
            ScoreType eval = Evaluate(game.GetPosition());

            if (game.GetSideToMove() == Black)
            {
                moveScore = -moveScore;
                eval = -eval;
            }

            const bool isCheck = game.GetPosition().IsInCheck();

            const bool moveSuccess = game.DoMove(move, moveScore);
            ASSERT(moveSuccess);
            (void)moveSuccess;

            if (std::abs(moveScore) < 4)
                drawScoreCounter++;
            else
                drawScoreCounter = 0;

            // adjudicate draw if eval is zero
            if (drawScoreCounter > 8 && halfMoveNumber >= 60)
            {
                game.SetScore(Game::Score::Draw);
            }

            // adjudicate win
            if (halfMoveNumber >= 20)
            {
                if (moveScore > c_maxEval && eval > c_maxEval / 4)
                {
                    whiteWinsCounter++;
                    if (whiteWinsCounter > 3) game.SetScore(Game::Score::WhiteWins);
                }
                else
                {
                    whiteWinsCounter = 0;
                }

                if (moveScore < -c_maxEval && eval < -c_maxEval / 4)
                {
                    blackWinsCounter++;
                    if (blackWinsCounter > 3) game.SetScore(Game::Score::BlackWins);
                }
                else
                {
                    blackWinsCounter = 0;
                }
            }

            // tablebase adjudication
            int32_t wdlScore = 0;
            if (!isCheck && ProbeSyzygy_WDL(game.GetPosition(), &wdlScore))
            {
                const auto stm = game.GetPosition().GetSideToMove();
                if (wdlScore == 1) game.SetScore(stm == White ? Game::Score::WhiteWins : Game::Score::BlackWins);
                if (wdlScore == 0) game.SetScore(Game::Score::Draw);
                if (wdlScore == -1) game.SetScore(stm == White ? Game::Score::BlackWins : Game::Score::WhiteWins);
            }

            if (game.GetPosition().IsMate())
            {
                ASSERT(moveScore >= TablebaseWinValue || moveScore <= -TablebaseWinValue);
            }

            if (game.GetScore() != Game::Score::Unknown)
            {
                if (game.GetScore() == Game::Score::WhiteWins) stats.numWhiteWins++;
                if (game.GetScore() == Game::Score::BlackWins) stats.numBlackWins++;
                if (game.GetScore() == Game::Score::Draw) stats.numDraws++;
                break;
            }
        }

        // save game
        if (halfMoveNumber > 0)
        {
            writer.WriteGame(game);

            GameMetadata metadata;
            metadata.roundNumber = index;
            game.SetMetadata(metadata);

            if (threadIndex == 0 && c_printPgnFrequency != 0 && (index % c_printPgnFrequency == 0))
            {
                const std::string pgn = game.ToPGN(true);
                std::cout << std::endl << pgn << std::endl;

                const uint32_t numGames = stats.numWhiteWins + stats.numBlackWins + stats.numDraws;
                std::cout << std::endl;
                std::cout << "White wins: " << stats.numWhiteWins << " (" << (stats.numWhiteWins * 100.0 / numGames) << "%)" << std::endl;
                std::cout << "Black wins: " << stats.numBlackWins << " (" << (stats.numBlackWins * 100.0 / numGames) << "%)" << std::endl;
                std::cout << "Draws:      " << stats.numDraws << " (" << (stats.numDraws * 100.0 / numGames) << "%)" << std::endl;
            }

            if (index % 64 == 0)
            {
                gamesFile.Flush();
            }
        }
    }

    return true;
}

void SelfPlay(const std::vector<std::string>& args)
{
    g_syzygyProbeLimit = 7;

    uint32_t nameSeed = 0;
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<uint32_t> distrib;
        nameSeed = distrib(gen);
    }

    std::cout << "Loading opening positions..." << std::endl;
    std::vector<PackedPosition> openingPositions;
    if (!args.empty())
    {
        LoadOpeningPositions(args[0], openingPositions);
    }

    alignas(CACHELINE_SIZE) SelfPlayStats stats;

    std::cout << "Starting games..." << std::endl;

    const uint32_t numThreads = std::max<uint32_t>(1, std::thread::hardware_concurrency());

    std::vector<std::thread> threads;
    for (uint32_t i = 0; i < numThreads; ++i)
    {
        threads.emplace_back([i, nameSeed, &openingPositions, &stats]()
        {
            SelfPlayThreadFunc(nameSeed, i, openingPositions, stats);
        });
    }

    for (auto& thread : threads)
    {
        thread.join();
    }
}
