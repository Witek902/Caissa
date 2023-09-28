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

static const bool writeQuietPositions = false;
static const bool probePositions = false;
static const bool randomizeOrder = true;
static const uint32_t c_printPgnFrequency = 32;
static const uint32_t c_minNodes = 8000;
static const uint32_t c_maxNodes = 12000;
static const uint32_t c_maxDepth = 24;
static const int32_t c_maxEval = 1200;
static const int32_t c_openingMaxEval = 800;
static const int32_t c_multiPv = 3;
static const int32_t c_multiPvMaxPly = 0;
static const int32_t c_multiPvScoreTreshold = 50;
static const uint32_t c_minRandomMoves = 8;
static const uint32_t c_maxRandomMoves = 12;
static const float c_blunderProbability = 0.01f;

using namespace threadpool;

uint32_t XorShift32(uint32_t state)
{
    /* Algorithm "xor" from p. 4 of Marsaglia, "Xorshift RNGs" */
    uint32_t x = state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return x;
}

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

void SelfPlay(const std::vector<std::string>& args)
{
    g_syzygyProbeLimit = 7;

    std::string outputFileName;
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<uint32_t> distrib;
        outputFileName = DATA_PATH "selfplayGames/selfplay_" + std::to_string(distrib(gen)) + ".dat";
    }

    FileOutputStream gamesFile(outputFileName.c_str());
    GameCollection::Writer writer(gamesFile);
    if (!writer.IsOK())
    {
        std::cerr << "Failed to open output file (games)!" << std::endl;
        return;
    }

    std::ofstream probedPositionsFile;
    if (probePositions)
    {
        probedPositionsFile.open("probed.epd");
        if (!probedPositionsFile.good())
        {
            std::cerr << "Failed to open output file (probed positions)!" << std::endl;
            return;
        }
    }

    std::ofstream quietPositionsFile;
    if (writeQuietPositions)
    {
        quietPositionsFile.open("quietPositions.epd");
        if (!quietPositionsFile.good())
        {
            std::cerr << "Failed to open quiet positions file!" << std::endl;
            return;
        }
    }

    const size_t numThreads = ThreadPool::GetInstance().GetNumThreads();

    std::vector<Search> searchArray{ numThreads };
    std::vector<TranspositionTable> ttArray;

    const size_t c_transpositionTableSize = 8ull * 1024ull * 1024ull;

    std::cout << "Allocating transposition table..." << std::endl;
    for (size_t i = 0; i < numThreads; ++i)
    {
        ttArray.emplace_back(c_transpositionTableSize);
    }

    std::cout << "Loading opening positions..." << std::endl;
    std::vector<PackedPosition> openingPositions;
    if (args.size() > 1)
    {
        LoadOpeningPositions(args[1], openingPositions);
    }

    std::cout << "Starting games..." << std::endl;

    alignas(CACHELINE_SIZE) std::mutex mutex;
    alignas(CACHELINE_SIZE) std::atomic<uint32_t> gameIndex = 0;
    alignas(CACHELINE_SIZE) std::atomic<uint64_t> quietPositionIndex = 0;
    std::atomic<uint32_t> numWhiteWins = 0;
    std::atomic<uint32_t> numBlackWins = 0;
    std::atomic<uint32_t> numDraws = 0;

    const auto gameTask = [&](const TaskContext& context, uint32_t)
    {
        std::random_device rd;
        std::mt19937 gen(rd());

        Search& search = searchArray[context.threadId];
        TranspositionTable& tt = ttArray[context.threadId];

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

        /*
        // randomize side to move
        if (!openingPos.IsInCheck() &&
            std::uniform_int_distribution<uint32_t>(0, 1)(gen) == 0)
        {
            openingPos.DoNullMove();
        }
        */

        if (openingPos.IsMate() || openingPos.IsStalemate())
        {
            return;
        }

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
            const bool playBlunder = std::bernoulli_distribution(c_blunderProbability)(gen);

            SearchParam searchParam{ tt };
            searchParam.debugLog = false;
            searchParam.useRootTablebase = false;
            searchParam.evalRandomization = 2;
            searchParam.seed = searchSeed;
            searchParam.numPvLines = (halfMoveNumber < c_multiPvMaxPly || playBlunder) ? c_multiPv : 1;
            searchParam.limits.maxDepth = c_maxDepth;
            searchParam.limits.maxNodes = c_minNodes + (c_maxNodes - c_minNodes) * std::max(0, 80 - halfMoveNumber) / 80;

            searchResult.clear();
            tt.NextGeneration();
            search.DoSearch(game, searchParam, searchResult);

            if (searchResult.empty())
            {
                return;
            }

            // skip game if starting position is unbalanced
            if (halfMoveNumber == 0 && std::abs(searchResult.begin()->score) > c_openingMaxEval)
            {
                return;
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

            ScoreType moveScore = searchResult[moveIndex].score;
            if (game.GetSideToMove() == Color::Black)
            {
                moveScore = -moveScore;
            }

            if (writeQuietPositions)
            {
                if (move.IsQuiet() && game.GetPosition().GetNumPieces() >= 5)
                {
                    std::unique_lock<std::mutex> lock(mutex);

                    quietPositionsFile << game.GetPosition().ToFEN() << '\n';

                    quietPositionIndex++;
                    if (quietPositionIndex % 10000 == 0)
                    {
                        std::cout << '\r' << "Generated " << quietPositionIndex << " quiet positions";
                    }
                }
            }

            // reduce threshold of picking worse move
            // this way the game will be more random at the beginning and there will be less blunders later in the game
            multiPvScoreTreshold = std::max(10, multiPvScoreTreshold - 2);

            const ScoreType eval = Evaluate(game.GetPosition());
            const bool isCheck = game.GetPosition().IsInCheck();

            const bool moveSuccess = game.DoMove(move, moveScore);
            ASSERT(moveSuccess);
            (void)moveSuccess;

            if (std::abs(moveScore) < 5)
                drawScoreCounter++;
            else
                drawScoreCounter = 0;

            // adjudicate draw if eval is zero
            if (drawScoreCounter > 8 &&
                halfMoveNumber >= 40 &&
                game.GetPosition().GetHalfMoveCount() > 10)
            {
                game.SetScore(Game::Score::Draw);
            }

            // adjudicate win
            if (!isCheck &&
                game.GetPosition().GetNumPieces() < 20 &&
                halfMoveNumber >= 20)
            {
                if (moveScore > c_maxEval && eval > c_maxEval / 2)
                {
                    whiteWinsCounter++;
                    if (whiteWinsCounter > 4) game.SetScore(Game::Score::WhiteWins);
                }
                else
                {
                    whiteWinsCounter = 0;
                }

                if (moveScore < -c_maxEval && eval < -c_maxEval / 2)
                {
                    blackWinsCounter++;
                    if (blackWinsCounter > 4) game.SetScore(Game::Score::BlackWins);
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
                if (wdlScore == 1) game.SetScore(stm == Color::White ? Game::Score::WhiteWins : Game::Score::BlackWins);
                if (wdlScore == 0) game.SetScore(Game::Score::Draw);
                if (wdlScore == -1) game.SetScore(stm == Color::White ? Game::Score::BlackWins : Game::Score::WhiteWins);
            }

            if (game.GetPosition().IsMate())
            {
                ASSERT(moveScore >= TablebaseWinValue || moveScore <= -TablebaseWinValue);
            }

            if (game.GetScore() != Game::Score::Unknown)
            {
                if (game.GetScore() == Game::Score::WhiteWins) numWhiteWins++;
                if (game.GetScore() == Game::Score::BlackWins) numBlackWins++;
                if (game.GetScore() == Game::Score::Draw) numDraws++;
                break;
            }
        }

        {
            std::unique_lock<std::mutex> lock(mutex);

            writer.WriteGame(game);

            GameMetadata metadata;
            metadata.roundNumber = index;
            game.SetMetadata(metadata);

            if (c_printPgnFrequency != 0 && (index % c_printPgnFrequency == 0))
            {
                const std::string pgn = game.ToPGN(true);
                std::cout << std::endl << pgn << std::endl;

                const uint32_t numGames = numWhiteWins + numBlackWins + numDraws;
                std::cout << std::endl;
                std::cout << "White wins: " << numWhiteWins << " (" << (numWhiteWins * 100.0 / numGames) << "%)" << std::endl;
                std::cout << "Black wins: " << numBlackWins << " (" << (numBlackWins * 100.0 / numGames) << "%)" << std::endl;
                std::cout << "Draws:      " << numDraws << " (" << (numDraws * 100.0 / numGames) << "%)" << std::endl;
            }
        }
    };

    // TODO could just spawn threads instead...
    for (;;)
    {
        Waitable waitable;
        {
            const uint32_t numGamesPerLoop = 16 * 1024;

            TaskBuilder taskBuilder(waitable);
            taskBuilder.ParallelFor("SelfPlay", numGamesPerLoop, gameTask);
        }
        waitable.Wait();
    }

#ifdef COLLECT_ENDGAME_STATISTICS
    PrintEndgameStatistics();
#endif // COLLECT_ENDGAME_STATISTICS
}
