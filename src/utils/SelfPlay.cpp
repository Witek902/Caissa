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
static const bool printGames = true;
static const uint32_t c_maxNodes = 40000;
static const uint32_t c_maxDepth = 12;
static const int32_t c_maxEval = 2000;
static const int32_t c_openingMaxEval = 500;
static const int32_t c_multiPv = 2;
static const int32_t c_multiPvMaxPly = 10;
static const int32_t c_multiPvScoreTreshold = 20;
static const uint32_t c_numRandomMoves = 4;

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

class EvalProbing : public EvalProbingInterface
{
public:
    static constexpr uint32_t ProbingFrequency = 1 << 10;

    EvalProbing(std::ofstream& outputFile, uint32_t seed)
        : probedPositionsFile(outputFile)
        , randomSeed(seed)
    {}

    virtual void ReportPosition(const Position& pos, ScoreType eval) override
    {
        if (!(pos.Whites().GetKingSquare().Rank() == 7 ||
              pos.Blacks().GetKingSquare().Rank() == 0)) return;

        if (std::abs(eval) >= 200) return;

        uint32_t numPieces = pos.GetNumPieces();

        if (numPieces < 14) return;

        randomSeed = XorShift32(randomSeed);

        if (!pos.IsQuiet()) return;

        PackedPosition packedPos;
        PackPosition(pos, packedPos);
        positions.push_back(packedPos);
    }

    void Flush()
    {
        for (const PackedPosition& pp : positions)
        {
            Position pos;
            UnpackPosition(pp, pos);
            probedPositionsFile << pos.ToFEN() << '\n';
        }

        positions.clear();

        probedPositionsFile.flush();
    }

private:
    std::ofstream& probedPositionsFile;
    std::vector<PackedPosition> positions;

    uint32_t randomSeed;
};

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

bool ApplyRandomMove(std::mt19937& randomGenerator, Position& pos)
{
    std::vector<Move> moves;
    pos.GetNumLegalMoves(&moves);

    if (moves.empty()) return false;

    Move move = moves.front();
    if (moves.size() > 1)
    {
        std::uniform_int_distribution<size_t> distr(0, moves.size() - 1);
        move = moves[distr(randomGenerator)];
    }

    return pos.DoMove(move);
}

void SelfPlay(const std::vector<std::string>& args)
{
    FileOutputStream gamesFile("selfplay.dat");
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

    std::cout << "Allocating transposition table..." << std::endl;
    for (size_t i = 0; i < numThreads; ++i)
    {
        ttArray.emplace_back(16ull * 1024ull * 1024ull);
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

    const auto gameTask = [&](const TaskContext& context, uint32_t)
    {
        std::random_device rd;
        std::mt19937 gen(rd());

        Search& search = searchArray[context.threadId];
        TranspositionTable& tt = ttArray[context.threadId];

        SearchResult searchResult;
        EvalProbing evalProbing(probedPositionsFile, gen());

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

        for (uint32_t i = 0; i < c_numRandomMoves; ++i)
        {
            ApplyRandomMove(gen, openingPos);
            ApplyRandomMove(gen, openingPos);
        }

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
        uint32_t halfMoveNumber = 0;
        uint32_t drawScoreCounter = 0;
        uint32_t whiteWinsCounter = 0;
        uint32_t blackWinsCounter = 0;

        for (;; ++halfMoveNumber)
        {
            SearchParam searchParam{ tt };
            searchParam.debugLog = false;
            searchParam.useRootTablebase = false;
            searchParam.useAspirationWindows = false; // disable aspiration windows so we don't get lower/upper bound scores
            searchParam.evalProbingInterface = probePositions ? &evalProbing : nullptr;
            searchParam.limits.maxDepth = c_maxDepth;
            searchParam.limits.maxNodes = c_maxNodes + std::uniform_int_distribution<int32_t>(0, c_maxNodes / 32)(gen);
            searchParam.numPvLines = halfMoveNumber < c_multiPvMaxPly ? c_multiPv : 1;

            if (halfMoveNumber == 0)
            {
                searchParam.limits.maxDepth *= 2;
                searchParam.limits.maxNodes *= 2;
            }

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
            const Move move = searchResult[moveIndex].moves.front();

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
            multiPvScoreTreshold = std::max(10, multiPvScoreTreshold - 1);

            const bool moveSuccess = game.DoMove(move, moveScore);
            ASSERT(moveSuccess);
            (void)moveSuccess;

            // adjudicate draw if eval is zero
            if (std::abs(moveScore) >= 2)
            {
                drawScoreCounter = 0;
            }
            else
            {
                drawScoreCounter++;

                if (game.GetPosition().GetNumPieces() < 20 &&
                    drawScoreCounter > 10 &&
                    halfMoveNumber >= 40 &&
                    game.GetPosition().GetHalfMoveCount() > 10)
                {
                    game.SetScore(Game::Score::Draw);
                }
            }

            // adjudicate win
            if (game.GetPosition().GetNumPieces() < 24 && halfMoveNumber >= 20)
            {
                if (moveScore > c_maxEval)
                {
                    whiteWinsCounter++;
                    if (whiteWinsCounter > 4) game.SetScore(Game::Score::WhiteWins);
                }
                else
                {
                    whiteWinsCounter = 0;
                }

                if (moveScore < -c_maxEval)
                {
                    blackWinsCounter++;
                    if (blackWinsCounter > 4) game.SetScore(Game::Score::BlackWins);
                }
                else
                {
                    blackWinsCounter = 0;
                }
            }

            if (game.GetPosition().IsMate())
            {
                ASSERT(moveScore >= TablebaseWinValue || moveScore <= -TablebaseWinValue);
            }

            if (game.GetScore() != Game::Score::Unknown)
            {
                break;
            }
        }

        {
            std::unique_lock<std::mutex> lock(mutex);

            writer.WriteGame(game);

            if (probePositions)
            {
                evalProbing.Flush();
            }

            GameMetadata metadata;
            metadata.roundNumber = index;
            game.SetMetadata(metadata);

            if (printGames)
            {
                const std::string pgn = game.ToPGN(true);
                std::cout << std::endl << pgn << std::endl;
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
