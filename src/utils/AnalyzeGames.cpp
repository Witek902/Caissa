#include "Common.hpp"
#include "GameCollection.hpp"
#include "ThreadPool.hpp"

#include "../backend/Position.hpp"
#include "../backend/Material.hpp"
#include "../backend/Game.hpp"
#include "../backend/Move.hpp"
#include "../backend/Search.hpp"
#include "../backend/TranspositionTable.hpp"
#include "../backend/Evaluate.hpp"
#include "../backend/Tablebase.hpp"
#include "../backend/Waitable.hpp"
#include "../backend/Time.hpp"

#include <filesystem>
#include <chrono>
#include <random>
#include <mutex>
#include <fstream>
#include <limits.h>
#include <iomanip>

static const bool c_collectMaterialStats = false;
static const bool c_dumpFortressPositions = false;
static const bool c_dumpKingOnFarRankPositions = false;

float GameScoreToExpectedGameScore(const Game::Score score)
{
    switch (score)
    {
    case Game::Score::BlackWins: return 0.0f;
    case Game::Score::WhiteWins: return 1.0f;
    default: return 0.5f;
    }
}

struct MaterialStats
{
    uint64_t wins;
    uint64_t draws;
    uint64_t losses;

    double avgEvalScore = 0.0;

    uint64_t NumPositions() const { return wins + draws + losses; }
};

struct GamesStats
{
    std::mutex mutex;

    std::ofstream fortressPosition;
    std::ofstream kingOnFarRankPositions;

    std::unordered_map<MaterialKey, MaterialStats> materialStats;

    uint64_t numGames = 0;
    uint64_t numPositions = 0;
    uint64_t numPawnlessPositions = 0;

    uint64_t pieceOccupancy[6][64] = { };
    uint64_t gameResultVsHalfMoveCounter[3][101] = { };

    // sum of squared eval errors (relative to WDL and search score)
    double evalErrorSum_WDL = 0.0;
    double evalErrorSum_Score = 0.0;
};

void AnalyzeGames(const char* path, GamesStats& outStats)
{
    std::cout << "Reading " << path << "..." << std::endl;

    FileInputStream gamesFile(path);

    GamesStats localStats;

    Game game;
    std::vector<Move> moves;

    while (GameCollection::ReadGame(gamesFile, game, moves))
    {
        Position pos = game.GetInitialPosition();

        if (game.GetScore() == Game::Score::Unknown) continue;

        ASSERT(game.GetMoves().size() == game.GetMoveScores().size());

        for (size_t i = 0; i < game.GetMoves().size(); ++i)
        {
            const Move move = pos.MoveFromPacked(game.GetMoves()[i]);
            const ScoreType moveScore = game.GetMoveScores()[i];

            if (move.IsQuiet() &&
                pos.GetNumPieces() >= 4 &&
                !pos.IsInCheck())
            {
                const ScoreType staticEval = Evaluate(pos);

                // skip unbalanced positions
                if (std::abs(moveScore) < 800 || std::abs(Evaluate(pos)) < 2000)
                {
                    const MaterialKey matKey = pos.GetMaterialKey();

                    if (pos.GetHalfMoveCount() <= 100)
                    {
                        localStats.gameResultVsHalfMoveCounter[(uint32_t)game.GetScore()][pos.GetHalfMoveCount()]++;
                    }

                    const float moveScoreAsGameScore = InternalEvalToExpectedGameScore(moveScore);
                    const float staticEvalAsGameScore = InternalEvalToExpectedGameScore(staticEval);

                    if (c_collectMaterialStats)
                    {
                        MaterialStats& matStats = localStats.materialStats[matKey];
                        matStats.wins += (game.GetScore() == Game::Score::WhiteWins ? 1 : 0);
                        matStats.draws += (game.GetScore() == Game::Score::Draw ? 1 : 0);
                        matStats.losses += (game.GetScore() == Game::Score::BlackWins ? 1 : 0);
                        matStats.avgEvalScore += moveScoreAsGameScore;
                    }

                    localStats.evalErrorSum_Score += Sqr(staticEvalAsGameScore - moveScoreAsGameScore);
                    localStats.evalErrorSum_WDL += Sqr(staticEvalAsGameScore - GameScoreToExpectedGameScore(game.GetScore()));

                    localStats.numPositions++;
                    if (matKey.numWhitePawns == 0 && matKey.numBlackPawns == 0) localStats.numPawnlessPositions++;

                    // piece occupancy
                    for (uint32_t pieceIndex = 0; pieceIndex < 6; ++pieceIndex)
                    {
                        const Piece piece = (Piece)(pieceIndex + (uint32_t)Piece::Pawn);
                        pos.Whites().GetPieceBitBoard(piece).Iterate([&](const uint32_t square) INLINE_LAMBDA{
                            localStats.pieceOccupancy[pieceIndex][square]++; });
                        pos.Blacks().GetPieceBitBoard(piece).Iterate([&](const uint32_t square) INLINE_LAMBDA{
                            localStats.pieceOccupancy[pieceIndex][Square(square).FlippedRank().Index()]++; });
                    }
                }
            }

            // dump potential fortress positions
            if (c_dumpFortressPositions)
            {
                const int32_t fortressTreshold = 300;
                int32_t wdl = 0;

                if (move.IsQuiet() &&
                    pos.GetNumPieces() <= 7 && pos.GetNumPieces() >= 4 &&
                    pos.GetHalfMoveCount() > 20)
                {
                    const ScoreType eval = Evaluate(pos);
                    if ((eval > fortressTreshold && moveScore > fortressTreshold) ||
                        (eval < -fortressTreshold && moveScore < -fortressTreshold))
                    {
                        if (ProbeSyzygy_WDL(pos, &wdl) && wdl == 0)
                        {
                            std::unique_lock<std::mutex> lock(outStats.mutex);
                            outStats.fortressPosition << pos.ToFEN() << std::endl;
                            break;
                        }
                    }
                }
            }

            // dump positions where the king is on the far rank
            if (c_dumpKingOnFarRankPositions)
            {
                if (pos.GetNumPieces() >= 16 && std::abs(moveScore) < 400)
                {
                    if (pos.Whites().GetKingSquare().Rank() >= 4 ||
                        pos.Blacks().GetKingSquare().Rank() <= 3)
                    {
                        std::unique_lock<std::mutex> lock(outStats.mutex);
                        outStats.kingOnFarRankPositions << pos.ToFEN() << std::endl;
                        //break;
                    }
                }
            }

            if (!pos.DoMove(move))
            {
                break;
            }
        }

        localStats.numGames++;
    }

    {
        std::unique_lock<std::mutex> lock(outStats.mutex);

        outStats.numGames += localStats.numGames;
        outStats.numPositions += localStats.numPositions;
        outStats.numPawnlessPositions += localStats.numPawnlessPositions;

        outStats.evalErrorSum_Score += localStats.evalErrorSum_Score;
        outStats.evalErrorSum_WDL += localStats.evalErrorSum_WDL;

        // accumulate WDL stats
        for (const auto& iter : localStats.materialStats)
        {
            MaterialStats& outMaterialStats = outStats.materialStats[iter.first];

            outMaterialStats.wins += iter.second.wins;
            outMaterialStats.draws += iter.second.draws;
            outMaterialStats.losses += iter.second.losses;
            outMaterialStats.avgEvalScore += iter.second.avgEvalScore / static_cast<double>(iter.second.NumPositions());
        }

        // accumulate piece occupancy stats
        for (uint32_t pieceIndex = 0; pieceIndex < 6; ++pieceIndex)
        {
            for (uint32_t square = 0; square < 64; ++square)
            {
                outStats.pieceOccupancy[pieceIndex][square] += localStats.pieceOccupancy[pieceIndex][square];
            }
        }
    }
}

void AnalyzeGames()
{
    GamesStats stats;
    stats.fortressPosition.open("fortress.epd");
    stats.kingOnFarRankPositions.open("kingOnFarRank.epd");

    const std::string gamesPath = DATA_PATH "selfplayGames/";

    Waitable waitable;
    {
        threadpool::TaskBuilder taskBuilder(waitable);

        std::vector<std::filesystem::path> paths;
        for (const auto& path : std::filesystem::directory_iterator(gamesPath))
        {
            paths.push_back(path.path());
        }

        // sort paths by file size
        std::sort(paths.begin(), paths.end(), [](const std::filesystem::path& a, const std::filesystem::path& b)
        {
            return std::filesystem::file_size(a) > std::filesystem::file_size(b);
        });

        std::cout << "Found " << paths.size() << " paths" << std::endl;

        for (const std::filesystem::path& path : paths)
        {
            taskBuilder.Task("LoadPositions", [path, &stats](const threadpool::TaskContext&)
            {
                AnalyzeGames(path.string().c_str(), stats);
            });
        }
    }

    waitable.Wait();

    // piece-count distribution (no queens)
    {
        uint64_t numPositions[31] = { 0 };
        for (const auto& iter : stats.materialStats)
        {
            const MaterialKey& key = iter.first;
            if (key.numWhiteQueens == 0 && key.numBlackQueens == 0)
            {
                numPositions[std::min(30u, key.CountAll())] += iter.second.NumPositions();
            }
        }

        std::cout << "Piece-count distribution (no queens): " << std::endl;
        for (uint32_t i = 1; i <= 31; ++i)
        {
            std::cout << i << " : " << numPositions[i] << std::endl;
        }

        std::cout << std::endl;
    }

    // piece-count distribution (with queens)
    {
        uint64_t numPositions[31] = { 0 };
        for (const auto& iter : stats.materialStats)
        {
            const MaterialKey& key = iter.first;
            if (key.numWhiteQueens != 0 || key.numBlackQueens != 0)
            {
                numPositions[std::min(30u, key.CountAll())] += iter.second.NumPositions();
            }
        }

        std::cout << "Piece-count distribution (with queens): " << std::endl;
        for (uint32_t i = 1; i <= 30; ++i)
        {
            std::cout << i << " : " << numPositions[i] << std::endl;
        }

        std::cout << std::endl;
    }

    // piece occupancy stats
    {
        std::cout << "Piece occupancy stats: " << std::endl;
        for (uint32_t pieceIndex = 0; pieceIndex < 6; ++pieceIndex)
        {
            std::cout << PieceToString((Piece)(pieceIndex + (uint32_t)Piece::Pawn)) << ": " << std::endl;
            for (uint32_t rank = 0; rank < 8; ++rank)
            {
                for (uint32_t file = 0; file < 8; ++file)
                {
                    std::cout << " " << std::setw(10) << stats.pieceOccupancy[pieceIndex][8 * rank + file];
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
    }

    // static eval error
    {
        std::cout << "Static eval error (WDL):          " << sqrt(stats.evalErrorSum_WDL / stats.numPositions) << std::endl;
        std::cout << "Static eval error (Search Score): " << sqrt(stats.evalErrorSum_Score / stats.numPositions) << std::endl;
        std::cout << std::endl;
    }

    /*
    {
        std::cout << "Unique material configurations: " << materialConfigurations.size() << std::endl;
        for (const auto& iter : materialConfigurations)
        {
            if (iter.second.occurences > 5 && iter.first.numBlackPawns == 0 && iter.first.numWhitePawns == 0)
            {
                const float averageEvalScore = static_cast<float>(iter.second.evalScore / static_cast<double>(iter.second.occurences));
                const float averageGameScore = static_cast<float>(iter.second.gameScore / static_cast<double>(iter.second.occurences));

                std::cout
                    << std::setw(33) << iter.first.ToString() << " "
                    << std::showpos << std::fixed << std::setprecision(2) << ExpectedGameScoreToPawns(averageEvalScore) << " "
                    << std::showpos << std::fixed << std::setprecision(2) << ExpectedGameScoreToPawns(averageGameScore) << std::endl;
                std::cout << std::resetiosflags(std::ios_base::showpos);
            }
        }
        std::cout << std::endl;
    }

    {
        std::cout << "WDL vs. half-move counter: " << std::endl;
        for (uint32_t ply = 0; ply <= 100; ++ply)
        {
            std::cout
                << std::setw(5) << ply << " "
                << std::setw(10) << gameResultVsHalfMoveCounter[0][ply] << " "
                << std::setw(10) << gameResultVsHalfMoveCounter[1][ply] << " "
                << std::setw(10) << gameResultVsHalfMoveCounter[2][ply] << std::endl;
        }
    }

    for (uint32_t pieceIndex = 0; pieceIndex < 6; ++pieceIndex)
    {
        const Piece piece = (Piece)(pieceIndex + (uint32_t)Piece::Pawn);
        std::cout << PieceToString(piece) << " occupancy: " << std::endl;
        for (uint32_t rank = 0; rank < 8; ++rank)
        {
            for (uint32_t file = 0; file < 8; ++file)
            {
                std::cout << std::setw(10) << pieceOccupancy[pieceIndex][8 * rank + file] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
    */

    {
        std::ofstream wdlStatsFile("wdlStats.csv");
        for (const auto& iter : stats.materialStats)
        {
            if (iter.second.NumPositions() < 5)
            {
                continue;
            }

            wdlStatsFile << iter.first.ToString() << ";" << iter.second.wins << ";" << iter.second.draws << ";" << iter.second.losses << std::endl;
        }
    }
}
