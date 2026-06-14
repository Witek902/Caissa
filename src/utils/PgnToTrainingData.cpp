#include "Common.hpp"
#include "ThreadPool.hpp"
#include "TrainerCommon.hpp"
#include "PgnParser.hpp"

#include "../backend/Move.hpp"
#include "../backend/Evaluate.hpp"
#include "../backend/Endgame.hpp"
#include "../backend/Tablebase.hpp"
#include "../backend/Waitable.hpp"

#include <filesystem>
#include <fstream>
#include <random>
#include <atomic>
#include <mutex>

using namespace threadpool;

struct SharedState
{
    std::mutex mutex;
    std::vector<PositionEntry> entries;

    std::atomic<uint64_t> totalGames{ 0 };
    std::atomic<uint64_t> totalPositions{ 0 };
    std::atomic<uint64_t> totalFiles{ 0 };
};

static void ProcessPgnFile(const std::string& path, SharedState& shared)
{
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open())
    {
        std::lock_guard<std::mutex> lock(shared.mutex);
        std::cerr << "ERROR: Cannot open " << path << std::endl;
        return;
    }

    std::vector<PositionEntry> localEntries;
    uint64_t numGames = 0;

    ParsePgn(file, [&](Game& game) -> bool
    {
        ++numGames;

        const Game::Score gameScore = game.GetScore();

        // skip games without a known result
        if (gameScore == Game::Score::Unknown)
            return true;

        const auto& moves     = game.GetMoves();
        const auto& scores    = game.GetMoveScores();
        const bool hasScores  = !scores.empty();

        Position pos = game.GetInitialPosition();

        for (size_t i = 0; i < moves.size(); ++i)
        {
            const Move move = moves[i];

            // Skip if the move has no score annotation
            const bool hasScore = hasScores && (i < scores.size());

            if (hasScore &&
                move.IsQuiet() &&
                pos.GetNumPieces() >= 4 &&
                !pos.IsInCheck())
            {
                // print position and score for debugging
                //std::cout << "FEN: " << pos.ToFEN() << " score=" << scores[i] << " result=" << static_cast<int>(gameScore) << std::endl;

                const ScoreType moveScore = scores[i]; // White's perspective

                PositionEntry entry{};
                entry.wdlScore = static_cast<uint8_t>(gameScore);
                entry.tbScore  = static_cast<uint8_t>(Game::Score::Unknown);
                entry.score    = moveScore;

                Position normalizedPos = pos;
                if (pos.GetSideToMove() == Black)
                {
                    normalizedPos = normalizedPos.SwappedColors();
                    entry.score   = -entry.score;
                    if (gameScore == Game::Score::WhiteWins)
                        entry.wdlScore = static_cast<uint8_t>(Game::Score::BlackWins);
                    else if (gameScore == Game::Score::BlackWins)
                        entry.wdlScore = static_cast<uint8_t>(Game::Score::WhiteWins);
                }

                // Enrich with Syzygy tablebase score when available
                int32_t wdl = 0;
                if (pos.GetNumPieces() <= 7 && ProbeSyzygy_WDL(pos, &wdl))
                {
                    if (wdl > 0)      entry.tbScore = static_cast<uint8_t>(Game::Score::WhiteWins);
                    else if (wdl < 0) entry.tbScore = static_cast<uint8_t>(Game::Score::BlackWins);
                    else              entry.tbScore = static_cast<uint8_t>(Game::Score::Draw);
                }

                ASSERT(normalizedPos.IsValid());
                VERIFY(PackPosition(normalizedPos, entry.pos));
                localEntries.push_back(entry);
            }

            if (!pos.DoMove(move))
                break;
        }

        // Game object is released here; only localEntries keep the extracted data.
        return true;
    });

    const uint64_t numPositions = localEntries.size();

    {
        std::lock_guard<std::mutex> lock(shared.mutex);
        shared.totalGames     += numGames;
        shared.totalPositions += numPositions;
        ++shared.totalFiles;

        std::cout << "[" << shared.totalFiles.load() << "] "
                  << path << ": "
                  << numGames << " games, "
                  << numPositions << " positions" << std::endl;

        // Move local batch into the shared vector
        shared.entries.insert(shared.entries.end(),
            std::make_move_iterator(localEntries.begin()),
            std::make_move_iterator(localEntries.end()));
    }
}

static void CollectPgnPaths(const std::string& input, std::vector<std::string>& out)
{
    namespace fs = std::filesystem;
    const fs::path p(input);

    if (fs::is_regular_file(p))
    {
        if (p.extension() == ".pgn" || p.extension() == ".PGN")
            out.push_back(input);
        else
            out.push_back(input); // accept any extension if given explicitly
    }
    else if (fs::is_directory(p))
    {
        for (const auto& entry : fs::recursive_directory_iterator(p))
        {
            if (entry.is_regular_file())
            {
                const auto ext = entry.path().extension().string();
                if (ext == ".pgn" || ext == ".PGN")
                    out.push_back(entry.path().string());
            }
        }
        std::sort(out.begin(), out.end());
    }
    else
    {
        std::cerr << "WARNING: Path not found: " << input << std::endl;
    }
}

void PgnToTrainingData(const std::vector<std::string>& args)
{
    if (args.size() < 2)
    {
        std::cout << "Usage: pgnToTrainingData <output.dat> <input.pgn|dir> [...]" << std::endl;
        return;
    }

    const std::string outputPath = args[0];

    // Collect all input PGN paths
    std::vector<std::string> inputPaths;
    for (size_t i = 1; i < args.size(); ++i)
        CollectPgnPaths(args[i], inputPaths);

    if (inputPaths.empty())
    {
        std::cerr << "ERROR: No PGN files found." << std::endl;
        return;
    }

    std::cout << "Processing " << inputPaths.size() << " PGN file(s)..." << std::endl;

    SharedState shared;

    // Process files in parallel using the thread pool
    {
        Waitable waitable;
        {
            TaskBuilder taskBuilder(waitable);

            for (const std::string& path : inputPaths)
            {
                taskBuilder.Task("PgnToTrainingData", [path, &shared](const TaskContext&)
                {
                    ProcessPgnFile(path, shared);
                });
            }
        }
        waitable.Wait();
    }

    std::cout << "Totals: "
        << shared.totalFiles.load() << " files, "
        << shared.totalGames.load() << " games, "
        << shared.totalPositions.load() << " positions" << std::endl;

    if (shared.entries.empty())
    {
        std::cerr << "ERROR: No positions extracted." << std::endl;
        return;
    }

    // Shuffle all positions in-place
    std::cout << "Shuffling " << shared.entries.size() << " entries..." << std::endl;
    {
        std::random_device rd;
        std::mt19937 rng(rd());
        std::shuffle(shared.entries.begin(), shared.entries.end(), rng);
    }

    // Write binary output
    std::cout << "Writing " << outputPath << "..." << std::endl;
    {
        std::ofstream out(outputPath, std::ios::binary);
        if (!out.is_open())
        {
            std::cerr << "ERROR: Cannot create output file: " << outputPath << std::endl;
            return;
        }

        out.write(reinterpret_cast<const char*>(shared.entries.data()),
            static_cast<std::streamsize>(shared.entries.size() * sizeof(PositionEntry)));

        if (!out)
        {
            std::cerr << "ERROR: Write failed." << std::endl;
            return;
        }
    }

    const uint64_t outputBytes = shared.entries.size() * sizeof(PositionEntry);
    std::cout
        << "Done. Output: " << outputBytes / (1024 * 1024) << " MB ("
        << shared.entries.size() << " entries)" << std::endl;
}