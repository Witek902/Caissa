#include "UCI.hpp"
#include "../backend/Move.hpp"
#include "../backend/Evaluate.hpp"
#include "../backend/Endgame.hpp"
#include "../backend/Tablebase.hpp"
#include "../backend/Material.hpp"
#include "../backend/TimeManager.hpp"
#include "../backend/PositionUtils.hpp"

#include <math.h>
#include <random>

#define VersionNumber "1.3.3"

#if defined(USE_BMI2) && defined(USE_AVX2) 
#define ArchitectureStr "AVX2/BMI2"
#define AppNamePostfix ""
#elif defined(USE_POPCNT) &&  defined(USE_SSE4) 
#define ArchitectureStr "POPCNT/SSE4"
#define AppNamePostfix " (" ArchitectureStr ")"
#else
#define ArchitectureStr "legacy"
#define AppNamePostfix " (" ArchitectureStr ")"
#endif

static const char* c_EngineName = "Caissa " VersionNumber AppNamePostfix;
static const char* c_Author = "Michal Witanowski";

// TODO set TT size based on current memory usage / total memory size
#ifndef _DEBUG
static const uint32_t c_DefaultTTSizeInMB = 256;
#else
static const uint32_t c_DefaultTTSizeInMB = 16;
#endif
static const uint32_t c_DefaultTTSize = 1024 * 1024 * c_DefaultTTSizeInMB;
static const uint32_t c_DefaultGaviotaTbCacheInMB = 64;
static const uint32_t c_MaxNumThreads = 1024;


using UniqueLock = std::unique_lock<std::mutex>;

UniversalChessInterface::UniversalChessInterface()
{
    mSearchThread = std::thread(&UniversalChessInterface::SearchThreadEntryFunc, this);

    mGame.Reset(Position(Position::InitPositionFEN));
    mTranspositionTable.Resize(c_DefaultTTSize);

    std::cout << c_EngineName << " by " << c_Author << std::endl;

    TryLoadingDefaultEvalFile();
    TryLoadingDefaultEndgameEvalFile();

    // Note: this won't allocate memory immediately, but will be deferred once tablebase is loaded
    SetGaviotaCacheSize(1024 * 1024 * c_DefaultGaviotaTbCacheInMB);


}

UniversalChessInterface::~UniversalChessInterface()
{
    StopSearchThread();
}

void UniversalChessInterface::Loop(int argc, const char* argv[])
{
    for (int i = 1; i < argc; ++i)
    {
        if (argv[i])
        {
            std::cout << "CommandLine: " << argv[i] << std::endl;
            if (!ExecuteCommand(argv[i]))
            {
                return;
            }
        }
    }

    std::string str;

    while (std::getline(std::cin, str))
    {
        if (!ExecuteCommand(str))
        {
            break;
        }
    }

    UnloadTablebase();
}

static void ParseCommandString(const std::string& str, std::vector<std::string>& outArgList)
{
    std::string tmp;

    // tokenize command string, keep string in quotes as a single string
    for (size_t i = 0; i < str.length(); i++)
    {
        char c = str[i];
        if (c == ' ')
        {
            outArgList.push_back(std::move(tmp));
        }
        else if (c == '\"')
        {
            i++;
            while (str[i] != '\"')
            {
                tmp += str[i];
                i++;
            }
        }
        else
        {
            tmp += c;
        }
    }

    if (!tmp.empty())
    {
        outArgList.push_back(std::move(tmp));
    }
}

bool UniversalChessInterface::ExecuteCommand(const std::string& commandString)
{
    std::vector<std::string> args;
    ParseCommandString(commandString, args);

    if (args.empty())
    {
        std::cout << "Invalid command" << std::endl;
        return true;
    }

    const std::string& command = args[0];

    if (command == "uci")
    {
        std::cout << "id name " << c_EngineName << "\n";
        std::cout << "id author " << c_Author << "\n";
        std::cout << "option name Hash type spin default " << c_DefaultTTSizeInMB  << " min 1 max 1048576\n";
        std::cout << "option name MultiPV type spin default 1 min 1 max 255\n";
        std::cout << "option name MoveOverhead type spin default " << mOptions.moveOverhead << " min 0 max 10000\n";
        std::cout << "option name Threads type spin default 1 min 1 max " << c_MaxNumThreads << "\n";
        std::cout << "option name Ponder type check default false\n";
        std::cout << "option name EvalFile type string default " << c_DefaultEvalFile << "\n";
        std::cout << "option name EndgameEvalFile type string default " << c_DefaultEndgameEvalFile << "\n";
        std::cout << "option name SyzygyPath type string default <empty>\n";
        std::cout << "option name GaviotaTbPath type string default <empty>\n";
        std::cout << "option name GaviotaTbCache type spin default " << c_DefaultGaviotaTbCacheInMB << " min 1 max 1048576\n";
        std::cout << "option name UCI_AnalyseMode type check default false\n";
        std::cout << "option name UCI_Chess960 type check default false\n";
        std::cout << "option name UseSAN type check default false\n";
        std::cout << "option name ColorConsoleOutput type check default false\n";
        std::cout << "option name TunedParam type spin default 0 min -1000000 max 1000000\n";
        std::cout << "uciok" << std::endl;
    }
    else if (command == "isready")
    {
        std::cout << "readyok" << std::endl;
    }
    else if (command == "ucinewgame")
    {
        mTranspositionTable.Clear();
        mSearch.Clear();
    }
    else if (command == "setoption")
    {
        if (args.size() >= 4 && args[1] == "name" && args[3] == "value")
        {
            size_t offset = commandString.find("value");
            offset += 5; // skip "value" string
            while (offset < commandString.size() && isspace(commandString[offset]))
            {
                offset++;
            }

            Command_SetOption(args[2], commandString.substr(offset));
        }
        else
        {
            std::cout << "Invalid command" << std::endl;
        }
    }
    else if (command == "position")
    {
        Command_Stop();
        Command_Position(args);
    }
    else if (command == "go")
    {
        Command_Stop();
        Command_Go(args);
    }
    else if (command == "ponderhit")
    {
        Command_PonderHit();
    }
    else if (command == "stop")
    {
        Command_Stop();
    }
    else if (command == "quit")
    {
        Command_Stop();
        return false;
    }
    else if (command == "perft")
    {
        Command_Perft(args);
    }
    else if (command == "print")
    {
        std::cout << "Init:    " << mGame.GetInitialPosition().ToFEN() << std::endl; 
        std::cout << "Moves:   " << mGame.ToPGNMoveList() << std::endl;
        std::cout << "Current: " << mGame.GetPosition().ToFEN() << std::endl << mGame.ToPGNMoveList() << std::endl;
        std::cout << mGame.GetPosition().Print() << std::endl;
    }
    else if (command == "eval")
    {
        std::cout << Evaluate(mGame.GetPosition()) << std::endl;
    }
    else if (command == "scoremoves")
    {
        Command_ScoreMoves();
    }
    else if (command == "ttinfo")
    {
        const size_t numEntriesUsed = mTranspositionTable.GetNumUsedEntries();
        const float percentage = 100.0f * (float)numEntriesUsed / (float)mTranspositionTable.GetSize();
        std::cout << "TT entries in use: " << numEntriesUsed << " (" << percentage << "%)" << std::endl;
        std::cout << "TT collisions: " << mTranspositionTable.GetNumCollisions() << std::endl;
    }
    else if (command == "ttprobe")
    {
        Command_TranspositionTableProbe();
    }
    else if (command == "tbprobe")
    {
        Command_TablebaseProbe();
    }
#ifndef CONFIGURATION_FINAL
    else if (command == "moveordererstats")
    {
        mSearch.GetMoveOrderer().DebugPrint();
    }
#endif // CONFIGURATION_FINAL
#ifdef COLLECT_ENDGAME_STATISTICS
    else if (command == "endgamestats")
    {
        PrintEndgameStatistics();
    }
#endif // COLLECT_ENDGAME_STATISTICS
    else
    {
        std::cout << "Invalid command" << std::endl;
    }

    return true;
}

bool UniversalChessInterface::Command_Position(const std::vector<std::string>& args)
{
    size_t extraMovesStart = 0;

    Position pos;

    if (args.size() >= 2 && args[1] == "startpos")
    {
        pos.FromFEN(Position::InitPositionFEN);

        if (args.size() >= 4 && args[2] == "moves")
        {
            extraMovesStart = 2;
        }
    }
    else if (args.size() >= 2 && args[1] == "random")
    {
        MaterialKey matKey = { 8, 2, 2, 2, 1, 8, 2, 2, 2, 1 };
        std::random_device rd;
        std::mt19937 mt(rd());
        GenerateRandomPosition(mt, matKey, pos);

        if (args.size() >= 4 && args[2] == "moves")
        {
            extraMovesStart = 2;
        }
    }
    else if (args.size() >= 2 && args[1] == "randomstartpos")
    {
        std::random_device rd;
        std::mt19937 mt(rd());
        GenerateTranscendentalChessPosition(mt, pos);

        if (args.size() >= 4 && args[2] == "moves")
        {
            extraMovesStart = 2;
        }
    }
    else if (args.size() > 2 && args[1] == "fen")
    {
        size_t numFenElements = 0;

        for (size_t i = 2; i < args.size(); ++i)
        {
            if (args[i] == "moves")
            {
                extraMovesStart = i;
                break;
            }
            numFenElements++;
        }

        // [board] [sideToMove] [castling] [enPassant] [halfMoves] [fullmove]
        std::string fenString;
        for (size_t i = 0; i < numFenElements; ++i)
        {
            fenString += args[2 + i];
            if (i < 5) fenString += ' ';
        }

        if (numFenElements < 3)
        {
            std::cout << "Invalid FEN" << std::endl;
            return false;
        }

        if (numFenElements < 4)
        {
            fenString += "- ";
        }

        if (numFenElements < 5)
        {
            fenString += "0 ";
        }

        if (numFenElements < 6)
        {
            fenString += "1";
        }

        
        if (!pos.FromFEN(fenString))
        {
            return false;
        }
    }
    else
    {
        return false;
    }

    Command_Stop();

    mGame.Reset(pos);

    if (extraMovesStart > 0)
    {
        for (size_t i = extraMovesStart + 1; i < args.size(); ++i)
        {
            const Move move = mGame.GetPosition().MoveFromString(args[i]);

            if (!move.IsValid() || !mGame.GetPosition().IsMoveValid(move))
            {
                std::cout << "Invalid move" << std::endl;
                return false;
            }

            if (!mGame.DoMove(move))
            {
                std::cout << "Invalid move" << std::endl;
                return false;
            }
        }
    }

    return true;
}

bool UniversalChessInterface::Command_Go(const std::vector<std::string>& args)
{
    Command_Stop();

    const TimePoint startTimePoint = TimePoint::GetCurrent();

    bool isInfinite = false;
    bool isPonder = false;
    bool verboseStats = false;
    bool waitForSearch = false;
    uint32_t maxDepth = UINT32_MAX;
    uint64_t maxNodes = UINT64_MAX;
    int32_t moveTime = INT32_MAX;
    int32_t whiteRemainingTime = INT32_MAX;
    int32_t blacksRemainingTime = INT32_MAX;
    int32_t whiteTimeIncrement = 0;
    int32_t blacksTimeIncrement = 0;
    uint32_t movesToGo = UINT32_MAX;
    uint32_t mateSearchDepth = 0;

    std::vector<Move> excludedMoves;

    for (size_t i = 1; i < args.size(); ++i)
    {
        if (args[i] == "depth" && i + 1 < args.size())
        {
            maxDepth = atoi(args[i + 1].c_str());
        }
        if (args[i] == "mate" && i + 1 < args.size())
        {
            mateSearchDepth = atoi(args[i + 1].c_str());
        }
        else if (args[i] == "infinite")
        {
            isInfinite = true;
        }
        else if (args[i] == "ponder")
        {
            isPonder = true;
        }
        else if (args[i] == "wait")
        {
            waitForSearch = true;
        }
        else if (args[i] == "verbosestats")
        {
            verboseStats = true;
        }
        else if (args[i] == "nodes" && i + 1 < args.size())
        {
            maxNodes = std::stoull(args[i + 1].c_str());
        }
        else if (args[i] == "movetime" && i + 1 < args.size())
        {
            moveTime = atoi(args[i + 1].c_str());
        }
        else if (args[i] == "wtime" && i + 1 < args.size())
        {
            whiteRemainingTime = atoi(args[i + 1].c_str());
        }
        else if (args[i] == "btime" && i + 1 < args.size())
        {
            blacksRemainingTime = atoi(args[i + 1].c_str());
        }
        else if (args[i] == "winc" && i + 1 < args.size())
        {
            whiteTimeIncrement = atoi(args[i + 1].c_str());
        }
        else if (args[i] == "binc" && i + 1 < args.size())
        {
            blacksTimeIncrement = atoi(args[i + 1].c_str());
        }
        else if (args[i] == "searchmoves" && i + 1 < args.size())
        {
            // TODO use 'excludedMoves' to implement this feature
            // restrict search to this moves only
            //for (size_t j = i + 1; j < args.size(); ++j)
            //{
            //    const Move move = mGame.GetPosition().MoveFromString(args[j]);
            //    if (move.IsValid())
            //    {
            //        rootMoves.push_back(move);
            //    }
            //    else
            //    {
            //        std::cout << "Invalid move: " << args[j] << std::endl;
            //        return false;
            //    }
            //}
        }
        else if (args[i] == "excludemoves" && i + 1 < args.size())
        {
            // exclude moves from search
            for (size_t j = i + 1; j < args.size(); ++j)
            {
                const Move move = mGame.GetPosition().MoveFromString(args[j]);
                if (move.IsValid())
                {
                    excludedMoves.push_back(move);
                }
                else
                {
                    std::cout << "Invalid move: " << args[j] << std::endl;
                    return false;
                }
            }
        }
        else if (args[i] == "movestogo" && i + 1 < args.size())
        {
            movesToGo = atoi(args[i + 1].c_str());
        }
    }

    mSearchCtx = std::make_unique<SearchTaskContext>(mTranspositionTable);

    mSearchCtx->searchParam.limits.startTimePoint = startTimePoint;

    // calculate time for move based on total remaining time and other heuristics
    {
        TimeManagerInitData data;
        data.moveTime = moveTime;
        data.remainingTime = mGame.GetSideToMove() == Color::White ? whiteRemainingTime : blacksRemainingTime;
        data.timeIncrement = mGame.GetSideToMove() == Color::White ? whiteTimeIncrement : blacksTimeIncrement;
        data.theirRemainingTime = mGame.GetSideToMove() == Color::White ? blacksRemainingTime : whiteRemainingTime;
        data.theirTimeIncrement = mGame.GetSideToMove() == Color::White ? blacksTimeIncrement : whiteTimeIncrement;
        data.movesToGo = movesToGo;
        data.moveOverhead = mOptions.moveOverhead;

        TimeManager::Init(mGame, data, mSearchCtx->searchParam.limits);
    }

    if (mateSearchDepth > 0)
    {
        // mate depth is in moves, not plies
        maxDepth = 2 * mateSearchDepth;
    }

    // TODO
    // Instead of pondering on suggested move, maybe undo last move and ponder on oponent's position instead.
    // This way we can consider all possible oponent's replies, not just focus on predicted one... UCI is lame...
    mSearchCtx->searchParam.isPonder = isPonder;

    mSearchCtx->searchParam.limits.maxDepth = (uint8_t)std::min<uint32_t>(maxDepth, UINT8_MAX);
    mSearchCtx->searchParam.limits.maxNodes = maxNodes;
    mSearchCtx->searchParam.limits.mateSearch = mateSearchDepth > 0;
    mSearchCtx->searchParam.limits.analysisMode = !isPonder && (isInfinite || mOptions.analysisMode); // run full analysis when pondering
    mSearchCtx->searchParam.numPvLines = mOptions.multiPV;
    mSearchCtx->searchParam.numThreads = mOptions.threads;
    mSearchCtx->searchParam.excludedMoves = std::move(excludedMoves);
    mSearchCtx->searchParam.verboseStats = verboseStats;
    mSearchCtx->searchParam.moveNotation = mOptions.useStandardAlgebraicNotation ? MoveNotation::SAN : MoveNotation::LAN;
    mSearchCtx->searchParam.colorConsoleOutput = mOptions.colorConsoleOutput;

    RunSearchTask();

    if (waitForSearch)
    {
        mSearchCtx->waitable.Wait();
    }

    return true;
}

void UniversalChessInterface::StopSearchThread()
{
    {
        std::unique_lock<std::mutex> lock(mNewSearchMutex);
        mStopSearchThread = true;
        mNewSearchConditionVariable.notify_one();
    }
    mSearchThread.join();
}

void UniversalChessInterface::SearchThreadEntryFunc()
{
    for (;;)
    {
        {
            // wait for a new search or a request to stop the thread
            std::unique_lock<std::mutex> lock(mNewSearchMutex);
            while (!mNewSearchContext && !mStopSearchThread)
            {
                mNewSearchConditionVariable.wait(lock);
            }
            mNewSearchContext = nullptr;
        }

        if (mStopSearchThread)
        {
            return;
        }

        DoSearch();
    }
}

void UniversalChessInterface::DoSearch()
{
    mTranspositionTable.NextGeneration();

    mSearch.DoSearch(mGame, mSearchCtx->searchParam, mSearchCtx->searchResult);

    // only report best move in non-pondering mode or if "stop" was called during ponder search
    if (!mSearchCtx->searchParam.isPonder || !mSearchCtx->ponderHit)
    {
        Move bestMove = Move::Invalid();
        if (!mSearchCtx->searchResult.empty())
        {
            const auto& bestLine = mSearchCtx->searchResult[0].moves;
            const MoveNotation notation = mOptions.useStandardAlgebraicNotation ? MoveNotation::SAN : MoveNotation::LAN;

            if (!bestLine.empty())
            {
                bestMove = bestLine[0];

                std::cout << "bestmove " << mGame.GetPosition().MoveToString(bestMove, notation);

                if (bestLine.size() > 1)
                {
                    Position posAfterBestMove = mGame.GetPosition();
                    posAfterBestMove.DoMove(bestMove);
                    std::cout << " ponder " << posAfterBestMove.MoveToString(bestLine[1], notation);
                }
            }
        }

        if (mSearchCtx->searchParam.verboseStats)
        {
            const float elapsedTime = (TimePoint::GetCurrent() - mSearchCtx->searchParam.limits.startTimePoint).ToSeconds();
            std::cout << std::endl << "info string total time " << elapsedTime << " seconds";
        }

        if (!bestMove.IsValid()) // null move
        {
            std::cout << "bestmove 0000";
        }

        std::cout << std::endl << std::flush;
    }
    
    mSearchCtx->waitable.OnFinished();
}

void UniversalChessInterface::RunSearchTask()
{
    {
        std::unique_lock<std::mutex> lock(mNewSearchMutex);
        mNewSearchContext = mSearchCtx.get();
        mNewSearchConditionVariable.notify_one();
    }
}

bool UniversalChessInterface::Command_Stop()
{
    if (mSearchCtx)
    {
        // wait for previous search to complete
        mSearch.StopSearch();
        mSearchCtx->waitable.Wait();

        mSearchCtx.reset();
    }

    return true;
}

bool UniversalChessInterface::Command_PonderHit()
{
    if (mSearchCtx)
    {
        if (!mSearchCtx->searchParam.isPonder)
        {
            std::cout << "Engine is not pondering right now" << std::endl;
            return false;
        }

        mSearchCtx->ponderHit = true;

        // wait for previous search to complete
        mSearch.StopSearch();
        mSearchCtx->waitable.Wait();
        mSearchCtx->waitable.Reset();

        // start searching again, this time not in pondering mode
        mSearchCtx->searchParam.isPonder = false;

        RunSearchTask();
    }

    return true;
}

bool UniversalChessInterface::Command_Perft(const std::vector<std::string>& args)
{
    if (args.size() != 2)
    {
        std::cout << "Invalid perft arguments" << std::endl;
        return false;
    }

    uint32_t maxDepth = atoi(args[1].c_str());

    mGame.GetPosition().Perft(maxDepth, true);

    return true;
}

static std::string ToLower(const std::string& str)
{
    std::string result = str;
    for (char& c : result)
    {
        if (c <= 'Z' && c >= 'A')
        {
            c = (c - ('Z' - 'z'));
        }
    }
    return result;
}

static bool ParseBool(const std::string& str, bool& outValue)
{
    if (str == "true" || str == "1")
    {
        outValue = true;
        return true;
    }
    else if (str == "false" || str == "0")
    {
        outValue = false;
        return true;
    }
    return false;
}

bool UniversalChessInterface::Command_SetOption(const std::string& name, const std::string& value)
{
    std::string lowerCaseName = ToLower(name);
    std::string lowerCaseValue = ToLower(value);

    if (lowerCaseName == "multipv")
    {
        mOptions.multiPV = atoi(value.c_str());
        mOptions.multiPV = std::max(1u, mOptions.multiPV);
    }
    else if (lowerCaseName == "threads")
    {
        mOptions.threads = atoi(value.c_str());
        mOptions.threads = std::max(1u, std::min(c_MaxNumThreads, mOptions.threads));
    }
    else if (lowerCaseName == "moveoverhead")
    {
        mOptions.moveOverhead = atoi(value.c_str());
        mOptions.moveOverhead = std::clamp(mOptions.moveOverhead, 0, 10000);
    }
    else if (lowerCaseName == "hash" || lowerCaseName == "hashsize")
    {
        size_t hashSize = 1024 * 1024 * static_cast<size_t>(std::max(1, atoi(value.c_str())));
        mTranspositionTable.Resize(hashSize);
    }
    else if (lowerCaseName == "usesan" || lowerCaseName == "usestandardalgebraicnotation")
    {
        if (!ParseBool(lowerCaseValue, mOptions.useStandardAlgebraicNotation))
        {
            std::cout << "Invalid value" << std::endl;
            return false;
        }
    }
    else if (lowerCaseName == "uci_analysemode" || lowerCaseName == "uci_analyzemode" || lowerCaseName == "analysis" || lowerCaseName == "analysismode")
    {
        if (!ParseBool(lowerCaseValue, mOptions.analysisMode))
        {
            std::cout << "Invalid value" << std::endl;
            return false;
        }
    }
    else if (lowerCaseName == "uci_chess960")
    {
        if (!ParseBool(lowerCaseValue, Position::s_enableChess960))
        {
            std::cout << "Invalid value" << std::endl;
            return false;
        }
    }
#ifdef USE_TABLE_BASES
    else if (lowerCaseName == "syzygypath")
    {
        LoadSyzygyTablebase(value.c_str());
    }
    else if (lowerCaseName == "gaviotatbpath")
    {
        LoadGaviotaTablebase(value.c_str());
    }
    else if (lowerCaseName == "gaviotatbcache")
    {
        const size_t cacheSize = 1024 * 1024 * static_cast<size_t>(std::max(1, atoi(value.c_str())));
        SetGaviotaCacheSize(cacheSize);
    }
#endif // USE_TABLE_BASES
    else if (lowerCaseName == "evalfile")
    {
        LoadMainNeuralNetwork(value.c_str());
    }
    else if (lowerCaseName == "endgameevalfile")
    {
        LoadEndgameNeuralNetwork(value.c_str());
    }
    else if (lowerCaseName == "ponder")
    {
        // nothing special here
    }
    else if (lowerCaseName == "colorconsoleoutput")
    {
        if (!ParseBool(lowerCaseValue, mOptions.colorConsoleOutput))
        {
            std::cout << "Invalid value" << std::endl;
            return false;
        }
    }
    else if (lowerCaseName == "tunedparam")
    {
        g_TunedParameter = atoi(value.c_str());
    }
    else
    {
        std::cout << "Invalid option: " << name << std::endl;
        return false;
    }

    return true;
}

bool UniversalChessInterface::Command_TranspositionTableProbe()
{
    TTEntry ttEntry;

    std::cout << "Hash:       " << mGame.GetPosition().GetHash() << std::endl;

    if (mTranspositionTable.Read(mGame.GetPosition(), ttEntry))
    {
        const char* boundsStr =
            ttEntry.bounds == TTEntry::Bounds::Exact ? "exact" :
            ttEntry.bounds == TTEntry::Bounds::Upper ? "upper" :
            ttEntry.bounds == TTEntry::Bounds::Lower ? "lower" :
            "invalid";

        std::cout << "Score:      " << ttEntry.score << std::endl;
        std::cout << "StaticEval: " << ttEntry.staticEval << std::endl;
        std::cout << "Depth:      " << (uint32_t)ttEntry.depth << std::endl;
        std::cout << "Bounds:     " << boundsStr << std::endl;
        std::cout << "Generation: " << (uint32_t)ttEntry.generation << std::endl;
        std::cout << "Moves:      ";

        uint32_t numMoves = 0;
        for (uint32_t i = 0; i < TTEntry::NumMoves; ++i)
        {
            if (!ttEntry.moves[i].IsValid()) break;
            std::cout << ttEntry.moves[i].ToString() << " ";
            numMoves++;
        }
        if (numMoves == 0) std::cout << "<none>";
        std::cout << std::endl;
    }
    else
    {
        std::cout << "(no entry found)" << std::endl;
    }

    return true;
}

bool UniversalChessInterface::Command_TablebaseProbe()
{
    {
        Move tbMove;
        int32_t wdl = 0;
        uint32_t dtz = 0;
        if (ProbeSyzygy_Root(mGame.GetPosition(), tbMove, &dtz, &wdl))
        {
            std::cout << "Syzygy tablebase entry found!" << std::endl;
            std::cout << "Score:            " << wdl << std::endl;
            std::cout << "Distance to zero: " << dtz << std::endl;
            std::cout << "Move:             " << tbMove.ToString() << std::endl;
        }
        else if (ProbeSyzygy_WDL(mGame.GetPosition(), &wdl))
        {
            std::cout << "Syzygy tablebase entry found!" << std::endl;
            std::cout << "Score: " << wdl << std::endl;
        }
    }

    {
        int32_t wdl = 0;
        uint32_t dtm = 0;
        if (ProbeGaviota(mGame.GetPosition(), &dtm, &wdl))
        {
            std::cout << "Gaviota tablebase entry found!" << std::endl;
            std::cout << "Score:            " << wdl << std::endl;
            std::cout << "Distance to mate: " << dtm << std::endl;
        }
    }

    return true;
}

bool UniversalChessInterface::Command_ScoreMoves()
{
    MoveList moves;
    mGame.GetPosition().GenerateMoveList(moves);

    NodeInfo nodeInfo;
    nodeInfo.position = mGame.GetPosition();

    TTEntry ttEntry;
    if (mTranspositionTable.Read(mGame.GetPosition(), ttEntry))
    {
        moves.AssignTTScores(ttEntry);
    }

    mSearch.GetMoveOrderer().ScoreMoves(nodeInfo, mGame, moves);

    moves.Sort();
    moves.Print(mGame.GetPosition());

    return true;
}
