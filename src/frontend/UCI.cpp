#include "UCI.hpp"
#include "../backend/Move.hpp"
#include "../backend/MoveGen.hpp"
#include "../backend/Evaluate.hpp"
#include "../backend/NeuralNetworkEvaluator.hpp"
#include "../backend/Endgame.hpp"
#include "../backend/Tablebase.hpp"
#include "../backend/Material.hpp"
#include "../backend/TimeManager.hpp"
#include "../backend/PositionUtils.hpp"
#include "../backend/Tuning.hpp"

#include <math.h>

#ifndef CAISSA_VERSION
#define CAISSA_VERSION "1.15.5"
#endif // CAISSA_VERSION

#if defined(USE_AVX512)
#define ArchitectureStr "AVX-512"
#elif defined(USE_BMI2) && defined(USE_AVX2)
#define ArchitectureStr "BMI2"
#elif defined(USE_AVX2)
#define ArchitectureStr "AVX2"
#elif defined(USE_POPCNT) &&  defined(USE_SSE4) 
#define ArchitectureStr "POPCNT"
#elif defined(USE_ARM_NEON) 
#define ArchitectureStr "ARM NEON"
#else
#define ArchitectureStr "legacy"
#endif

#if defined(CONFIGURATION_FINAL)
#define ConfigurationStr ""
#elif defined(CONFIGURATION_RELEASE)
#define ConfigurationStr " RELEASE"
#elif defined(CONFIGURATION_DEBUG)
#define ConfigurationStr " DEBUG"
#else
#error "Unknown configuration"
#endif

static const char* c_EngineName = "Caissa " CAISSA_VERSION " " ArchitectureStr ConfigurationStr;
static const char* c_Author = "Michal Witanowski";

// TODO set TT size based on current memory usage / total memory size
#ifndef _DEBUG
static const uint32_t c_DefaultTTSizeInMB = 64;
#else
static const uint32_t c_DefaultTTSizeInMB = 16;
#endif
static const uint32_t c_DefaultTTSize = 1024 * 1024 * c_DefaultTTSizeInMB;
#ifdef USE_GAVIOTA_TABLEBASES
static const uint32_t c_DefaultGaviotaTbCacheInMB = 64;
#endif // USE_GAVIOTA_TABLEBASES
static const uint32_t c_MaxNumThreads = 1024;


using UniqueLock = std::unique_lock<std::mutex>;

UniversalChessInterface::UniversalChessInterface()
{
    mSearchThread = std::thread(&UniversalChessInterface::SearchThreadEntryFunc, this);

    mGame.Reset(Position(Position::InitPositionFEN));
    mTranspositionTable.Resize(c_DefaultTTSize);

    std::cout << c_EngineName << " by " << c_Author << std::endl;

    TryLoadingDefaultEvalFile();

#ifdef USE_GAVIOTA_TABLEBASES
    // Note: this won't allocate memory immediately, but will be deferred once tablebase is loaded
    SetGaviotaCacheSize(1024 * 1024 * c_DefaultGaviotaTbCacheInMB);
#endif // USE_GAVIOTA_TABLEBASES
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
            std::string cmd = argv[i];

            std::cout << "CommandLine: " << cmd << std::endl;

            if (!ExecuteCommand(cmd))
            {
                return;
            }

            // "bench" command is used to run benchmark and exit immediately to comply with OpenBench
            if (cmd == "bench")
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
    std::vector<std::string>& args = mCommandArgs;
    args.clear();
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
        std::cout << "option name MultiPV type spin default 1 min 1 max " << MaxAllowedMoves << "\n";
        std::cout << "option name MoveOverhead type spin default " << mOptions.moveOverhead << " min 0 max 10000\n";
        std::cout << "option name Threads type spin default 1 min 1 max " << c_MaxNumThreads << "\n";
        std::cout << "option name Ponder type check default false\n";
        std::cout << "option name EvalFile type string default " << c_DefaultEvalFile << "\n";
        std::cout << "option name EvalRandomization type spin default 0 min 0 max 100\n";
        std::cout << "option name StaticContempt type spin default 0 min -1000 max 1000\n";
        std::cout << "option name DynamicContempt type spin default 0 min -1000 max 1000\n";
#ifdef USE_SYZYGY_TABLEBASES
        std::cout << "option name SyzygyPath type string default <empty>\n";
        std::cout << "option name SyzygyProbeLimit type spin default 6 min 4 max 7\n";
#endif // USE_SYZYGY_TABLEBASES
#ifdef USE_GAVIOTA_TABLEBASES
        std::cout << "option name GaviotaTbPath type string default <empty>\n";
        std::cout << "option name GaviotaTbCache type spin default " << c_DefaultGaviotaTbCacheInMB << " min 1 max 1048576\n";
#endif // USE_GAVIOTA_TABLEBASES
        std::cout << "option name UCI_AnalyseMode type check default false\n";
        std::cout << "option name UCI_Chess960 type check default false\n";
        std::cout << "option name UCI_ShowWDL type check default false\n";
        std::cout << "option name UseSAN type check default false\n";
        std::cout << "option name ColorConsoleOutput type check default false\n";
#ifdef ENABLE_TUNING
        for (const TunableParameter& param : g_TunableParameters)
        {
            std::cout << "option name " << param.m_name << " type spin default " << param.m_value << "\n";
        }
#endif // ENABLE_TUNING
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

            Command_Stop();
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
    else if (command == "quit" || command == "exit")
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
    else if (command == "threats")
    {
        Threats threats;
        mGame.GetPosition().ComputeThreats(threats);
        std::cout << "Attacked by pawns" << std::endl << threats.attackedByPawns.Print() << std::endl;
        std::cout << "Attacked by minors" << std::endl << threats.attackedByMinors.Print() << std::endl;
        std::cout << "Attacked by rooks" << std::endl << threats.attackedByRooks.Print() << std::endl;
        std::cout << "All threats:" << std::endl << threats.allThreats.Print() << std::endl;
    }
    else if (command == "ttinfo")
    {
        mTranspositionTable.PrintInfo();
    }
    else if (command == "ttprobe")
    {
        Command_TranspositionTableProbe();
    }
    else if (command == "tbprobe")
    {
        Command_TablebaseProbe();
    }
    else if (command == "cacheprobe")
    {
        Command_NodeCacheProbe();
    }
    else if (command == "bench" || command == "benchmark")
    {
        Command_Benchmark();
    }
#ifdef ENABLE_TUNING
    else if (command == "printparams")
    {
        PrintParametersForTuning();
    }
#endif // ENABLE_TUNING
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
    else if (command == "help")
    {
        // print all available commands
        std::cout << "Available UCI commands:" << std::endl;
        std::cout << " * uci - print available UCI options list" << std::endl;
        std::cout << " * isready - print 'readyok' when the engine is ready" << std::endl;
        std::cout << " * ucinewgame - prepare for new game by clearing transposition table and search data" << std::endl;
        std::cout << " * setoption name <name> value <value> - set an UCI option" << std::endl;
        std::cout << " * position [startpos | fen <fenstring> ] moves <move1> ... <movei> - set position" << std::endl;
        std::cout << " * go [depth <depth> | movetime <time> | wtime <time> | btime <time> | winc <time> | binc <time> | movestogo <moves> | infinite] - start search" << std::endl;
        std::cout << " * ponderhit - start searching in pondering mode" << std::endl;
        std::cout << " * stop - stop searching" << std::endl;
        std::cout << " * quit|exit - quit the engine" << std::endl;
        std::cout << " * perft <depth> - run perft test on current position" << std::endl;
        std::cout << " * print - print current position" << std::endl;
        std::cout << " * eval - evaluate current position" << std::endl;
        std::cout << " * scoremoves - print all legal moves with their move orderer scores" << std::endl;
        std::cout << " * ttinfo - print transposition table info" << std::endl;
        std::cout << " * ttprobe - probe transposition table with current position" << std::endl;
        std::cout << " * tbprobe - probe tablebases with current position" << std::endl;
        std::cout << " * cacheprobe - probe node cache" << std::endl;
        std::cout << " * bench|benchmark - run benchmark" << std::endl;
    }
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
#ifndef CONFIGURATION_FINAL
    else if (args.size() >= 2 && args[1] == "random")
    {
        MaterialKey matKey = { 8, 2, 2, 2, 1, 8, 2, 2, 2, 1 };
        std::random_device rd;
        std::mt19937 mt(rd());
        GenerateRandomPosition(mt, RandomPosDesc{ matKey }, pos);

        if (args.size() >= 4 && args[2] == "moves")
        {
            extraMovesStart = 2;
        }
    }
#endif // CONFIGURATION_FINAL
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
    uint64_t maxNodesSoft = UINT64_MAX;
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
        else if (args[i] == "nodes_soft" && i + 1 < args.size())
        {
            maxNodesSoft = std::stoull(args[i + 1].c_str());
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
            mGame.GetPosition().GetNumLegalMoves(&excludedMoves);

            for (size_t j = i + 1; j < args.size(); ++j)
            {
                const Move move = mGame.GetPosition().MoveFromString(args[j]);
                if (move.IsValid())
                {
                    excludedMoves.erase(
                        remove(excludedMoves.begin(), excludedMoves.end(), move),
                        excludedMoves.end());
                }
                else
                {
                    std::cout << "Invalid move: " << args[j] << std::endl;
                    return false;
                }
            }
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
    // Instead of pondering on suggested move, maybe undo last move and ponder on opponent's position instead.
    // This way we can consider all possible opponent's replies, not just focus on predicted one... UCI is lame...
    mSearchCtx->searchParam.isPonder = isPonder;
    mSearchCtx->startedAsPondering = isPonder;

    mSearchCtx->searchParam.limits.maxDepth = (uint8_t)std::min<uint32_t>(maxDepth, UINT8_MAX);
    mSearchCtx->searchParam.limits.maxNodes = maxNodes;
    mSearchCtx->searchParam.limits.maxNodesSoft = maxNodesSoft;
    mSearchCtx->searchParam.limits.mateSearch = mateSearchDepth > 0;
    mSearchCtx->searchParam.limits.analysisMode = !isPonder && (isInfinite || mOptions.analysisMode); // run full analysis when pondering
    mSearchCtx->searchParam.numPvLines = mOptions.multiPV;
    mSearchCtx->searchParam.numThreads = mOptions.threads;
    mSearchCtx->searchParam.evalRandomization = mOptions.evalRandomization;
    mSearchCtx->searchParam.staticContempt = mOptions.staticContempt;
    mSearchCtx->searchParam.dynamicContempt = mOptions.dynamicContempt;
    mSearchCtx->searchParam.excludedMoves = std::move(excludedMoves);
    mSearchCtx->searchParam.verboseStats = verboseStats;
    mSearchCtx->searchParam.moveNotation = mOptions.useStandardAlgebraicNotation ? MoveNotation::SAN : MoveNotation::LAN;
    mSearchCtx->searchParam.colorConsoleOutput = mOptions.colorConsoleOutput;
    mSearchCtx->searchParam.showWDL = mOptions.showWDL;

    {
        std::unique_lock<std::mutex> lock(mSearchThreadMutex);
        mNewSearchContext = mSearchCtx.get();
        mNewSearchConditionVariable.notify_one();
    }

    // make sure search thread actually started running before exiting from this function
    while (!mSearchCtx->searchStarted.load(std::memory_order_acquire))
        ;

    if (waitForSearch)
    {
        mSearchCtx->waitable.Wait();
    }

    return true;
}

void UniversalChessInterface::StopSearchThread()
{
    {
        std::unique_lock<std::mutex> lock(mSearchThreadMutex);
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
            std::unique_lock<std::mutex> lock(mSearchThreadMutex);
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

#ifdef NN_ACCUMULATOR_STATS
static void PrintNNEvaluatorStats()
{
    uint64_t numUpdates = 0, numRefreshes = 0;
    NNEvaluator::GetStats(numUpdates, numRefreshes);
    std::cout << "NN accumulator updates: " << numUpdates << std::endl;
    std::cout << "NN accumulator refreshes: " << numRefreshes << std::endl;
}
#endif // NN_ACCUMULATOR_STATS

void UniversalChessInterface::DoSearch()
{
    mSearchCtx->searchParam.stopSearch = false;
    mSearchCtx->searchStarted.store(true, std::memory_order_release);

    mTranspositionTable.NextGeneration();
    mSearch.DoSearch(mGame, mSearchCtx->searchParam, mSearchCtx->searchResult);

    // make sure we're not pondering (search was either stopped or 'ponderhit' was called)
    while (mSearchCtx->searchParam.isPonder.load(std::memory_order_acquire))
        ;

    // report best move
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

        std::cout << std::endl;

#ifdef NN_ACCUMULATOR_STATS
        PrintNNEvaluatorStats();
#endif // NN_ACCUMULATOR_STATS
    }
    
    mSearchCtx->waitable.OnFinished();
}

bool UniversalChessInterface::Command_Stop()
{
    if (mSearchCtx)
    {
        // wait for previous search to complete
        mSearchCtx->searchParam.stopSearch = true;
        mSearchCtx->searchParam.isPonder = false;
        mSearchCtx->waitable.Wait();
    }

    mSearchCtx.reset();

    return true;
}

bool UniversalChessInterface::Command_PonderHit()
{
    if (mSearchCtx)
    {
        mSearchCtx->ponderHit = true;
        mSearchCtx->searchParam.isPonder.store(false, std::memory_order_release);
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
        mOptions.multiPV = std::clamp((uint32_t)atoi(value.c_str()), 1u, MaxAllowedMoves);
    }
    else if (lowerCaseName == "threads")
    {
        uint32_t newNumThreads = atoi(value.c_str());
        newNumThreads = std::max(1u, std::min(c_MaxNumThreads, newNumThreads));

        if (mOptions.threads != newNumThreads)
        {
            mSearch.StopWorkerThreads();
            mOptions.threads = newNumThreads;
        }
    }
    else if (lowerCaseName == "moveoverhead")
    {
        mOptions.moveOverhead = std::clamp(atoi(value.c_str()), 0, 10000);
    }
    else if (lowerCaseName == "evalrandomization")
    {
        mOptions.evalRandomization = std::clamp(atoi(value.c_str()), 0, 100);
    }
    else if (lowerCaseName == "staticcontempt")
    {
        mOptions.staticContempt = std::clamp(atoi(value.c_str()), -1000, 1000);
    }
    else if (lowerCaseName == "dynamiccontempt")
    {
        mOptions.dynamicContempt = std::clamp(atoi(value.c_str()), -1000, 1000);
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
    else if (lowerCaseName == "uci_showwdl")
    {
        if (!ParseBool(lowerCaseValue, mOptions.showWDL))
        {
            std::cout << "Invalid value" << std::endl;
            return false;
        }
    }
#ifdef USE_SYZYGY_TABLEBASES
    else if (lowerCaseName == "syzygypath")
    {
        LoadSyzygyTablebase(value.c_str());
    }
    else if (lowerCaseName == "syzygyprobelimit")
    {
        g_syzygyProbeLimit = std::clamp(atoi(value.c_str()), 4, 7);
    }
#endif // USE_SYZYGY_TABLEBASES
#ifdef USE_GAVIOTA_TABLEBASES
    else if (lowerCaseName == "gaviotatbpath")
    {
        LoadGaviotaTablebase(value.c_str());
    }
    else if (lowerCaseName == "gaviotatbcache")
    {
        const size_t cacheSize = 1024 * 1024 * static_cast<size_t>(std::max(1, atoi(value.c_str())));
        SetGaviotaCacheSize(cacheSize);
    }
#endif // USE_GAVIOTA_TABLEBASES
    else if (lowerCaseName == "evalfile")
    {
        LoadMainNeuralNetwork(value.c_str());
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
    else
    {
#ifdef ENABLE_TUNING
        for (const TunableParameter& param : g_TunableParameters)
        {
            if (name == param.m_name)
            {
                param.m_value = std::clamp(atoi(value.c_str()), param.m_min, param.m_max);
                return true;
            }
        }
#endif // ENABLE_TUNING

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
        std::cout << "Moves:      " << ttEntry.move.ToString() << std::endl;
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

bool UniversalChessInterface::Command_NodeCacheProbe()
{
    const NodeCacheEntry* entry = mSearch.GetNodeCache().TryGetEntry(mGame.GetPosition());

    if (entry)
    {
        std::cout << "Node Cache entry found!" << std::endl;
        entry->PrintMoves();
    }
    else
    {
        std::cout << "Node Cache not found" << std::endl;
    }

    return true;
}


bool UniversalChessInterface::Command_ScoreMoves()
{
    MoveList moves;
    GenerateMoveList(mGame.GetPosition(), moves);

    NodeInfo nodeInfo;
    nodeInfo.position = mGame.GetPosition();
    mGame.GetPosition().ComputeThreats(nodeInfo.threats);

    const NodeCacheEntry* nodeCacheEntry = mSearch.GetNodeCache().TryGetEntry(mGame.GetPosition());

    mSearch.GetMoveOrderer().ScoreMoves(nodeInfo, moves, true, nodeCacheEntry);

    moves.Sort();
    PrintMoveList(mGame.GetPosition(), moves);

    return true;
}

bool UniversalChessInterface::Command_Benchmark()
{
    const char* testPositions[] =
    {
        "1brr2k1/1p3p2/p7/P2N2p1/1P2p2p/2P1P2P/1Q2KPq1/R2R4 w - - 0 32",
        "1k2r3/1p2r2q/2b2b2/1pPpNp2/pP1P1Ppn/P2QP1Bp/7P/K1R1N1R1 w - - 10 35",
        "1k6/1pp1r2p/1n6/p1b1PQ2/P1P5/1q3NP1/3B2KP/4R3 w - - 1 36",
        "1k6/5R2/1pn1q1p1/2p3p1/3p4/1P1P1Q2/1KP2P2/8 w - - 6 33",
        "1kr5/1pp4R/3bNq2/3P4/1r4pP/6P1/2Q2PK1/3R4 w - - 13 39",
        "1r1qkbnr/2pppppp/pp6/8/P1PP4/N6N/1P1PBPbP/R1BQK2R w KQk - 0 3",
        "1r4k1/1b1np2p/p1q1pbp1/R1Pr4/2QB4/2P2N2/4BPPP/1R4K1 w - - 11 23",
        "1r4k1/1pNb1p1p/5b2/2pBnp2/p1P2p2/2P5/PP1R2PP/3R2K1 w - - 0 22",
        "1r4k1/1q4p1/3rpb1p/p1p1Np1b/2P2P2/RnBPRN2/1PQ3PP/7K w - - 2 27",
        "1r4k1/5bpp/8/2b1p3/8/2Nr4/1P4PP/R1B2R1K w - - 2 25",
        "1R6/1p3r2/1P1p2p1/3N2k1/3K2Pp/5P2/2b3P1/8 w - - 1 60",
        "1r6/6k1/2n1p1p1/p1Ppr1q1/P6p/2P4Q/KPB3P1/3R1R2 w - - 0 39",
        "1R6/8/5p1k/p3nB1P/6P1/8/5K2/r7 w - - 0 59",
        "1R6/8/8/1p5k/1P4p1/n1P5/2r2BK1/8 w - - 33 67",
        "1rbqkbnr/ppn1pppp/8/2p5/P1pP4/2N3PP/1P2PP2/R1BQKBNR w KQk - 0 2",
        "1rbr3k/2N1b1pp/p3P3/p4p2/1P6/6P1/4PKBP/3R3R w - - 0 24",
        "2b1k3/4b1p1/1pp1pn2/q2n1p2/3B1P2/1P1QPBP1/2NP1N2/4K3 w - - 0 17",
        "2k5/2p2p2/2n1p1p1/1pPrP3/p4P1p/P3n1rP/PB2R1P1/2R1N1K1 w - - 0 32",
        "2k5/2p3pp/1p2b3/p3Pp2/2P1rP2/1P1r3P/P1R1NKP1/5R2 w - - 6 27",
        "2r1k2r/1p2bp2/4pn1p/pN1pN1p1/3P4/2P3P1/P3PPqP/1R1QKR2 w k - 3 12",
        "2r3k1/7p/1P1B1R2/2Ppp3/pr6/4P3/P5P1/6K1 w - - 0 43",
        "2r3k1/pp2q1pn/7p/2PRpr2/2P1p3/2B1P2P/4QPP1/5RK1 w - - 3 19",
        "2r4k/8/7p/1PNpR3/3r2b1/P5Qq/6RP/7K w - - 1 39",
        "2r5/r1Pq1ppk/4p2p/P1R5/5Q2/4P2P/5PK1/2R5 w - - 3 44",
        "3b4/3qk3/p5p1/4P2p/3Q4/P7/5K2/8 w - - 0 54",
        "3k4/5p1p/2bp2pP/2p1p1P1/2P1P3/1K1P2P1/8/R7 w - - 16 89",
        "3q3k/r4bp1/5p1p/2pPp3/pr1nB3/3PN1P1/PP4QP/4RRK1 w - - 8 30",
        "3r2k1/4npp1/1pqrp2p/p1p4P/P1P2PP1/1PBP4/3RQ3/5RK1 w - - 1 26",
        "3r2k1/6b1/8/p4p2/N1p2P1p/2PpK2P/6P1/1R6 w - - 0 37",
        "3r4/4kp2/4p1p1/4P2p/1p2n3/1B1R2PP/1PK2P2/8 w - - 0 38",
        "3R4/8/8/3pn1P1/4bk2/8/8/3K4 w - - 0 77",
        "3rk2r/1bpqbp2/np1ppn1p/p5p1/2PPP3/P1N2NP1/1PQ2PBP/R1BR2K1 w k - 4 15",
        "4B3/8/3b2p1/4kp2/6rp/4B3/3R1KP1/8 w - - 2 64",
        "4Bb2/5P2/1p3K2/1P2P3/8/5k2/8/8 w - - 3 113",
        "4k3/p6Q/2p5/2P2rP1/3Pq3/4p2K/P5R1/8 w - - 13 48",
        "4r1k1/P2R4/5p2/3K1P2/5P2/8/8/8 w - - 1 64",
        "4r3/5p2/3pp3/pPkp1p1p/P4PrP/K1PRR1P1/8/8 w - - 54 100",
        "4rk2/3n1p2/3r3p/q1pPpB1n/1pP1P2B/p5Q1/P4PP1/2RR2K1 w - - 1 36",
        "4rrk1/1R3pb1/p2p2p1/3nn1B1/6qP/6P1/3NNP2/3Q1RK1 w - - 5 21",
        "5k2/1p2b1p1/p1pn1p1p/P6P/1P2PPN1/2PQ2P1/q3B1K1/8 w - - 2 44",
        "5k2/3np3/3p1n2/7p/7P/1Q6/6P1/6K1 w - - 35 58",
        "5n2/4kp2/NR2p1p1/4P3/b1B2P2/4K1P1/8/8 w - - 3 40",
        "5qk1/1pn3bp/1n1p2p1/r3p3/2PPPPb1/p2BQNP1/P1P1N3/1K1R3R w - - 0 19",
        "5rk1/R4ppp/3p1b2/1r1n4/2pp4/6P1/1P2PPBP/2B2RK1 w - - 1 25",
        "6B1/8/7p/1pk1p1p1/4KbP1/5P2/8/8 w - - 4 46",
        "6k1/5pp1/4p2p/8/3PqP2/4P2P/6P1/3Q2K1 w - - 0 29",
        "6k1/6p1/8/5P2/4K3/R3B2r/8/5r2 w - - 36 70",
        "6k1/6p1/8/P5q1/1P6/7Q/3r2B1/6K1 w - - 11 45",
        "6k1/BBn2p1p/6p1/p7/P2R4/1bP1K3/6PP/5r2 w - - 6 27",
        "6k1/p3b2p/1p4p1/1Ppp2P1/3P4/P2RP3/2r3r1/1K1N3R w - - 0 33",
        "6R1/2p5/1p3p2/1P6/6pk/2BP1r2/4K3/8 w - - 3 59",
        "7k/7P/8/P1b3p1/2P1B1P1/1K3P2/8/8 w - - 1 80",
        "7r/1p1rqpk1/pR2p3/P2pP2p/1P3QpP/2pBP3/2Pn1PP1/3R2K1 w - - 3 25",
        "8/1k6/3R1p1p/2P1n3/3N1r2/8/4K3/8 w - - 4 88",
        "8/1N3kp1/5p1p/4n3/6P1/3P4/5K2/8 w - - 5 48",
        "8/1p3k2/1b1r1pp1/7p/4pP1B/PRP5/4K1PP/8 w - - 0 29",
        "8/1p4r1/p2P3p/P1R3p1/6P1/5k1P/5P2/5K2 w - - 0 52",
        "8/1p5p/p1p1kpp1/P7/1P1R1PP1/4P2P/6K1/1r6 w - - 8 40",
        "8/1p6/p4pk1/P2p2p1/1Pq1b2p/2P1P2P/3Qn1PK/3R4 w - - 4 39",
        "8/1q4k1/5pp1/8/1N3P2/pQ4P1/1b1K4/8 w - - 6 73",
        "8/1r2pk2/3p3R/1p1P3P/P5P1/1P6/6K1/8 w - - 0 37",
        "8/2B2p2/4bP2/7p/p3k2P/3p2P1/1P1K4/8 w - - 21 108",
        "8/2R5/6k1/3N4/4Ppb1/3p2p1/3Nn1P1/4K3 w - - 0 47",
        "8/3k4/5PP1/5K2/3p4/r7/4P3/8 w - - 0 62",
        "8/3q1pkp/1p3p2/2pr4/1p1p1P1P/1P1Q2P1/P3RP2/6K1 w - - 5 25",
        "8/4kpp1/6p1/2br2P1/1p2pP2/pP2P2P/K1P1R3/2B5 w - - 4 33",
        "8/4p1k1/3p4/3P4/5P1p/2b2K1B/8/8 w - - 32 102",
        "8/4p1k1/4Q1p1/p3P2p/1b3P2/6PP/P5K1/3r4 w - - 7 46",
        "8/4R3/P1Bk4/1P6/r7/3K4/8/6b1 w - - 15 78",
        "8/5k2/p5p1/P2p4/2nB1PP1/4P1KP/8/8 w - - 3 50",
        "8/5p2/8/4pk2/6R1/4P2P/n2PKP1P/1r6 w - - 4 36",
        "8/6p1/2k4p/4p1PP/4N3/b3PK2/1p1N4/2n5 w - - 0 56",
        "8/6pk/4p1q1/r1n4p/2R5/6P1/3NQ2K/8 w - - 2 32",
        "8/8/1R6/1p6/1bk2B2/8/5p2/3n1K2 w - - 3 114",
        "8/8/3r4/p3n3/4k3/1P5Q/P7/2K5 w - - 1 55",
        "8/8/4k1p1/1R6/4r1pP/6P1/5P2/5K2 w - - 76 100",
        "8/8/5k2/1R4p1/4Kn2/7b/2B4P/8 w - - 26 70",
        "8/8/6b1/6k1/3r2p1/4K3/p7/7R w - - 0 94",
        "8/8/7P/5NK1/5PP1/3k1q2/4p3/8 w - - 0 107",
        "8/8/8/1Rp1r3/P1r2PK1/1N1k4/8/8 w - - 1 79",
        "N1b2k1r/pp1pqpp1/5n2/3P2p1/1b2p2n/4P3/PPPB1PP1/R2QKB1R w KQ - 0 12",
        "r1b1k2r/1pq1n1pp/p4n2/3p1p2/Pb1Pp3/1PN1P2P/2Q1BPP1/RNB1K2R w KQkq - 5 14",
        "r1b2rk1/1pq1p1b1/2p1p1pn/p2p2Np/N2P4/6PP/PPPQ1PB1/4RRK1 w - - 0 17",
        "r1bqk2r/1p2pnbp/2p3p1/3p1p2/p2PnP2/P1PN2P1/1P1NP1BP/R1BQ1RK1 w kq - 2 7",
        "r1bqkb1r/1p4pp/2p2n2/p2n1p2/5P2/2N1P2P/PP2N1P1/R1BQKB1R w KQkq - 0 8",
        "r1bqkb1r/pp3ppp/2n1pn2/2Pp4/3P1B2/5N2/PP1N1PPP/R2QKB1R w KQkq - 1 9",
        "r1br2k1/1p3ppp/5n2/2p5/1R2P3/P4NP1/4PPBP/4K2R w K - 0 17",
        "r1br4/2p2pk1/P1R3pn/3Pp2p/n3P3/8/P2NBPPP/3R2K1 w - - 3 25",
        "r1q2bk1/1b3ppp/2p1p3/pp6/3PPP2/8/PP3PBP/R2Q1RK1 w - - 0 19",
        "r2q1rk1/1bpp1pp1/1p2pn1p/p7/PbPP1P2/1P2PN2/R3B1PP/2BQ1RK1 w - - 1 8",
        "r2q1rk1/p2nbpp1/2n1b2p/1p2p3/1P1pP3/P2P1NPP/3N1PB1/1RBQ1RK1 w - - 0 14",
        "r2q2k1/1pp1br2/2n1bpp1/3Np2p/2PpP1nP/PN1P2P1/5PB1/1RBQ1RK1 w - - 1 16",
        "r2qk1nr/pp2b1pp/6n1/2pB3Q/P3N1b1/1P6/2PP1PPP/1RB1K2R w Kkq - 1 7",
        "r2qk1nr/ppp3b1/3pp1pp/8/b1P1P3/4B2P/PP2NPP1/R1Q1KB1R w KQkq - 0 9",
        "r2qkb1r/1bpn1p1p/pp4n1/3p4/3QNB2/7P/PPP2PP1/2KR1BNR w kq - 1 8",
        "r2qkbnr/3n1ppp/ppbp4/4p3/1P6/P1N1P2P/1BPQ1PP1/R3KBNR w KQkq - 2 7",
        "r3k1nr/ppbb2pp/4p3/2Bp4/P2P1q1P/2PB4/1P3P2/RN1QK2R w KQkq - 0 11",
        "r3k2r/1p2q1p1/2p2n1p/p2p4/Nb1PpBbP/1Q4P1/PP2PP2/R2K1BR1 w kq - 6 16",
        "r3k2r/pp2qppp/2p1b3/2b1P3/4B3/2P5/PPP3PP/R1BQR2K w kq - 0 15",
        "r3k2r/ppp1q1pp/2nbbp2/3p4/8/2PQ1N2/PPP2BPP/2KR1B1R w kq - 6 12",
        "r3kb1r/ppp1qp1p/2npbnp1/8/4PP2/2N1Q3/PPPB2PP/2KR1BNR w kq - 0 9",
        "r4bk1/2p2p1p/4p1p1/2nnP1NP/8/P1q3P1/2B2PK1/2BQR3 w - - 5 28",
        "r4rk1/1pp2p1p/6p1/n3q3/1pB1P1b1/8/P1QN1PPP/R4RK1 w - - 0 17",
        "r4rk1/2qn1pbp/3Npnp1/p1Pb2B1/1p1P4/1P3NP1/2Q2PBP/R2R2K1 w - - 2 18",
        "r4rk1/p1pq2bp/2pp2p1/4p1B1/4P3/2NP2PP/PPPQ4/R3K2b w Q - 0 14",
        "r5k1/1p3pp1/2p4p/3n1b2/2Nb4/P7/5PPP/2RBR1K1 w - - 4 24",
        "r5kr/ppB2ppp/8/2n5/5Pb1/3B4/PPP3PP/R3K2R w KQ - 1 20",
        "r6k/2p2q2/2n2p1p/2Np1bp1/3P3n/2B1P2P/4BPP1/3Q1RK1 w - - 5 24",
        "r7/1p4k1/3Prnpp/pP6/3B4/6P1/4P2P/5QK1 w - - 1 26",
        "r7/3r4/k2p1p2/2pPp1p1/p1N1PnPp/Pp3P1P/1P1K4/3R3R w - - 48 60",
        "rn1qk1nr/1p1b1p2/p2p2pb/2pPp2p/P1P1P2P/2NB1P2/1P1BN1P1/R2QK2R w KQkq - 5 13",
        "rn1qr1k1/2pb1ppp/1p3n2/p2pN3/Pb1P1P2/2NB4/1PPQ2PP/R1B2RK1 w - - 7 8",
        "rn2k1nr/1b3ppp/1p1qp3/1P1p4/1p6/3PPN2/P3BPPP/RN1Q1RK1 w kq - 2 7",
        "rn2k2r/pp2b1p1/2p1bq1p/3p1p2/3Pp3/PN2P1P1/2P1BP1P/RN1QK2R w KQkq - 3 12",
        "rn2kbnr/1p1bqp2/p1pp2p1/7p/P2P4/2N3N1/1PP1BPPP/R1BQK2R w KQkq - 0 5",
        "rn2r1k1/pb2bppp/1p1p1n2/3P1q2/1PP5/P4B2/1B1N1PPP/R2Q1RK1 w - - 5 9",
        "rn3rk1/p5p1/2P2ppp/8/1bpq2P1/7P/1P1NPPB1/R2QK2R w KQ - 0 15",
        "rnb1k2r/pp1pbpp1/2p1p2p/7B/4P2n/3P3P/q1PN1PPB/1R1QK1NR w Kkq - 4 5",
        "rnbq1rk1/ppp1bppp/4p3/3n4/2B1N3/5N2/PPPP1PPP/R1BQR1K1 w - - 8 9",
        "rnbqkb1r/1p3p2/p3p1p1/2ppP1P1/3P2P1/2P4p/PP5P/RNBQKB1R w KQkq - 0 6",
        "3r3k/7p/2Q5/8/2B2PK1/6P1/4P3/5q2 b - - 98 99",
    };

    const uint32_t maxDepth = 12;

    Search search;
    TranspositionTable tt(4 * 1024 * 1024);

    uint64_t totalNodes = 0;
    double totalTime = 0.0;

    for (const char* testPosition : testPositions)
    {
        printf("Benchmarking position: %s ...", testPosition);

        Position pos;
        VERIFY(pos.FromFEN(testPosition));

        Game game;
        game.Reset(pos);

        search.Clear();
        tt.Clear();

        SearchParam searchParam{ tt };
        searchParam.debugLog = false;
        searchParam.limits.maxDepth = maxDepth;

        const TimePoint startTimePoint = TimePoint::GetCurrent();

        SearchStats stats;
        SearchResult searchResult;
        search.DoSearch(game, searchParam, searchResult, &stats);

        const TimePoint endTimePoint = TimePoint::GetCurrent();

        totalNodes += stats.nodes;
        totalTime += (endTimePoint - startTimePoint).ToSeconds();

        // print best move and stats
        printf(" Move: %s, Nodes: %" PRId64 ", Time: %.2f MNPS: %.2f\n",
            searchResult[0].moves.front().ToString().c_str(),
            stats.nodes.load(),
            (endTimePoint - startTimePoint).ToSeconds(),
            stats.nodes.load() / (endTimePoint - startTimePoint).ToSeconds() / 1000000.0);
    }

    std::cout << totalNodes << " nodes " << static_cast<int64_t>(totalNodes / totalTime) << " nps" << std::endl;

#ifdef NN_ACCUMULATOR_STATS
    PrintNNEvaluatorStats();
#endif // NN_ACCUMULATOR_STATS

    return true;
}
