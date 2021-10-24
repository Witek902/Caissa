#include "UCI.hpp"
#include "../backend/Move.hpp"
#include "../backend/Evaluate.hpp"
#include "../backend/Tablebase.hpp"

#include "../backend/nnue-probe/nnue.h"

#include <math.h>

// TODO set TT size based on current memory usage / total memory size
#ifndef _DEBUG
static const uint32_t c_DefaultTTSize = 16 * 1024 * 1024;
#else
static const uint32_t c_DefaultTTSize = 1024 * 1024;
#endif

UniversalChessInterface::UniversalChessInterface(int argc, const char* argv[])
{
    // init threadpool
    threadpool::ThreadPool::GetInstance();

    mGame.Reset(Position(Position::InitPositionFEN));
    mTranspositionTable.Resize(c_DefaultTTSize);

    for (int i = 1; i < argc; ++i)
    {
        if (argv[i])
        {
            std::cout << "CommandLine: " << argv[i] << std::endl;
            ExecuteCommand(argv[i]);
        }
    }
}

void UniversalChessInterface::Loop()
{
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
        std::cout << "id name MWCE\n";
        std::cout << "id author Michal Witanowski\n";
        std::cout << "\n";
        std::cout << "option name Hash type spin default " << c_DefaultTTSize << " min 1 max 1048576\n";
        std::cout << "option name MultiPV type spin default 1 min 1 max 255\n";
        std::cout << "option name Ponder type check default false\n";
        std::cout << "option name EvalFile type string default nn-04cf2b4ed1da.nnue\n";
        std::cout << "option name SyzygyPath type string default <empty>\n";
        std::cout << "option name UCI_AnalyseMode type check default false\n";
        std::cout << "option name UseSAN type check default false\n";
        std::cout << "uciok" << std::endl;
    }
    else if (command == "isready")
    {
        std::unique_lock<std::mutex> lock(mMutex);
        std::cout << "readyok" << std::endl;
    }
    else if (command == "ucinewgame")
    {
        std::unique_lock<std::mutex> lock(mMutex);
        // TODO
    }
    else if (command == "setoption")
    {
        if (args[1] == "name" && args[3] == "value")
        {
            std::unique_lock<std::mutex> lock(mMutex);
            Command_SetOption(args[2], args.size() > 4 ? args[4] : "");
        }
        else
        {
            std::cout << "Invalid command" << std::endl;
        }
    }
    else if (command == "position")
    {
        std::unique_lock<std::mutex> lock(mMutex);
        Command_Position(args);
    }
    else if (command == "go")
    {
        std::unique_lock<std::mutex> lock(mMutex);
        Command_Go(args);
    }
    else if (command == "ponderhit")
    {
        std::unique_lock<std::mutex> lock(mMutex);
        Command_PonderHit();
    }
    else if (command == "stop")
    {
        std::unique_lock<std::mutex> lock(mMutex);
        Command_Stop();
    }
    else if (command == "quit")
    {
        std::unique_lock<std::mutex> lock(mMutex);
        Command_Stop();
        return false;
    }
    else if (command == "perft")
    {
        Command_Perft(args);
    }
    else if (command == "print")
    {
        std::unique_lock<std::mutex> lock(mMutex);
        std::cout << "Init:  " << mGame.GetInitialPosition().ToFEN() << std::endl << mGame.ToPGN() << std::endl;
        std::cout << mGame.GetPosition().Print() << std::endl;
    }
    else if (command == "eval")
    {
        std::cout << Evaluate(mGame.GetPosition()) << std::endl;
    }
    else if (command == "ttinfo")
    {
        std::unique_lock<std::mutex> lock(mMutex);
        const size_t numEntriesUsed = mTranspositionTable.GetNumUsedEntries();
        const float percentage = 100.0f * (float)numEntriesUsed / (float)mTranspositionTable.GetSize();
        std::cout << "TT entries in use: " << numEntriesUsed << " (" << percentage << "%)" << std::endl;
        std::cout << "TT collisions: " << mTranspositionTable.GetNumCollisions() << std::endl;
    }
    else if (command == "ttprobe")
    {
        Command_TTProbe();
    }
    else if (command == "moveordererstats")
    {
        mSearch.GetMoveOrderer().DebugPrint();
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

    if (args.size() >= 2 && args[1] == "random")
    {
        MaterialKey material;
        material.numWhitePawns = 4;
        material.numBlackPawns = 4;
        material.numWhiteRooks = 4;
        material.numBlackRooks = 4;

        if (!GenerateRandomPosition(material, pos))
        {
            std::cout << "Failed to generate random position" << std::endl;
            return false;
        }

        if (args.size() >= 4 && args[2] == "moves")
        {
            extraMovesStart = 2;
        }
    }

    if (args.size() > 2 && args[1] == "fen")
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

static float EstimateMovesLeft(const float ply)
{
    // based on LeelaChessZero 
    const float move = ply / 2.0f;
    const float midpoint = 50.0f;
    const float steepness = 6.0f;
    return midpoint * std::pow(1.0f + 2.0f * std::pow(move / midpoint, steepness), 1.0f / steepness) - move;
}

bool UniversalChessInterface::Command_Go(const std::vector<std::string>& args)
{
    Command_Stop();

    const auto startTimePoint = std::chrono::high_resolution_clock::now();

    bool isInfinite = false;
    bool isPonder = false;
    bool printMoves = false;
    bool verboseStats = false;
    uint32_t maxDepth = UINT8_MAX;
    uint64_t maxNodes = UINT64_MAX;
    int32_t moveTime = INT32_MAX;
    int32_t whiteRemainingTime = INT32_MAX;
    int32_t blacksRemainingTime = INT32_MAX;
    int32_t whiteTimeIncrement = 0;
    int32_t blacksTimeIncrement = 0;
    uint32_t movesToGo = UINT32_MAX;
    uint32_t mateSearchDepth = 0;

    std::vector<Move> rootMoves;

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
        else if (args[i] == "printmoves")
        {
            printMoves = true;
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
            // restrict search to this moves only
            for (size_t j = i + 1; j < args.size(); ++j)
            {
                const Move move = mGame.GetPosition().MoveFromString(args[j]);
                if (move.IsValid())
                {
                    rootMoves.push_back(move);
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

    // calculate time for move based on total remaining time and other heuristics
    {
        const int32_t moveOverhead = 5; // make configurable
        const int32_t remainingTime = mGame.GetSideToMove() == Color::White ? whiteRemainingTime : blacksRemainingTime;
        const int32_t remainingTimeInc = mGame.GetSideToMove() == Color::White ? whiteTimeIncrement : blacksTimeIncrement;

        // soft limit
        if (remainingTime != INT32_MAX)
        {
            const float movesLeftEstimated = movesToGo != UINT32_MAX ? (float)movesToGo : EstimateMovesLeft((float)mGame.GetMoves().size());
            const float timeEstimated = std::min((float)remainingTime, (float)remainingTime / movesLeftEstimated + (float)remainingTimeInc);
            const int32_t timeEstimatedMs = static_cast<uint32_t>(std::max(0.0f, timeEstimated) + 0.5f);

            // use at least 75% of estimated time
            // TODO some better heuristics:
            // for example, estimate time spent in each iteration based on previous searches
            mSearchCtx->searchParam.limits.maxTimeSoft = timeEstimatedMs * 3 / 4;
        }

        // hard limit
        int32_t hardLimit = std::min(remainingTime, moveTime);
        if (hardLimit != INT32_MAX)
        {
            hardLimit = std::max(0, hardLimit - moveOverhead);
            mSearchCtx->searchParam.limits.maxTime = hardLimit;
        }
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

    mSearchCtx->searchParam.startTimePoint = startTimePoint;
    mSearchCtx->searchParam.limits.maxDepth = (uint8_t)std::min<uint32_t>(maxDepth, UINT8_MAX);
    mSearchCtx->searchParam.limits.maxNodes = maxNodes;
    mSearchCtx->searchParam.limits.mateSearch = mateSearchDepth > 0;
    mSearchCtx->searchParam.limits.analysisMode = !isPonder && (isInfinite || mOptions.analysisMode); // run full analysis when pondering
    mSearchCtx->searchParam.numPvLines = mOptions.multiPV;
    mSearchCtx->searchParam.numThreads = mOptions.threads;
    mSearchCtx->searchParam.rootMoves = std::move(rootMoves);
    mSearchCtx->searchParam.printMoves = printMoves;
    mSearchCtx->searchParam.verboseStats = verboseStats;
    mSearchCtx->searchParam.moveNotation = mOptions.useStandardAlgebraicNotation ? MoveNotation::SAN : MoveNotation::LAN;

    RunSearchTask();

    return true;
}

void UniversalChessInterface::RunSearchTask()
{
    threadpool::TaskDesc taskDesc;
    taskDesc.waitable = &mSearchCtx->waitable;
    taskDesc.function = [this](const threadpool::TaskContext&)
    {
        mSearch.DoSearch(mGame, mSearchCtx->searchParam, mSearchCtx->searchResult);

        const auto endTimePoint = std::chrono::high_resolution_clock::now();
        const auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTimePoint - mSearchCtx->searchParam.startTimePoint).count();

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

            if (!bestMove.IsValid()) // null move
            {
                std::cout << "bestmove 0000";
            }

            std::cout << std::endl;
        }

        if (mSearchCtx->searchParam.verboseStats)
        {
            std::cout << "Elapsed time: " << elapsedTime << " ms" << std::endl;
        }
    };

    threadpool::ThreadPool::GetInstance().CreateAndDispatchTask(taskDesc);
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
        mSearchCtx->searchParam.startTimePoint = std::chrono::high_resolution_clock::now();

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
        mOptions.threads = std::max(1u, std::min(64u, mOptions.threads));
    }
    else if (lowerCaseName == "hash" || lowerCaseName == "hashsize")
    {
        size_t hashSize = 1024 * 1024 * static_cast<size_t>(atoi(value.c_str()));
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
#ifdef USE_TABLE_BASES
    else if (lowerCaseName == "syzygypath")
    {
        LoadTablebase(value.c_str());
    }
#endif // USE_TABLE_BASES
    else if (lowerCaseName == "evalfile")
    {
        nnue_init(value.c_str());
    }
    else if (lowerCaseName == "ponder")
    {
        // nothing special here
    }
    else
    {
        std::cout << "Invalid option: " << name << std::endl;
        return false;
    }

    return true;
}

bool UniversalChessInterface::Command_TTProbe()
{
    std::unique_lock<std::mutex> lock(mMutex);

    TTEntry ttEntry;

    if (mTranspositionTable.Read(mGame.GetPosition(), ttEntry))
    {
        const char* boundsStr =
            ttEntry.flag == TTEntry::Flag_Exact ? "exact" :
            ttEntry.flag == TTEntry::Flag_UpperBound ? "upper" :
            ttEntry.flag == TTEntry::Flag_LowerBound ? "lower" :
            "invalid";

        std::cout << "Score:      " << ttEntry.score << std::endl;
        std::cout << "StaticEval: " << ttEntry.staticEval << std::endl;
        std::cout << "Depth:      " << (uint32_t)ttEntry.depth << std::endl;
        std::cout << "Bounds:     " << boundsStr << std::endl;
    }
    else
    {
        std::cout << "(no entry found)" << std::endl;
    }

    return true;
}
