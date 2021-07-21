#include "UCI.hpp"
#include "Move.hpp"
#include "Evaluate.hpp"

#include "tablebase/tbprobe.h"

// TODO set TT size based on current memory usage / total memory size
#ifndef _DEBUG
static const uint32_t c_DefaultTTSize = 32 * 1024 * 1024;
#else
static const uint32_t c_DefaultTTSize = 1024 * 1024;
#endif

extern void RunUnitTests();
extern void RunPerft();
extern bool RunSearchTests(const char* path);
extern void SelfPlay();
extern bool Train();

void LoadTablebase(const char* path)
{
    if (tb_init(path))
    {
        std::cout << "Tablebase loaded successfully. Size = " << TB_LARGEST << std::endl;
    }
    else
    {
        std::cout << "Failed to load tablebase" << std::endl;
    }
}

UniversalChessInterface::UniversalChessInterface(int argc, const char* argv[])
{
    mGame.Reset(Position(Position::InitPositionFEN));
    mSearch.GetTranspositionTable().Resize(c_DefaultTTSize);

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

    tb_free();
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
        std::cout << "option name Hash type spin default " << c_DefaultTTSize << " min 1 max 1048576\n";
        std::cout << "option name MultiPV type spin default 1 min 1 max 255\n";
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
        // TODO
    }
    else if (command == "stop")
    {
        std::unique_lock<std::mutex> lock(mMutex);
        Command_Stop();
    }
    else if (command == "quit")
    {
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
        const size_t numEntriesUsed = mSearch.GetTranspositionTable().GetNumUsedEntries();
        const float percentage = 100.0f * (float)numEntriesUsed / (float)mSearch.GetTranspositionTable().GetSize();
        std::cout << "TT entries in use: " << numEntriesUsed << " (" << percentage << "%)" << std::endl;
    }
    else if (command == "unittest")
    {
        RunUnitTests();
        std::cout << "Unit tests done." << std::endl;
    }
    else if (command == "selfplay")
    {
        SelfPlay();
    }
    else if (command == "train")
    {
        Train();
    }
    else if (command == "searchtest")
    {
        RunSearchTests(args[1].c_str());
        std::cout << "Search tests done." << std::endl;
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
            std::cout << "Invalid FEN" << std::endl;
            return false;
        }
    }

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
    const float steepness = 5.0f;
    return midpoint * std::pow(1.0f + 2.0f * std::pow(move / midpoint, steepness), 1.0f / steepness) - move;
}

bool UniversalChessInterface::Command_Go(const std::vector<std::string>& args)
{
    Command_Stop();

    const auto startTimePoint = std::chrono::high_resolution_clock::now();

    bool isInfinite = false;
    bool isPonder = false;
    bool printMoves = false;
    uint32_t maxDepth = UINT8_MAX;
    uint64_t maxNodes = UINT64_MAX;
    uint32_t moveTime = UINT32_MAX;
    uint32_t whiteRemainingTime = UINT32_MAX;
    uint32_t blacksRemainingTime = UINT32_MAX;
    uint32_t whiteTimeIncrement = 0;
    uint32_t blacksTimeIncrement = 0;
    uint32_t movesToGo = UINT32_MAX;

    std::vector<Move> rootMoves;

    for (size_t i = 1; i < args.size(); ++i)
    {
        if (args[i] == "depth" && i + 1 < args.size())
        {
            maxDepth = atoi(args[i + 1].c_str());
        }
        else if (args[i] == "infinite")
        {
            isInfinite = true;
        }
        else if (args[i] == "ponder")
        {
            isPonder = true;
        }
        else if (args[i] == "printMoves")
        {
            printMoves = true;
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

    // calculate time for move based on total remaining time and other heuristics
    uint32_t timeEstimatedMs = UINT32_MAX;
    {
        const float minTimePerMove = 1; // make configurable
        const float moveOverhead = 20; // make configurable
        const uint32_t remainingTime = mGame.GetSideToMove() == Color::White ? whiteRemainingTime : blacksRemainingTime;
        const uint32_t remainingTimeInc = mGame.GetSideToMove() == Color::White ? whiteTimeIncrement : blacksTimeIncrement;

        if (remainingTime != UINT32_MAX)
        {
            const float movesLeftEstimated = EstimateMovesLeft(static_cast<float>(mGame.GetMoves().size()));
            const float timeEstimated = std::min((float)remainingTime, remainingTime / movesLeftEstimated + remainingTimeInc);

            timeEstimatedMs = static_cast<uint32_t>(std::max(minTimePerMove, timeEstimated - moveOverhead) + 0.5f);
        }
    }

    mSearchCtx = std::make_unique<SearchTaskContext>();

    mSearchCtx->searchParam.startTimePoint = startTimePoint;
    mSearchCtx->searchParam.limits.maxTime = std::min(moveTime, timeEstimatedMs);
    mSearchCtx->searchParam.limits.maxDepth = (uint8_t)std::min<uint32_t>(maxDepth, UINT8_MAX);
    mSearchCtx->searchParam.limits.maxNodes = maxNodes;
    mSearchCtx->searchParam.numPvLines = mOptions.multiPV;
    mSearchCtx->searchParam.rootMoves = std::move(rootMoves);
    mSearchCtx->searchParam.printMoves = printMoves;

    threadpool::TaskDesc taskDesc;
    taskDesc.waitable = &mSearchCtx->waitable;
    taskDesc.function = [this](const threadpool::TaskContext&)
    {
        mSearch.DoSearch(mGame, mSearchCtx->searchParam, mSearchCtx->searchResult);

        const auto& bestLine = mSearchCtx->searchResult[0].moves;

        if (!bestLine.empty())
        {
            std::cout << "bestmove " << bestLine[0].ToString();

            if (bestLine.size() > 1)
            {
                std::cout << " ponder " << bestLine[1].ToString();
            }

            std::cout << std::endl;
        }
    };

    threadpool::ThreadPool().GetInstance().CreateAndDispatchTask(taskDesc);

    return true;
}

bool UniversalChessInterface::Command_Stop()
{
    if (mSearchCtx)
    {
        mSearch.StopSearch();

        mSearchCtx->waitable.Wait();
        mSearchCtx.reset();
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

static void ToLower(std::string& str)
{
    for (char& c : str)
    {
        if (c <= 'Z' && c >= 'A')
        {
            c = (c - ('Z' - 'z'));
        }
    }
}

bool UniversalChessInterface::Command_SetOption(const std::string& name, const std::string& value)
{
    std::string lowerCaseName = name;
    ToLower(lowerCaseName);

    if (lowerCaseName == "multipv")
    {
        mOptions.multiPV = atoi(value.c_str());
        mOptions.multiPV = std::max(1u, mOptions.multiPV);
    }
    else if (lowerCaseName == "hash")
    {
        size_t hashSize = 1024 * 1024 * static_cast<size_t>(atoi(value.c_str()));
        size_t numEntries = hashSize / sizeof(TranspositionTableEntry);
        mSearch.GetTranspositionTable().Resize(numEntries);
    }
    else if (lowerCaseName == "syzygypath")
    {
        LoadTablebase(value.c_str());
    }
    else
    {
        std::cout << "Invalid option: " << name << std::endl;
        return false;
    }

    return true;
}