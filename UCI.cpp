#include "UCI.hpp"
#include "Move.hpp"
#include "Evaluate.hpp"

#include "tablebase/tbprobe.h"

extern void RunUnitTests();
extern void RunPerft();
extern bool RunSearchTests();
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
    mPosition.FromFEN(Position::InitPositionFEN);

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
        // TODO
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
        std::cout << mPosition.Print() << std::endl;
    }
    else if (command == "eval")
    {
        std::cout << Evaluate(mPosition) << std::endl;
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
        RunSearchTests();
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

    if (args.size() >= 2 && args[1] == "startpos")
    {
        mPosition.FromFEN(Position::InitPositionFEN);

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

        if (!mPosition.FromFEN(fenString))
        {
            std::cout << "Invalid FEN" << std::endl;
            return false;
        }
    }

    if (extraMovesStart > 0)
    {
        for (size_t i = extraMovesStart + 1; i < args.size(); ++i)
        {
            const Position prevPosition = mPosition;
            const Move move = mPosition.MoveFromString(args[i]);

            if (!move.IsValid() || !mPosition.IsMoveValid(move))
            {
                std::cout << "Invalid move" << std::endl;
                return false;
            }

            if (!mPosition.DoMove(move))
            {
                std::cout << "Invalid move" << std::endl;
                return false;
            }

            mSearch.RecordBoardPosition(prevPosition);
        }
    }

    return true;
}

bool UniversalChessInterface::Command_Go(const std::vector<std::string>& args)
{
    bool isInfinite = false;
    bool printMoves = false;
    uint32_t maxDepth = UINT8_MAX;
    uint64_t maxNodes = UINT64_MAX;
    uint32_t moveTime = UINT32_MAX;
    uint32_t whiteRemainingTime = 0;
    uint32_t blacksRemainingTime = 0;
    uint32_t whiteTimeIncrement = 0;
    uint32_t blacksTimeIncrement = 0;

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
                const Move move = mPosition.MoveFromString(args[j]);
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
            // TODO
            // there are x moves to the next time control,
            // this will only be sent if x > 0,
            // if you don't get this and get the wtime and btime it's sudden death
        }
    }

    SearchParam searchParam;
    searchParam.startTimePoint = std::chrono::high_resolution_clock::now();
    searchParam.maxTime = moveTime;
    searchParam.maxDepth = (uint8_t)std::min<uint32_t>(maxDepth, UINT8_MAX);
    searchParam.numPvLines = mOptions.multiPV;
    searchParam.rootMoves = std::move(rootMoves);
    searchParam.printMoves = printMoves;

    SearchResult searchResult;
    mSearch.DoSearch(mPosition, searchParam, searchResult);

    if (!searchResult[0].moves.empty())
    {
        const Move bestMove = searchResult[0].moves[0];
        std::cout << "bestmove " << bestMove.ToString() << std::endl;
    }

    // TODO ponder

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

    mPosition.Perft(maxDepth, true);

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
        std::cout << "Invalid option" << std::endl;
        return false;
    }

    return true;
}