#include "UCI.hpp"
#include "Move.hpp"
#include "Evaluate.hpp"

#include <sstream>
#include <iostream>

extern void RunUnitTests();
extern void RunPerft();
extern bool RunSearchTests();
extern void RunSearchPerfTest();

UniversalChessInterface::UniversalChessInterface()
{
    mPosition.FromFEN(Position::InitPositionFEN);
}

bool UniversalChessInterface::Loop()
{
    std::string str;

    while (std::getline(std::cin, str))
    {
        std::istringstream iss(str);
        std::vector<std::string> args(
            std::istream_iterator<std::string>{iss},
            std::istream_iterator<std::string>());

        if (args.empty())
        {
            std::cout << "Invalid command" << std::endl;
            continue;
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
            std::unique_lock<std::mutex> lock(mMutex);
            // TODO
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
            break;
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
            std::cout << "TT entries in use: " << mSearch.GetTranspositionTable().GetNumUsedEntries() << std::endl;
        }
        else if (command == "unittest")
        {
            RunUnitTests();
            std::cout << "Unit tests done." << std::endl;
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
        }
    }

    return true;
}

bool UniversalChessInterface::Command_Go(const std::vector<std::string>& args)
{
    bool isInfinite = false;
    uint32_t maxDepth = 8; // TODO
    uint64_t maxNodes = UINT64_MAX;
    uint32_t moveTime = UINT32_MAX;
    uint32_t whiteRemainingTime = 0;
    uint32_t blacksRemainingTime = 0;
    uint32_t whiteTimeIncrement = 0;
    uint32_t blacksTimeIncrement = 0;

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
            // TODO
            // restrict search to this moves only
            // Example : After "position startpos" and "go infinite searchmoves e2e4 d2d4"
            // the engine should only search the two moves e2e4 and d2d4 in the initial position.
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
    searchParam.maxDepth = (uint8_t)std::min<uint32_t>(maxDepth, UINT8_MAX);
    //searchParam.numPvLines = 4;

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