#pragma once

#include "../backend/Game.hpp"
#include "../backend/Search.hpp"
#include "../backend/TranspositionTable.hpp"
#include "../backend/ThreadPool.hpp"

#include <iostream>
#include <mutex>
#include <vector>

struct Options
{
    uint32_t multiPV = 1;
    uint32_t threads = 1;
    bool analysisMode = false;
    bool useStandardAlgebraicNotation = false;
};

struct SearchTaskContext
{
    SearchParam searchParam;
    SearchResult searchResult;
    threadpool::Waitable waitable;
    std::atomic<bool> ponderHit = false;

    SearchTaskContext(TranspositionTable& tt) : searchParam{ tt } { }
};

class UniversalChessInterface
{
public:
    UniversalChessInterface(int argc, const char* argv[]);
    void Loop();
    bool ExecuteCommand(const std::string& commandString);

private:
    bool Command_Position(const std::vector<std::string>& args);
    bool Command_Go(const std::vector<std::string>& args);
    bool Command_Stop();
    bool Command_PonderHit();
    bool Command_Perft(const std::vector<std::string>& args);
    bool Command_SetOption(const std::string& name, const std::string& value);
    bool Command_TTProbe();

    void RunSearchTask();

    Game mGame;
    Search mSearch;
    TranspositionTable mTranspositionTable;
    Options mOptions;

    std::mutex mMutex;

    std::unique_ptr<SearchTaskContext> mSearchCtx;
};