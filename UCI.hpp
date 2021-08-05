#pragma once

#include "Game.hpp"
#include "Search.hpp"
#include "ThreadPool.hpp"

#include <iostream>
#include <mutex>
#include <vector>

struct Options
{
    uint32_t multiPV = 1;
    uint32_t threads = 1;
};

struct SearchTaskContext
{
    SearchParam searchParam;
    SearchResult searchResult;
    threadpool::Waitable waitable;
    std::atomic<bool> ponderHit = false;
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
    Options mOptions;

    std::mutex mMutex;

    std::unique_ptr<SearchTaskContext> mSearchCtx;
};