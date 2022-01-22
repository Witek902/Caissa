#pragma once

#include "../backend/Game.hpp"
#include "../backend/Search.hpp"
#include "../backend/TranspositionTable.hpp"
#include "../backend/Waitable.hpp"

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
    Waitable waitable;
    std::atomic<bool> ponderHit = false;

    SearchTaskContext(TranspositionTable& tt) : searchParam{ tt } { }
};

class UniversalChessInterface
{
public:
    UniversalChessInterface(int argc, const char* argv[]);
    ~UniversalChessInterface();

    void Loop();
    bool ExecuteCommand(const std::string& commandString);

private:
    bool Command_Position(const std::vector<std::string>& args);
    bool Command_Go(const std::vector<std::string>& args);
    bool Command_Stop();
    bool Command_PonderHit();
    bool Command_Perft(const std::vector<std::string>& args);
    bool Command_SetOption(const std::string& name, const std::string& value);
    bool Command_TranspositionTableProbe();
    bool Command_TablebaseProbe();
    bool Command_ScoreMoves();

    void StopSearchThread();
    void RunSearchTask();
    void DoSearch();

    void SearchThreadEntryFunc();

    Game mGame;
    Search mSearch;
    TranspositionTable mTranspositionTable;
    Options mOptions;

    std::thread mSearchThread;

    std::mutex mNewSearchMutex;
    std::condition_variable mNewSearchConditionVariable;
    std::atomic<bool> mStopSearchThread = false;
    SearchTaskContext* mNewSearchContext = nullptr;

    std::unique_ptr<SearchTaskContext> mSearchCtx;
};