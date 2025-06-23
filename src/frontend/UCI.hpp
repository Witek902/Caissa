#pragma once

#include "../backend/Game.hpp"
#include "../backend/Search.hpp"
#include "../backend/TranspositionTable.hpp"
#include "../backend/Waitable.hpp"

#include <mutex>
#include <vector>
#include <thread>

struct Options
{
    uint32_t multiPV = 1;
    uint32_t threads = 1;
    int32_t moveOverhead = 10;
    int32_t evalRandomization = 0;
    bool analysisMode = false;
    bool useStandardAlgebraicNotation = false;
    bool colorConsoleOutput = false;
    bool showWDL = false;
};

struct SearchTaskContext
{
    SearchParam searchParam;
    SearchResult searchResult;
    Waitable waitable;
    bool startedAsPondering = false;
    std::atomic<bool> ponderHit = false;
    std::atomic<bool> searchStarted = false;

    SearchTaskContext(TranspositionTable& tt) : searchParam{ tt } { }
};

class UniversalChessInterface
{
public:
    UniversalChessInterface();
    ~UniversalChessInterface();

    void Loop(int argc, const char* argv[]);
    bool ExecuteCommand(const std::string& commandString);

private:
    bool Command_Position(const std::vector<std::string>& args);
    bool Command_Go(const std::vector<std::string>& args);
    bool Command_Stop();
    bool Command_PonderHit();
    bool Command_Perft(const std::vector<std::string>& args);
    bool Command_SetOption(const std::string& name, const std::string& value);
    bool Command_NodeCacheProbe();
    bool Command_TranspositionTableProbe();
    bool Command_TablebaseProbe();
    bool Command_ScoreMoves();
    bool Command_Benchmark();

    void StopSearchThread();
    void DoSearch();

    void SearchThreadEntryFunc();

    Game mGame;
    Search mSearch;
    TranspositionTable mTranspositionTable;
    Options mOptions;

    Position mPrevSearchPosition;
    std::vector<Move> mPrevSearchPvLine;
    bool mIsFirstSearch = true;

    std::thread mSearchThread;

    std::mutex mSearchThreadMutex;
    std::condition_variable mNewSearchConditionVariable;
    std::atomic<bool> mStopSearchThread = false;
    SearchTaskContext* mNewSearchContext = nullptr;

    std::unique_ptr<SearchTaskContext> mSearchCtx;

    std::vector<std::string> mCommandArgs;
};