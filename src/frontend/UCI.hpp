#pragma once

#include "../backend/Game.hpp"
#include "../backend/Search.hpp"
#include "../backend/Waitable.hpp"
#include "../backend/containers/String.hpp"

class PlayoutRunner;

using CommandArgs = DynArray<String>;

struct Options
{
    uint32_t multiPV = 1;
    uint32_t threads = 1;
    int32_t moveOverhead = 10;
    int32_t evalRandomization = 0;
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
    bool ExecuteCommand(const char* commandString);

private:
    bool Command_Position(const CommandArgs& args);
    bool Command_Go(const CommandArgs& args);
    bool Command_Stop();
    bool Command_PonderHit();
    bool Command_Perft(const CommandArgs& args);
    bool Command_SetOption(const String& name, const char* value);
    bool Command_NodeCacheProbe();
    bool Command_TranspositionTableProbe();
    bool Command_TablebaseProbe();
    bool Command_ScoreMoves();
    bool Command_EvalDetailed(const CommandArgs& args);
    bool Command_Benchmark(uint32_t depth);
    bool Command_Playout(const CommandArgs& args);

    void StopSearchThread();
    void DoSearch();

    void SearchThreadEntryFunc();

    Game mGame;
    Search mSearch;
    TranspositionTable mTranspositionTable;
    Options mOptions;

    Position mPrevSearchPosition;
    DynArray<Move> mPrevSearchPvLine;

    // search thread with its mutex and condition variable, defined in UCI.cpp
    struct SearchThread;
    SearchThread* mSearchThread = nullptr;

    std::atomic<bool> mStopSearchThread = false;
    SearchTaskContext* mNewSearchContext = nullptr;

    // owned
    SearchTaskContext* mSearchCtx = nullptr;
    PlayoutRunner* mPlayout = nullptr;

    CommandArgs mCommandArgs;
};
