#pragma once

#include "Position.hpp"
#include "Search.hpp"

#include <iostream>
#include <mutex>
#include <vector>

struct Options
{
    uint32_t multiPV = 1;
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
    bool Command_Perft(const std::vector<std::string>& args);
    bool Command_SetOption(const std::string& name, const std::string& value);

    Position mPosition;
    Search mSearch;
    Options mOptions;

    std::mutex mMutex;
};