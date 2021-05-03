#pragma once

#include "Position.hpp"
#include "Search.hpp"

#include <iostream>
#include <mutex>
#include <vector>

class UniversalChessInterface
{
public:
    UniversalChessInterface();
    bool Loop();

private:
    bool Command_Position(const std::vector<std::string>& args);
    bool Command_Go(const std::vector<std::string>& args);
    bool Command_Perft(const std::vector<std::string>& args);

    Position mPosition;
    Search mSearch;

    std::mutex mMutex;
};