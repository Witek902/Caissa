#pragma once

#include <string>

class Game;

struct PGNGame
{

};

class PGNParser
{
public:
    PGNParser(std::istream& stream);

    bool GetGame(PGNGame& outGame);

private:
    std::istream& mStream;

    bool ParseTags(PGNGame& outGame);
    bool ParseMoves(PGNGame& outGame);

    void SkipBlank();
};