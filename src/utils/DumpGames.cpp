#include "Common.hpp"
#include "GameCollection.hpp"

#include <filesystem>
#include <fstream>

static bool DumpGames(const std::string& path)
{
    std::vector<Move> moves;

    if (!std::filesystem::exists(path))
    {
        return false;
    }

    FileInputStream gamesFile(path.c_str());
    if (!gamesFile.IsOpen())
    {
        std::cout << "ERROR: Failed to load selfplay data file: " << path << std::endl;
        return false;
    }

    Game game;
    while (GameCollection::ReadGame(gamesFile, game, moves))
    {
        std::cout << game.ToPGN() << std::endl << std::endl;
    }

    return true;
}

void DumpGames(const std::vector<std::string>& args)
{
    for (const auto& path : args)
    {
        DumpGames(path);
    }
}
