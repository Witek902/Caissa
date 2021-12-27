#pragma once

#include <string>

class Game;

class GameCollection
{
public:
	struct Header
	{
		uint64_t numGames;
	};

	struct GameHeader
	{
		uint16_t whiteElo;
		uint16_t blackElo;
		uint16_t numMoves;
	};

	bool LoadPGN(const std::string& path);
	void LoadBIN(const std::string& path);

	void AddGame(const Game& game);
};