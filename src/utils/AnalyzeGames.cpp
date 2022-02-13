#include "Common.hpp"
#include "GameCollection.hpp"

#include "../backend/Position.hpp"
#include "../backend/Material.hpp"
#include "../backend/Game.hpp"
#include "../backend/Move.hpp"
#include "../backend/Search.hpp"
#include "../backend/TranspositionTable.hpp"
#include "../backend/Evaluate.hpp"
#include "../backend/Endgame.hpp"
#include "../backend/Tablebase.hpp"
#include "../backend/NeuralNetwork.hpp"
#include "../backend/Waitable.hpp"

#include <chrono>
#include <random>
#include <mutex>
#include <fstream>
#include <limits.h>

void AnalyzeGames()
{
    FileInputStream gamesFile("selfplay.dat");
    GameCollection::Reader reader(gamesFile);

    uint32_t numGames = 0;

    Game game;
    while (reader.ReadGame(game))
    {
        Position pos = game.GetInitialPosition();

        ASSERT(game.GetMoves().size() == game.GetMoveScores().size());

		for (size_t i = 0; i < game.GetMoves().size(); ++i)
		{
			const Move move = pos.MoveFromPacked(game.GetMoves()[i]);

            if (!pos.IsInCheck(pos.GetSideToMove()) &&
                std::abs(game.GetMoveScores()[i]) < 1600)
            {
                const MaterialKey key = pos.GetMaterialKey();
                if (key.numWhitePawns > 1 && key.numWhiteKnights == 0 && key.numWhiteBishops == 0 && key.numWhiteRooks == 0 && key.numWhiteQueens == 0 &&
                    key.numBlackPawns > 1 && key.numBlackKnights == 0 && key.numBlackBishops == 0 && key.numBlackRooks == 0 && key.numBlackQueens == 0)
                {
                    std::cout << pos.ToFEN() << " score: " << game.GetMoveScores()[i] << std::endl;
                    break;
                }
            }

			pos.DoMove(move);
		}

        numGames++;
    }

    std::cout << "Found " << numGames << " games" << std::endl;
}
