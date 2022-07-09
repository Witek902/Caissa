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
#include <iomanip>

struct MaterialConfigInfo
{
    uint64_t occurences = 0;
    double evalScore = 0.0;
    double gameScore = 0.0;
};

float GameScoreToWinProbability(const Game::Score score)
{
    switch (score)
    {
    case Game::Score::BlackWins: return 0.0f;
    case Game::Score::WhiteWins: return 1.0f;
    default: return 0.5f;
    }
}

void AnalyzeGames()
{
    FileInputStream gamesFile("../../data/selfplayGames/selfplay4.dat");
    //FileInputStream gamesFile("selfplay5.dat");
    GameCollection::Reader reader(gamesFile);

    uint64_t numGames = 0;
    uint64_t numPositions = 0;

    std::unordered_map<MaterialKey, MaterialConfigInfo> materialConfigurations;
    
    uint64_t whiteKingOccupancy[64] = { 0 };
    uint64_t blackKingOccupancy[64] = { 0 };

    Game game;
    while (reader.ReadGame(game))
    {
        Position pos = game.GetInitialPosition();

        ASSERT(game.GetMoves().size() == game.GetMoveScores().size());

        if (game.GetScore() == Game::Score::Unknown)
        {
            continue;
        }

        //std::cout << game.ToPGN(true);
        //std::cout << std::endl << std::endl;

		for (size_t i = 0; i < game.GetMoves().size(); ++i)
		{
			const Move move = pos.MoveFromPacked(game.GetMoves()[i]);
            const ScoreType moveScore = game.GetMoveScores()[i];

            if (!pos.IsInCheck(pos.GetSideToMove()) && !move.IsCapture() && !move.IsPromotion() &&
                std::abs(game.GetMoveScores()[i]) < KnownWinValue)
            {
                numPositions++;

                whiteKingOccupancy[FirstBitSet(pos.Whites().king)]++;
                blackKingOccupancy[FirstBitSet(pos.Blacks().king)]++;

                MaterialConfigInfo& matConfigInfo = materialConfigurations[pos.GetMaterialKey()];
                matConfigInfo.occurences++;
                matConfigInfo.evalScore += CentiPawnToWinProbability(moveScore);
                matConfigInfo.gameScore += GameScoreToWinProbability(game.GetScore());

                //const MaterialKey key = pos.GetMaterialKey();
                //if (key.numWhitePawns > 1 && key.numWhiteKnights == 0 && key.numWhiteBishops == 0 && key.numWhiteRooks == 0 && key.numWhiteQueens == 0 &&
                //    key.numBlackPawns > 1 && key.numBlackKnights == 0 && key.numBlackBishops == 0 && key.numBlackRooks == 0 && key.numBlackQueens == 0)
                //{
                //    std::cout << pos.ToFEN() << " score: " << game.GetMoveScores()[i] << std::endl;
                //    break;
                //}
            }

			pos.DoMove(move);
		}

        numGames++;
    }

    std::cout << "Found " << numGames << " games" << std::endl;

    {
        std::cout << "Unique material configurations: " << materialConfigurations.size() << std::endl;
        for (const auto& iter : materialConfigurations)
        {
            if (iter.second.occurences > 10000)
            {
                const float averageEvalScore = static_cast<float>(iter.second.evalScore / static_cast<double>(iter.second.occurences));
                const float averageGameScore = static_cast<float>(iter.second.gameScore / static_cast<double>(iter.second.occurences));

                std::cout
                    << std::setw(33) << iter.first.ToString() << " "
                    << std::showpos << std::fixed << std::setprecision(2) << WinProbabilityToPawns(averageEvalScore) << " "
                    << std::showpos << std::fixed << std::setprecision(2) << WinProbabilityToPawns(averageGameScore) << std::endl;
                std::cout << std::resetiosflags(std::ios_base::showpos);
            }
        }
        std::cout << std::endl;
    }

    {
        std::cout << "White king occupancy: " << std::endl;
        for (uint32_t rank = 0; rank < 8; ++rank)
        {
            for (uint32_t file = 0; file < 8; ++file)
            {
                std::cout << std::fixed << std::setprecision(4) << ((float)whiteKingOccupancy[8 * rank + file] / (float)numPositions) << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    {
        std::cout << "Black king occupancy: " << std::endl;
        for (uint32_t rank = 0; rank < 8; ++rank)
        {
            for (uint32_t file = 0; file < 8; ++file)
            {
                std::cout << std::fixed << std::setprecision(4) << ((float)blackKingOccupancy[8 * rank + file] / (float)numPositions) << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
}
