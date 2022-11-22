#include "GameCollection.hpp"
#include "../backend/Game.hpp"

namespace GameCollection
{

	bool Reader::ReadGame(Game& game)
	{
		std::vector<MoveAndScore> moves;
		GameHeader header{};

		{
			std::unique_lock<std::mutex> lock(mMutex);

			if (!mStream.Read(&header, sizeof(header)))
			{
				std::cout << "Failed to read games collection stream" << std::endl;
				return false;
			}

			moves.resize(header.numMoves);

			if (!mStream.Read(moves.data(), sizeof(MoveAndScore) * header.numMoves))
			{
				std::cout << "Failed to read games collection stream" << std::endl;
				return false;
			}
		}

		if (header.forcedScore != Game::Score::Unknown &&
			header.forcedScore != Game::Score::WhiteWins &&
			header.forcedScore != Game::Score::BlackWins &&
			header.forcedScore != Game::Score::Draw)
		{
			std::cout << "Failed to parse game from collection: invalid game score" << std::endl;
			return false;
		}

		Position initialPosition;
		if (!UnpackPosition(header.initialPosition, initialPosition))
		{
			return false;
		}
		game.Reset(initialPosition);

		for (uint32_t i = 0; i < header.numMoves; ++i)
		{
			const Move move = game.GetPosition().MoveFromPacked(moves[i].move);

			if (!move.IsValid())
			{
				std::cout
					<< "Failed to parse game from collection: move " << moves[i].move.ToString()
					<< " is invalid in position " << game.GetPosition().ToFEN() << std::endl;
				return false;
			}

#ifndef CONFIGURATION_FINAL
			bool moveSucessful;
			if (header.hasMoveScores)
			{
				moveSucessful = game.DoMove(move, moves[i].score);
			}
			else
			{
				moveSucessful = game.DoMove(move);
			}
			ASSERT(moveSucessful);
#endif // CONFIGURATION_FINAL
		}

		game.SetScore(header.forcedScore);

		return true;
	}

	bool Writer::WriteGame(const Game& game)
	{
		ASSERT(game.GetMoves().size() <= UINT16_MAX);

		GameHeader header{};
		header.forcedScore = game.GetForcedScore();
		header.numMoves = (uint16_t)game.GetMoves().size();
		header.hasMoveScores = game.GetMoves().size() == game.GetMoveScores().size();

		if (!PackPosition(game.GetInitialPosition(), header.initialPosition))
		{
			return false;
		}

		std::vector<MoveAndScore> moves;
		moves.reserve(game.GetMoves().size());

		for (size_t i = 0; i < game.GetMoves().size(); ++i)
		{
			const int16_t moveScore = header.hasMoveScores ? game.GetMoveScores()[i] : 0;
			moves.push_back({ game.GetMoves()[i], moveScore });
		}

		{
			std::unique_lock<std::mutex> lock(mMutex);

			if (!mStream.Write(&header, sizeof(header)))
			{
				std::cout << "Failed to write games collection stream" << std::endl;
				return false;
			}

			if (!mStream.Write(moves.data(), sizeof(MoveAndScore) * game.GetMoves().size()))
			{
				std::cout << "Failed to write games collection stream" << std::endl;
				return false;
			}
		}

		return true;
	}

} // namespace GameCollection
