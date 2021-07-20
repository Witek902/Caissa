#include "Game.hpp"
#include "Evaluate.hpp"

void Game::Reset(const Position& pos)
{
    mInitPosition = pos;
    mPosition = pos;
    mMoves.clear();
    mHistoryGamePositions.clear();

    RecordBoardPosition(pos);
}

bool Game::DoMove(const Move& move)
{
    if (mPosition.DoMove(move))
    {
        mMoves.push_back(move);

        RecordBoardPosition(mPosition);

        return true;
    }

    return false;
}

void Game::RecordBoardPosition(const Position& position)
{
    HistoryPositions& entry = mHistoryGamePositions[position.GetHash()];

    for (HistoryPosition& historyPosition : entry)
    {
        if (historyPosition.first == position)
        {
            historyPosition.second++;
            return;
        }
    }

    entry.emplace_back(position, 1u);
}

uint32_t Game::GetRepetitionCount(const Position& position) const
{
    const auto iter = mHistoryGamePositions.find(position.GetHash());
    if (iter == mHistoryGamePositions.end())
    {
        return false;
    }

    const HistoryPositions& entry = iter->second;

    for (const HistoryPosition& historyPosition : entry)
    {
        if (historyPosition.first == position)
        {
            return historyPosition.second;
        }
    }

    return 0;
}

bool Game::IsDrawn() const
{
    // NOTE: two-fold repetition rule is enough
    if (GetRepetitionCount(mPosition) >= 2)
    {
        return true;
    }

    if (mPosition.GetHalfMoveCount() >= 100)
    {
        return true;
    }

    if (CheckInsufficientMaterial(mPosition))
    {
        return true;
    }

    return false;
}

std::string Game::ToPGN() const
{
    std::string str;

    Position pos = mInitPosition;

    for (size_t i = 0; i < mMoves.size(); ++i)
    {
        if (i % 2 == 0)
        {
            str += std::to_string(1 + (i / 2));
            str += ". ";
        }

        str += pos.MoveToString(mMoves[i]);
        str += ' ';

        const bool moveResult = pos.DoMove(mMoves[i]);
        ASSERT(moveResult);
    }

    ASSERT(pos == mPosition);

    return str;
}