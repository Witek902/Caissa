#include "Game.hpp"
#include "Evaluate.hpp"

Game::Game()
    : mInitPosition(Position::InitPositionFEN)
    , mPosition(Position::InitPositionFEN)
    , mForcedScore(Score::Unknown)
{}

void Game::Reset(const Position& pos)
{
    mInitPosition = pos;
    mPosition = pos;
    mForcedScore = Score::Unknown;
    mMoves.clear();
    mHistoryGamePositions.clear();

    RecordBoardPosition(pos);
}

void Game::SetScore(Score score)
{
    mForcedScore = score;
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

bool Game::DoMove(const Move& move, ScoreType score)
{
    ASSERT(mForcedScore == Score::Unknown);

    if (mPosition.DoMove(move))
    {
        mMoves.push_back(move);
        mMoveScores.push_back(score);

        RecordBoardPosition(mPosition);

        return true;
    }

    return false;
}

void Game::RecordBoardPosition(const Position& position)
{
    mHistoryGamePositions[position]++;
}

uint32_t Game::GetRepetitionCount(const Position& position) const
{
    const auto& iter = mHistoryGamePositions.find(position);
    if (iter == mHistoryGamePositions.end())
    {
        return 0;
    }

    return iter->second;
}

Game::Score Game::CalculateScore() const
{
    if (mPosition.IsMate())
    {
        return mPosition.GetSideToMove() == Color::White ? Score::BlackWins : Score::WhiteWins;
    }

    if (IsDrawn())
    {
        return Score::Draw;
    }

    return Score::Unknown;
}

Game::Score Game::GetScore() const
{
    if (mForcedScore != Score::Unknown)
    {
        return mForcedScore;
    }

    return CalculateScore();
}

bool Game::IsDrawn() const
{
    if (GetRepetitionCount(mPosition) >= 3)
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

bool Game::operator == (const Game& rhs) const
{
    return
        mInitPosition == rhs.mInitPosition &&
        mPosition == rhs.mPosition &&
        mForcedScore == rhs.mForcedScore &&
        mMoves == rhs.mMoves &&
        mMoveScores == rhs.mMoveScores;
}

bool Game::operator != (const Game& rhs) const
{
    return
        mInitPosition != rhs.mInitPosition ||
        mPosition != rhs.mPosition ||
        mForcedScore != rhs.mForcedScore ||
        mMoves != rhs.mMoves ||
        mMoveScores != rhs.mMoveScores;
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
        (void)moveResult;
    }

    ASSERT(pos == mPosition);

    return str;
}