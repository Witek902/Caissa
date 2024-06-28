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
    mMoveScores.clear();
    for (auto& history : mHistoryGamePositions) history.clear();

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
    auto& historyPositions = mHistoryGamePositions[position.GetHash() % GameHistoryBuckets];
    for (auto& iter : historyPositions)
    {
        if (iter.first == position)
        {
            iter.second++;
            return;
        }
    }
    historyPositions.emplace_back(position, 1);
}

uint32_t Game::GetRepetitionCount(const Position& position) const
{
    const auto& historyPositions = mHistoryGamePositions[position.GetHash() % GameHistoryBuckets];
    for (const auto& iter : historyPositions)
    {
        if (iter.first == position)
            return iter.second;
    }
    return 0;
}

Game::Score Game::CalculateScore() const
{
    if (mPosition.IsMate())
    {
        return mPosition.GetSideToMove() == White ? Score::BlackWins : Score::WhiteWins;
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

    if (mPosition.IsFiftyMoveRuleDraw())
    {
        return true;
    }

    if (CheckInsufficientMaterial(mPosition))
    {
        return true;
    }

    if (mPosition.IsStalemate())
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

std::string Game::ToPGNMoveList(bool includeScores) const
{
    std::stringstream str;

    Position pos = mInitPosition;

    for (size_t i = 0; i < mMoves.size(); ++i)
    {
        if (i % 2 == 0)
        {
            str << pos.GetMoveCount() << ". ";
        }

        str << pos.MoveToString(mMoves[i]) << ' ';

        if (includeScores && i < mMoveScores.size())
        {
            str << '{';
            str << ScoreToStr(mMoveScores[i]);
            str << "} ";
        }

        const bool moveResult = pos.DoMove(mMoves[i]);
        ASSERT(moveResult);
        (void)moveResult;
    }

    ASSERT(pos == mPosition);

    return str.str();
}

std::string Game::ToPGN(bool includeScores) const
{
    std::stringstream str;

    std::string resultStr;
    std::string terminationStr;

    const Game::Score gameScore = GetScore();
    if (gameScore == Game::Score::WhiteWins)
    {
        resultStr = "1-0";
        terminationStr = "checkmate";
    }
    else if (gameScore == Game::Score::BlackWins)
    {
        resultStr = "0-1";
        terminationStr = "checkmate";
    }
    else if (gameScore == Game::Score::Draw)
    {
        resultStr = "1/2-1/2";
        if (GetRepetitionCount(GetPosition()) >= 2) terminationStr = "3-fold repetition";
        else if (GetPosition().IsFiftyMoveRuleDraw()) terminationStr = "50 moves rule";
        else if (CheckInsufficientMaterial(GetPosition())) terminationStr = "insufficient material";
        else terminationStr = "unknown";
    }

    if (mForcedScore != Score::Unknown)
    {
        terminationStr = "adjudication";
    }

    str << "[Round \"1." << mMetadata.roundNumber << "\"]" << std::endl;
    str << "[White \"Caissa\"]" << std::endl;
    str << "[Black \"Caissa\"]" << std::endl;
    str << "[Result \"" << resultStr << "\"]" << std::endl;
    str << "[Termination \"" << terminationStr << "\"]" << std::endl;
    str << "[FEN \"" << mInitPosition.ToFEN() << "\"]" << std::endl;
    str << std::endl;
    str << ToPGNMoveList(includeScores) << resultStr;

    return str.str();
}