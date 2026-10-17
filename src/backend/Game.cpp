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
    mMoves.Clear();
    mMoveScores.Clear();
    for (auto& history : mHistoryGamePositions) history.Clear();

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
        mMoves.PushBack(move);

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
        mMoves.PushBack(move);
        mMoveScores.PushBack(score);

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
        if (iter.position == position)
        {
            iter.count++;
            return;
        }
    }
    historyPositions.PushBack({ position, 1 });
}

uint32_t Game::GetRepetitionCount(const Position& position) const
{
    const auto& historyPositions = mHistoryGamePositions[position.GetHash() % GameHistoryBuckets];
    for (const auto& iter : historyPositions)
    {
        if (iter.position == position)
            return iter.count;
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

String Game::ToPGNMoveList(bool includeScores) const
{
    String str;

    Position pos = mInitPosition;

    for (uint32_t i = 0; i < mMoves.Size(); ++i)
    {
        const bool isWhiteMove = pos.GetSideToMove() == White;

        if (isWhiteMove)
        {
            // White moves always get a move number indication: "N. "
            str.AppendFormat("%u. ", (uint32_t)pos.GetMoveCount());
        }
        else if (i == 0 || includeScores)
        {
            // Black move number indication required when:
            // - game starts with black to move (i == 0, no preceding white move)
            // - commentary preceded this black move (every white move has {score})
            str.AppendFormat("%u... ", (uint32_t)pos.GetMoveCount());
        }

        str.Append(pos.MoveToString(mMoves[i]));
        str.Append(' ');

        if (includeScores && i < mMoveScores.Size())
        {
            // Scores are stored from white's perspective; negate for black's moves
            // so the comment reflects the side-to-move's perspective.
            // Format: {score/0} — fishtest/scoreWDLstat-compatible
            const ScoreType score = isWhiteMove ? mMoveScores[i] : ScoreType(-mMoveScores[i]);
            str.AppendFormat("{%s/0} ", ScoreToStr(score).CStr());
        }

        const bool moveResult = pos.DoMove(mMoves[i]);
        ASSERT(moveResult);
        (void)moveResult;
    }

    ASSERT(pos == mPosition);

    return str;
}

String Game::ToPGN(bool includeScores) const
{
    String str;

    const char* resultStr = "";
    const char* terminationStr = "";

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

    const bool isNonStandardStart = mInitPosition.ToFEN() != Position::InitPositionFEN;

    str.Append("[Event \"?\"]\n");
    str.Append("[Site \"?\"]\n");
    str.Append("[Date \"????.??.??\"]\n");
    str.AppendFormat("[Round \"%u\"]\n", mMetadata.roundNumber);
    str.Append("[White \"Caissa\"]\n");
    str.Append("[Black \"Caissa\"]\n");
    str.AppendFormat("[Result \"%s\"]\n", resultStr);
    // Supplemental tags in ASCII order (§8.1.1): FEN(F=70) < SetUp(S=83) < Termination(T=84)
    if (isNonStandardStart)
    {
        str.AppendFormat("[FEN \"%s\"]\n", mInitPosition.ToFEN().CStr());
        str.Append("[SetUp \"1\"]\n");
    }
    str.AppendFormat("[Termination \"%s\"]\n", terminationStr);
    str.Append('\n');
    str.Append(ToPGNMoveList(includeScores));
    str.Append(resultStr);

    return str;
}