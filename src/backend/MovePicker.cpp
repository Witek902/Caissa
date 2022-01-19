#include "MovePicker.hpp"
#include "MoveOrderer.hpp"
#include "Position.hpp"

MovePicker::MovePicker(const Position& pos, const MoveOrderer& moveOrderer, const TTEntry& ttEntry, uint32_t moveGenFlags)
    : position(pos)
    , moveOrderer(moveOrderer)
    , moveGenFlags(0)
{
    position.GenerateMoveList(moves, moveGenFlags);

    // resolve move scoring
    // the idea here is to defer scoring if we have a TT/PV move
    // most likely we'll get beta cutoff on it so we won't need to score any other move
    numScoredMoves = moves.Size();

    if (numScoredMoves > 1u)
    {
        numScoredMoves = moves.AssignTTScores(ttEntry);
    }
}

void MovePicker::Shuffle()
{
    moves.Shuffle();
    shuffleEnabled = true;
}

bool MovePicker::IsOnlyOneLegalMove()
{
    // TODO how to determine this if moves are not generated yet?
    return false;
}

bool MovePicker::PickMove(const NodeInfo& node, Move& outMove, int32_t& outScore)
{
    if (moveIndex >= moves.Size())
    {
        return false;
    }

    if (moveIndex == numScoredMoves && !shuffleEnabled)
    {
        // we reached a point where moves are not scored anymore, so score them now
        moveOrderer.ScoreMoves(node, moves);
    }

    outMove = moves.PickBestMove(moveIndex++, outScore);
    return true;
}
