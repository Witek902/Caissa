#include "MovePicker.hpp"
#include "MoveOrderer.hpp"
#include "Position.hpp"
#include "TranspositionTable.hpp"

bool MovePicker::PickMove(const NodeInfo& node, const Game& game, Move& outMove, int32_t& outScore)
{
    const bool generateQuiets = moveGenFlags & MOVE_GEN_MASK_QUIET;

    switch (stage)
    {
        case Stage::PVMove:
        {
            stage = Stage::TTMove;
            if (pvMove.IsValid() && (!pvMove.IsQuiet() || generateQuiets))
            {
                outMove = pvMove;
                outScore = MoveOrderer::PVMoveValue;
                return true;
            }

            [[fallthrough]];
        }

        case Stage::TTMove:
        {
            for (; moveIndex < TTEntry::NumMoves; moveIndex++)
            {
                const Move move = position.MoveFromPacked(ttEntry.moves[moveIndex]);
                if (move.IsValid() && (!move.IsQuiet() || generateQuiets) && move != pvMove)
                {
                    outMove = move;
                    outScore = MoveOrderer::TTMoveValue - moveIndex;
                    moveIndex++;
                    return true;
                }
            }

            // TT move not found - go to next stage
            stage = Stage::Captures;
            moveIndex = 0;
            position.GenerateMoveList(moves, moveGenFlags & (MOVE_GEN_MASK_CAPTURES | MOVE_GEN_MASK_PROMOTIONS));

            // remove PV and TT moves from generated list
            moves.RemoveMove(pvMove);
            for (uint32_t i = 0; i < TTEntry::NumMoves; i++) moves.RemoveMove(ttEntry.moves[i]);

            moveOrderer.ScoreMoves(node, game, moves, false);

            [[fallthrough]];
        }

        case Stage::Captures:
        {
            if (moves.Size() > 0)
            {
                const uint32_t index = moves.BestMoveIndex();
                outMove = moves.GetMove(index);
                outScore = moves.GetScore(index);

                ASSERT(outMove.IsValid());
                ASSERT(outScore > INT32_MIN);

                if (outScore >= MoveOrderer::PromotionValue)
                {
                    moves.RemoveByIndex(index);
                    return true;
                }
            }

            stage = Stage::Quiet;
            moveIndex = 0;

            if (moveGenFlags & MOVE_GEN_MASK_QUIET)
            {
                position.GenerateMoveList(moves, MOVE_GEN_MASK_QUIET);

                // remove PV and TT moves from generated list
                moves.RemoveMove(pvMove);
                for (uint32_t i = 0; i < TTEntry::NumMoves; i++) moves.RemoveMove(ttEntry.moves[i]);

                moveOrderer.ScoreMoves(node, game, moves, true, nodeCacheEntry);
            }

            [[fallthrough]];
        }

        case Stage::Quiet:
        {
            if (moves.Size() > 0)
            {
                const uint32_t index = moves.BestMoveIndex();
                outMove = moves.GetMove(index);
                outScore = moves.GetScore(index);

                ASSERT(outMove.IsValid());
                ASSERT(outScore > INT32_MIN);

                moves.RemoveByIndex(index);

                return true;
            }

            stage = Stage::End;

            break;
        }
    }

    return false;
}
