#include "MovePicker.hpp"
#include "MoveOrderer.hpp"
#include "Position.hpp"
#include "TranspositionTable.hpp"
#include "Search.hpp"

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

            if (!generateQuiets)
            {
                stage = Stage::End;
                return false;
            }

            stage = Stage::Killer1;
            [[fallthrough]];
        }

        case Stage::Killer1:
        {
            stage = Stage::Killer2;
            const Move move = position.MoveFromPacked(moveOrderer.GetKillerMoves(node.height).moves[0]);
            if (move.IsValid() && !move.IsCapture() && move != pvMove && !ttEntry.moves.HasMove(move))
            {
                outMove = move;
                outScore = MoveOrderer::KillerMoveBonus;
                return true;
            }
            [[fallthrough]];
        }

        case Stage::Killer2:
        {
            stage = Stage::GenerateQuiets;
            const Move move = position.MoveFromPacked(moveOrderer.GetKillerMoves(node.height).moves[1]);
            if (move.IsValid() && !move.IsCapture() && move != pvMove && !ttEntry.moves.HasMove(move))
            {
                outMove = move;
                outScore = MoveOrderer::KillerMoveBonus - 1;
                return true;
            }
            [[fallthrough]];
        }

        case Stage::GenerateQuiets:
        {
            stage = Stage::PickQuiets;
            if (moveGenFlags & MOVE_GEN_MASK_QUIET)
            {
                position.GenerateMoveList(moves, MOVE_GEN_MASK_QUIET);

                // remove PV and TT moves from generated list
                moves.RemoveMove(pvMove);
                for (uint32_t i = 0; i < TTEntry::NumMoves; i++) moves.RemoveMove(ttEntry.moves[i]);

                const auto& killerMoves = moveOrderer.GetKillerMoves(node.height).moves;
                if (killerMoves[0].IsValid()) moves.RemoveMove(killerMoves[0]);
                if (killerMoves[1].IsValid()) moves.RemoveMove(killerMoves[1]);

                moveOrderer.ScoreMoves(node, game, moves, true, nodeCacheEntry);
            }
            [[fallthrough]];
        }

        case Stage::PickQuiets:
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
