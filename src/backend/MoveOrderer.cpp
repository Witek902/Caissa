#include "MoveOrderer.hpp"
#include "Search.hpp"
#include "Tuning.hpp"

DEFINE_PARAM(QuietMoveHistoryClear, 628, -2000, 2000);
DEFINE_PARAM(ContinuationHistoryClear, 736, -2000, 2000);
DEFINE_PARAM(CapturesHistoryClear, 322, -2000, 2000);

DEFINE_PARAM(QuietBonusOffset, -113, -200, 0);
DEFINE_PARAM(QuietBonusLinear, 159, 100, 250);
DEFINE_PARAM(QuietBonusScoreDiff, 153, 0, 400);
DEFINE_PARAM(QuietBonusLimit, 2090, 1000, 4000);

DEFINE_PARAM(QuietMalusOffset, -51, -200, 50);
DEFINE_PARAM(QuietMalusLinear, 160, 75, 200);
DEFINE_PARAM(QuietMalusScoreDiff, 167, 0, 400);
DEFINE_PARAM(QuietMalusLimit, 1912, 1000, 4000);

DEFINE_PARAM(CaptureBonusOffset, 27, 0, 100);
DEFINE_PARAM(CaptureBonusLinear, 71, 20, 120);
DEFINE_PARAM(CaptureBonusLimit, 2543, 1000, 4000);

DEFINE_PARAM(CaptureMalusOffset, 25, 0, 100);
DEFINE_PARAM(CaptureMalusLinear, 48, 20, 120);
DEFINE_PARAM(CaptureMalusLimit, 1793, 1000, 4000);

MoveOrderer::MoveOrderer()
{
    Clear();
}

void MoveOrderer::DebugPrint() const
{
#ifndef CONFIGURATION_FINAL
    std::cout << "=== QUIET MOVES HISTORY HEURISTICS ===" << std::endl;
    /*
    for (uint32_t fromIndex = 0; fromIndex < 64; ++fromIndex)
    {
        for (uint32_t toIndex = 0; toIndex < 64; ++toIndex)
        {
            for (uint32_t color = 0; color < 2; ++color)
            {
                const CounterType count = quietMoveHistory[color][fromIndex][toIndex];

                if (count)
                {
                    std::cout
                        << Square(fromIndex).ToString() << Square(toIndex).ToString()
                        << (color > 0 ? " (black)" : " (white)")
                        << " ==> " << count << '\n';
                }
            }
        }
    }*/

    std::cout << "=== QUIET MOVES CONTINUATION HISTORY HEURISTICS ===" << std::endl;

    for (uint32_t prevPiece = 0; prevPiece < 6; ++prevPiece)
    {
        for (uint32_t prevToIndex = 0; prevToIndex < 64; ++prevToIndex)
        {
            for (uint32_t piece = 0; piece < 6; ++piece)
            {
                for (uint32_t toIndex = 0; toIndex < 64; ++toIndex)
                {
                    // TODO color
                    const CounterType count = continuationHistory[0][0][0][prevPiece][prevToIndex][piece][toIndex];

                    if (count)
                    {
                        std::cout
                            << PieceToChar(Piece(prevPiece + 1)) << Square(prevToIndex).ToString()
                            << ", "
                            << PieceToChar(Piece(piece + 1)) << Square(toIndex).ToString()
                            << " ==> " << count << '\n';
                    }
                }
            }
        }
    }

    std::cout << std::endl;
    std::cout << "=== KILLER MOVE HEURISTICS ===" << std::endl;
    {
        uint32_t lastValidDepth = 0;
        for (uint32_t d = 0; d < MaxSearchDepth; ++d)
        {
            if (killerMoves[d].IsValid())
            {
                lastValidDepth = std::max(lastValidDepth, d);
            }
        }

        for (uint32_t d = 0; d < lastValidDepth; ++d)
        {
            std::cout << d;
            std::cout << "\t" << killerMoves[d].ToString() << " ";
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    std::cout << std::endl;
    std::cout << "=== CAPTURE HISTORY ===" << std::endl;

    for (uint32_t piece = 0; piece < 6; ++piece)
    {
        for (uint32_t capturedPiece = 0; capturedPiece < 5; ++capturedPiece)
        {
            std::cout << PieceToChar(Piece(piece + 1)) << 'x' << PieceToChar(Piece(capturedPiece + 1)) << std::endl;

            for (uint32_t rank = 0; rank < 8; ++rank)
            {
                for (uint32_t file = 0; file < 8; ++file)
                {
                    const uint32_t square = 8 * (7 - rank) + file;
                    const CounterType count = capturesHistory[0][piece][capturedPiece][square];
                    std::cout << std::fixed << std::setw(8) << count;
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
    }

    std::cout << std::endl;

#endif // CONFIGURATION_FINAL
}

void MoveOrderer::InitContinuationHistoryPointers(NodeInfo& node)
{
    const uint32_t color = (uint32_t)node.position.GetSideToMove();
    const NodeInfo* nodePtr = &node;
    for (uint32_t i = 0; i < 6; ++i)
    {
        if (!nodePtr || nodePtr->ply == 0)
            break;
        if (nodePtr->previousMove.IsValid())
        {
            const uint32_t prevIsCapture = (uint32_t)nodePtr->previousMove.IsCapture();
            const uint32_t prevPiece = (uint32_t)nodePtr->previousMove.GetPiece() - 1;
            const uint32_t prevTo = nodePtr->previousMove.ToSquare().Index();
            const uint32_t prevColor = (uint32_t)(nodePtr - 1)->position.GetSideToMove();
            node.continuationHistories[i] = &(continuationHistory[prevIsCapture][color][prevColor][prevPiece][prevTo]);
        }
        --nodePtr;
    }
}

void MoveOrderer::NewSearch()
{
    const CounterType scaleDownFactor = 2;

    for (uint32_t i = 0; i < sizeof(quietMoveHistory) / sizeof(CounterType); ++i)
        reinterpret_cast<CounterType*>(quietMoveHistory)[i] /= scaleDownFactor;

    for (uint32_t i = 0; i < sizeof(capturesHistory) / sizeof(CounterType); ++i)
        reinterpret_cast<CounterType*>(capturesHistory)[i] /= scaleDownFactor;

    memset(killerMoves, 0, sizeof(killerMoves));
}

static void ClearHistoryTable(MoveOrderer::CounterType* table, MoveOrderer::CounterType value, size_t size)
{
    for (size_t i = 0; i < size; ++i)
    {
        table[i] = value;
    }
}

void MoveOrderer::Clear()
{
    // clear history tables with a value
    ClearHistoryTable(reinterpret_cast<CounterType*>(quietMoveHistory),
        static_cast<CounterType>(QuietMoveHistoryClear), sizeof(quietMoveHistory) / sizeof(CounterType));
    ClearHistoryTable(reinterpret_cast<CounterType*>(continuationHistory),
        static_cast<CounterType>(ContinuationHistoryClear), sizeof(continuationHistory) / sizeof(CounterType));
    ClearHistoryTable(reinterpret_cast<CounterType*>(capturesHistory),
        static_cast<CounterType>(CapturesHistoryClear), sizeof(capturesHistory) / sizeof(CounterType));

    memset(counterMoves, 0, sizeof(counterMoves));
    memset(killerMoves, 0, sizeof(killerMoves));
}

MoveOrderer::CounterType MoveOrderer::GetHistoryScore(const NodeInfo& node, const Move move) const
{
    ASSERT(move.IsValid());
    const Bitboard threats = node.threats.allThreats;
    const uint32_t from = move.FromSquare().Index();
    const uint32_t to = move.ToSquare().Index();
    ASSERT(from < 64);
    ASSERT(to < 64);
    return quietMoveHistory[(uint32_t)node.position.GetSideToMove()][threats.IsBitSet(from)][threats.IsBitSet(to)][move.FromTo()];
}

Move MoveOrderer::GetCounterMove(const NodeInfo& node) const
{
    if (node.previousMove.IsValid())
    {
        const Move prevMove = node.previousMove;
        const uint32_t piece = (uint32_t)prevMove.GetPiece() - 1;
        const uint32_t to = prevMove.ToSquare().Index();
        return counterMoves[(uint32_t)node.position.GetSideToMove()][piece][to];
    }
    return Move::Invalid();
}

INLINE static void UpdateHistoryCounter(MoveOrderer::CounterType& counter, int32_t delta)
{
    int32_t newValue = (int32_t)counter + delta - (int32_t)counter * std::abs(delta) / 16384;

    // there should be no saturation
    ASSERT(newValue > std::numeric_limits<MoveOrderer::CounterType>::min());
    ASSERT(newValue < std::numeric_limits<MoveOrderer::CounterType>::max());

    counter = static_cast<MoveOrderer::CounterType>(newValue);
}

void MoveOrderer::UpdateQuietMovesHistory(const NodeInfo& node, const Move* moves, uint32_t numMoves, const Move bestMove, int32_t scoreDiff)
{
    ASSERT(node.depth >= 0);
    ASSERT(numMoves > 0);
    ASSERT(moves[0].IsQuiet());

    const uint32_t color = (uint32_t)node.position.GetSideToMove();

    // update counter move
    if (bestMove.IsQuiet() && node.previousMove.IsValid())
    {
        const Move prevMove = node.previousMove;
        const uint32_t piece = (uint32_t)prevMove.GetPiece() - 1;
        const uint32_t to = prevMove.ToSquare().Index();
        counterMoves[color][piece][to] = bestMove;
    }

    // don't update uncertain moves
    if (numMoves <= 1 && node.depth < 2)
    {
        return;
    }

    const int32_t bonus = std::min<int32_t>(QuietBonusOffset + QuietBonusLinear * node.depth + QuietBonusScoreDiff * scoreDiff / 64, QuietBonusLimit);
    const int32_t malus = -std::min<int32_t>(QuietMalusOffset + QuietMalusLinear * node.depth + QuietMalusScoreDiff * scoreDiff / 64, QuietMalusLimit);

    const Bitboard threats = node.threats.allThreats;

    for (uint32_t i = 0; i < numMoves; ++i)
    {
        const Move move = moves[i];
        const int32_t delta = move == bestMove ? bonus : malus;

        const uint32_t piece = (uint32_t)move.GetPiece() - 1;
        const uint32_t from = move.FromSquare().Index();
        const uint32_t to = move.ToSquare().Index();
        
        UpdateHistoryCounter(quietMoveHistory[color][threats.IsBitSet(from)][threats.IsBitSet(to)][move.FromTo()], delta);

        if (PieceSquareHistory* h = node.continuationHistories[0]) UpdateHistoryCounter((*h)[piece][to], delta);
        if (PieceSquareHistory* h = node.continuationHistories[1]) UpdateHistoryCounter((*h)[piece][to], delta);
        if (PieceSquareHistory* h = node.continuationHistories[2]) UpdateHistoryCounter((*h)[piece][to], delta / 4);
        if (PieceSquareHistory* h = node.continuationHistories[3]) UpdateHistoryCounter((*h)[piece][to], delta);
        if (PieceSquareHistory* h = node.continuationHistories[5]) UpdateHistoryCounter((*h)[piece][to], delta);
    }
}

void MoveOrderer::UpdateCapturesHistory(const NodeInfo& node, const Move* moves, uint32_t numMoves, const Move bestMove)
{
    // depth can be negative in QSearch
    int32_t depth = std::max<int32_t>(0, node.depth);

    // don't update uncertain moves
    if (numMoves <= 1)
    {
        return;
    }

    const uint32_t color = (uint32_t)node.position.GetSideToMove();

    const int32_t bonus = std::min<int32_t>(CaptureBonusOffset + CaptureBonusLinear * depth, CaptureBonusLimit);
    const int32_t malus = -std::min<int32_t>(CaptureMalusOffset + CaptureMalusLinear * depth, CaptureMalusLimit);

    for (uint32_t i = 0; i < numMoves; ++i)
    {
        const Move move = moves[i];
        ASSERT(move.IsCapture());

        const int32_t delta = move == bestMove ? bonus : malus;

        const Piece captured = node.position.GetCapturedPiece(move);
        ASSERT(captured > Piece::None);
        ASSERT(captured < Piece::King);

        const uint32_t capturedIdx = (uint32_t)captured - 1;
        const uint32_t pieceIdx = (uint32_t)move.GetPiece() - 1;

        ASSERT(pieceIdx < 6);
        ASSERT(capturedIdx < 5);
        UpdateHistoryCounter(capturesHistory[color][pieceIdx][capturedIdx][move.ToSquare().Index()], delta);
    }
}

void MoveOrderer::ScoreMoves(
    const NodeInfo& node,
    MoveList& moves,
    bool withQuiets,
    const NodeCacheEntry* nodeCacheEntry) const
{
    const Position& pos = node.position;

    const uint32_t color = (uint32_t)pos.GetSideToMove();
    const Bitboard threats = node.threats.allThreats;

    for (uint32_t i = 0; i < moves.Size(); ++i)
    {
        const Move move = moves.GetMove(i);
        ASSERT(move.IsValid());

        const uint32_t piece = (uint32_t)move.GetPiece() - 1;
        const uint32_t from = move.FromSquare().Index();
        const uint32_t to = move.ToSquare().Index();

        ASSERT(piece < 6);
        ASSERT(from < 64);
        ASSERT(to < 64);

        // skip moves that has been scored
        if (moves.GetScore(i) > INT32_MIN) continue;

        int32_t score = 0;

        if (move.IsCapture())
        {
            const Piece attackingPiece = move.GetPiece();
            const Piece capturedPiece = pos.GetCapturedPiece(move);
            ASSERT(capturedPiece > Piece::None);
            ASSERT(capturedPiece < Piece::King);

            if ((uint32_t)attackingPiece < (uint32_t)capturedPiece)     score = WinningCaptureValue;
            else if (attackingPiece == capturedPiece)                   score = GoodCaptureValue;
            else if (pos.StaticExchangeEvaluation(move))                score = GoodCaptureValue;
            else                                                        score = LosingCaptureValue;

            // most valuable victim first
            score += 6 * (int32_t)capturedPiece * UINT16_MAX / 128;

            // capture history
            {
                const uint32_t capturedIdx = (uint32_t)capturedPiece - 1;
                const uint32_t pieceIdx = (uint32_t)attackingPiece - 1;
                ASSERT(capturedIdx < 5);
                ASSERT(pieceIdx < 6);
                const int32_t historyScore = ((int32_t)capturesHistory[color][pieceIdx][capturedIdx][move.ToSquare().Index()] - INT16_MIN) / 128;
                ASSERT(historyScore >= 0);
                score += historyScore;
            }
        }
        else if (withQuiets) // non-capture
        {
            // killer moves should be filtered by move picker
            ASSERT(killerMoves[node.ply] != move);

            // history heuristics
            score += quietMoveHistory[color][threats.IsBitSet(from)][threats.IsBitSet(to)][move.FromTo()];

            // continuation history
            if (const PieceSquareHistory* h = node.continuationHistories[0]) score += (*h)[piece][to];
            if (const PieceSquareHistory* h = node.continuationHistories[1]) score += (*h)[piece][to];
            if (const PieceSquareHistory* h = node.continuationHistories[3]) score += (*h)[piece][to] / 2;
            if (const PieceSquareHistory* h = node.continuationHistories[5]) score += (*h)[piece][to] / 2;

            switch (move.GetPiece())
            {
                case Piece::Knight: [[fallthrough]];
                case Piece::Bishop:
                    if (node.threats.attackedByPawns & move.FromSquare())   score += 4000;
                    if (node.threats.attackedByPawns & move.ToSquare())     score -= 4000;
                    break;
                case Piece::Rook:
                    if (node.threats.attackedByMinors & move.FromSquare())  score += 8000;
                    if (node.threats.attackedByMinors & move.ToSquare())    score -= 8000;
                    break;
                case Piece::Queen:
                    if (node.threats.attackedByRooks & move.FromSquare())   score += 12000;
                    if (node.threats.attackedByRooks & move.ToSquare())     score -= 12000;
                    break;
                default:
                    break;
            }

            // use node cache for scoring moves near the root
            if (nodeCacheEntry && nodeCacheEntry->nodesSum > 512)
            {
                if (const NodeCacheEntry::MoveInfo* moveInfo = nodeCacheEntry->GetMove(move))
                {
                    const float fraction = static_cast<float>(moveInfo->nodesSearched) / static_cast<float>(nodeCacheEntry->nodesSum);
                    ASSERT(fraction >= 0.0f);
                    ASSERT(fraction <= 1.0f);
                    score += static_cast<int32_t>(4096.0f * sqrtf(fraction) * FastLog2(static_cast<float>(nodeCacheEntry->nodesSum) / 512.0f));
                }
            }
        }

        if (move.GetPromoteTo() != Piece::None)
        {
            ASSERT((uint32_t)move.GetPromoteTo() >= (uint32_t)Piece::Knight && (uint32_t)move.GetPromoteTo() <= (uint32_t)Piece::Queen);
            score += PromotionValues[uint32_t(move.GetPromoteTo())];
        }

        moves.entries[i].score = score;
    }
}
