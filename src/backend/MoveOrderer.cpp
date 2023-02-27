#include "MoveOrderer.hpp"
#include "Search.hpp"
#include "MoveList.hpp"
#include "Evaluate.hpp"
#include "Game.hpp"

#include <algorithm>
#include <limits>
#include <iomanip>

static constexpr int32_t RecaptureBonus = 100000;

MoveOrderer::MoveOrderer()
{
    Clear();
}

void MoveOrderer::DebugPrint() const
{
#ifndef CONFIGURATION_FINAL
    std::cout << "=== QUIET MOVES HISTORY HEURISTICS ===" << std::endl;

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
    }

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
                    const CounterType count = continuationHistory[0][prevPiece][prevToIndex][piece][toIndex];

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

    for (uint32_t numPieces = 3; numPieces <= MaxNumPieces; ++numPieces)
    {
        std::cout << numPieces << " pieces:" << std::endl;

        uint32_t lastValidDepth = 0;
        for (uint32_t d = 0; d < MaxSearchDepth; ++d)
        {
            for (uint32_t i = 0; i < NumKillerMoves; ++i)
            {
                if (killerMoves[numPieces][d].moves[i].IsValid())
                {
                    lastValidDepth = std::max(lastValidDepth, d);
                }
            }
        }

        for (uint32_t d = 0; d < lastValidDepth; ++d)
        {
            std::cout << d;
            for (uint32_t i = 0; i < NumKillerMoves; ++i)
            {
                std::cout << "\t" << killerMoves[numPieces][d].moves[i].ToString() << " ";
            }
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

void MoveOrderer::NewSearch()
{
    const CounterType scaleDownFactor = 2;

    for (uint32_t i = 0; i < sizeof(quietMoveHistory) / sizeof(CounterType); ++i)
    {
        reinterpret_cast<CounterType*>(quietMoveHistory)[i] /= scaleDownFactor;
    }

    memset(killerMoves, 0, sizeof(killerMoves));
}

void MoveOrderer::Clear()
{
    memset(quietMoveHistory, 0, sizeof(quietMoveHistory));
    memset(continuationHistory, 0, sizeof(continuationHistory));
    memset(counterMoveHistory, 0, sizeof(counterMoveHistory));
    memset(capturesHistory, 0, sizeof(capturesHistory));
    memset(killerMoves, 0, sizeof(killerMoves));
}

INLINE static void UpdateHistoryCounter(MoveOrderer::CounterType& counter, int32_t delta)
{
    int32_t newValue = (int32_t)counter + delta - ((int32_t)counter * std::abs(delta) + 8192) / 16384;

    // there should be no saturation
    ASSERT(newValue > std::numeric_limits<MoveOrderer::CounterType>::min());
    ASSERT(newValue < std::numeric_limits<MoveOrderer::CounterType>::max());

    counter = static_cast<MoveOrderer::CounterType>(newValue);
}

void MoveOrderer::UpdateQuietMovesHistory(const NodeInfo& node, const Move* moves, uint32_t numMoves, const Move bestMove, int32_t depth)
{
    ASSERT(depth >= 0);

    // don't update uncertain moves
    if (numMoves <= 1 && depth < 2)
    {
        return;
    }

    const uint32_t color = (uint32_t)node.position.GetSideToMove();

    PieceSquareHistory* continuationHistories[6] = { nullptr, nullptr, nullptr, nullptr, nullptr, nullptr };
    {
        const NodeInfo* prevNode = &node;
        for (uint32_t i = 0; i < 6; ++i)
        {
            if (!prevNode || prevNode->isNullMove) break;
            if (prevNode->previousMove.IsValid())
            {
                const uint32_t prevPiece = (uint32_t)prevNode->previousMove.GetPiece() - 1;
                const uint32_t prevTo = prevNode->previousMove.ToSquare().Index();
                ContinuationHistory& historyTable = (i % 2 == 0) ? counterMoveHistory : continuationHistory;
                continuationHistories[i] = &(historyTable[color][prevPiece][prevTo]);
            }
            prevNode = prevNode->parentNode;
        }
    }

    const int32_t bonus = std::min(64 * (depth - 1) + depth * depth, 2000);

    for (uint32_t i = 0; i < numMoves; ++i)
    {
        const Move move = moves[i];
        const int32_t delta = move == bestMove ? bonus : -bonus;

        const uint32_t piece = (uint32_t)move.GetPiece() - 1;
        const uint32_t from = move.FromSquare().Index();
        const uint32_t to = move.ToSquare().Index();

        UpdateHistoryCounter(quietMoveHistory[color][from][to], delta);

        if (PieceSquareHistory* h = continuationHistories[0]) UpdateHistoryCounter((*h)[piece][to], delta);
        if (PieceSquareHistory* h = continuationHistories[1]) UpdateHistoryCounter((*h)[piece][to], delta);
        if (PieceSquareHistory* h = continuationHistories[3]) UpdateHistoryCounter((*h)[piece][to], delta);
        if (PieceSquareHistory* h = continuationHistories[5]) UpdateHistoryCounter((*h)[piece][to], delta);
    }
}

void MoveOrderer::UpdateCapturesHistory(const NodeInfo& node, const Move* moves, uint32_t numMoves, const Move bestMove, int32_t depth)
{
    // depth can be negative in QSearch
    depth = std::max(0, depth);

    // don't update uncertain moves
    if (numMoves <= 1)
    {
        return;
    }

    const uint32_t color = (uint32_t)node.position.GetSideToMove();

    const int32_t bonus = std::min(16 + 32 * depth + depth * depth, 2000);

    for (uint32_t i = 0; i < numMoves; ++i)
    {
        const Move move = moves[i];
        ASSERT(move.IsCapture());

        const int32_t delta = move == bestMove ? bonus : -bonus;

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

void MoveOrderer::UpdateKillerMove(const NodeInfo& node, const Move move)
{
    if (node.height < MaxSearchDepth)
    {
        const uint32_t numPieces = std::min(MaxNumPieces, node.position.GetNumPieces());
        killerMoves[numPieces][node.height].Push(move);
    }
}

void MoveOrderer::ClearKillerMoves(uint32_t depth)
{
    if (depth < MaxSearchDepth)
    {
        for (uint32_t numPieces = 0; numPieces <= MaxNumPieces; ++numPieces)
        {
            killerMoves[numPieces][depth].Clear();
        }
    }
}

void MoveOrderer::ScoreMoves(
    const NodeInfo& node,
    const Game& game,
    MoveList& moves,
    bool withQuiets,
    const NodeCacheEntry* nodeCacheEntry) const
{
    const Position& pos = node.position;

    const uint32_t color = (uint32_t)pos.GetSideToMove();
    const uint32_t numPieces = std::min(MaxNumPieces, pos.GetNumPieces());
    
    Bitboard attackedByPawns = 0;
    Bitboard attackedByMinors = 0;
    Bitboard attackedByRooks = 0;

    if (withQuiets)
    {
        const SidePosition& currentSide = pos.GetCurrentSide();
        const SidePosition& opponentSide = pos.GetOpponentSide();
        const Bitboard occupied = pos.Occupied();

        attackedByPawns = (pos.GetSideToMove() == Color::White) ?
            Bitboard::GetPawnAttacks<Color::Black>(opponentSide.pawns) :
            Bitboard::GetPawnAttacks<Color::White>(opponentSide.pawns);

        if (currentSide.rooks | currentSide.queens)
        {
            attackedByMinors = attackedByPawns |
                Bitboard::GetKnightAttacks(pos.GetOpponentSide().knights);
            opponentSide.bishops.Iterate([&](uint32_t fromIndex) INLINE_LAMBDA{
                attackedByMinors |= Bitboard::GenerateBishopAttacks(Square(fromIndex), occupied); });
        }

        if (currentSide.queens)
        {
            attackedByRooks = attackedByMinors;
            opponentSide.rooks.Iterate([&](uint32_t fromIndex) INLINE_LAMBDA{
                attackedByRooks |= Bitboard::GenerateRookAttacks(Square(fromIndex), occupied); });
        }
    }

    const auto& killerMovesForCurrentNode = killerMoves[numPieces][node.height];

    Move prevMove = !node.isNullMove ? node.previousMove : Move::Invalid();

    // at the root node, obtain previous move from the game data
    if (node.height == 0)
    {
        ASSERT(!prevMove.IsValid());
        if (!game.GetMoves().empty())
        {
            prevMove = game.GetMoves().back();
        }
    }

    const PieceSquareHistory* continuationHistories[6] = { nullptr, nullptr, nullptr, nullptr, nullptr, nullptr };
    {
        const NodeInfo* prevNode = &node;
        for (uint32_t i = 0; i < 6; ++i)
        {
            if (!prevNode || prevNode->isNullMove) break;
            if (prevNode->previousMove.IsValid())
            {
                const uint32_t prevPiece = (uint32_t)prevNode->previousMove.GetPiece() - 1;
                const uint32_t prevTo = prevNode->previousMove.ToSquare().Index();
                const ContinuationHistory& historyTable = (i % 2 == 0) ? counterMoveHistory : continuationHistory;
                continuationHistories[i] = &(historyTable[color][prevPiece][prevTo]);
            }
            prevNode = prevNode->parentNode;
        }
    }

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
        if (moves.scores[i] > INT32_MIN) continue;

        int64_t score = 0;

        if (move.IsCapture())
        {
            const Piece attackingPiece = move.GetPiece();
            const Piece capturedPiece = pos.GetCapturedPiece(move);
            ASSERT(capturedPiece > Piece::None);
            ASSERT(capturedPiece < Piece::King);

            if ((uint32_t)attackingPiece <= (uint32_t)capturedPiece)
            {
                score = GoodCaptureValue;
            }
            else
            {
                if (pos.StaticExchangeEvaluation(move))     score = GoodCaptureValue;
                else                                        score = LosingCaptureValue;
            }

            // most valuable victim first
            score += 6 * (int32_t)capturedPiece * UINT16_MAX / 128;

            // capture history
            {
                const uint32_t capturedIdx = (uint32_t)capturedPiece - 1;
                const uint32_t pieceIdx = (uint32_t)attackingPiece - 1;
                ASSERT(capturedIdx < 5);
                ASSERT(pieceIdx < 6);
                score += ((int32_t)capturesHistory[color][pieceIdx][capturedIdx][move.ToSquare().Index()] - INT16_MIN) / 128;
            }

            // bonus for capturing previously moved piece
            if (prevMove.IsValid() && move.ToSquare() == prevMove.ToSquare())
            {
                score += RecaptureBonus;
            }
        }
        else if (withQuiets) // non-capture
        {
            // killer moves heuristics
            if (node.height < MaxSearchDepth)
            {
                const int32_t killerMoveIndex = killerMovesForCurrentNode.Find(move);
                if (killerMoveIndex >= 0)
                {
                    score = KillerMoveBonus - killerMoveIndex;
                }
            }

            // not killer move
            if (score <= KillerMoveBonus - NumKillerMoves)
            {
                // history heuristics
                score += quietMoveHistory[color][from][to];

                // continuation history
                if (const PieceSquareHistory* h = continuationHistories[0]) score += (*h)[piece][to];
                if (const PieceSquareHistory* h = continuationHistories[1]) score += (*h)[piece][to];
                if (const PieceSquareHistory* h = continuationHistories[3]) score += (*h)[piece][to];
                if (const PieceSquareHistory* h = continuationHistories[5]) score += (*h)[piece][to];

                // bonus for moving a piece out of attack
                if (move.GetPiece() == Piece::Queen && (attackedByRooks & move.FromSquare().GetBitboard()))
                {
                    score += 16000;
                }
                else if (move.GetPiece() == Piece::Rook && (attackedByMinors & move.FromSquare().GetBitboard()))
                {
                    score += 12000;
                }
                else if ((move.GetPiece() == Piece::Knight || move.GetPiece() == Piece::Bishop) && (attackedByPawns & move.FromSquare().GetBitboard()))
                {
                    score += 8000;
                }

                // penalty for moving a piece into attack
                if (move.GetPiece() == Piece::Queen && (attackedByRooks & move.ToSquare().GetBitboard()))
                {
                    score -= 16000 + 4000;
                }
                else if (move.GetPiece() == Piece::Rook && (attackedByMinors & move.ToSquare().GetBitboard()))
                {
                    score -= 12000 + 4000;
                }
                else if ((move.GetPiece() == Piece::Knight || move.GetPiece() == Piece::Bishop) && (attackedByPawns & move.ToSquare().GetBitboard()))
                {
                    score -= 8000 + 4000;
                }

                // pawn push bonus
                if (move.GetPiece() == Piece::Pawn)
                {
                    constexpr int32_t bonus[] = { 0, 0, 0, 0, 500, 2000, 8000, 0 };
                    const uint8_t rank = move.ToSquare().RelativeRank(pos.GetSideToMove());
                    score += bonus[rank];
                }

                // use node cache for scoring moves near the root
                if (nodeCacheEntry && nodeCacheEntry->nodesSum > 512)
                {
                    if (const NodeCacheEntry::MoveInfo* moveInfo = nodeCacheEntry->GetMove(move))
                    {
                        const float fraction = static_cast<float>(moveInfo->nodesSearched) / static_cast<float>(nodeCacheEntry->nodesSum);
                        ASSERT(fraction >= 0.0f);
                        ASSERT(fraction <= 1.0f);
                        score += static_cast<int32_t>(4096.0f * sqrtf(fraction) * log2f(static_cast<float>(nodeCacheEntry->nodesSum) / 512.0f));
                    }
                }
            }
        }

        if (move.GetPromoteTo() == Piece::Queen)
        {
            score += PromotionValue;
        }

        moves.scores[i] = (int32_t)std::min<int64_t>(score, INT32_MAX);
    }
}
