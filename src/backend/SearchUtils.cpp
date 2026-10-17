#include "SearchUtils.hpp"
#include "Search.hpp"

// Upcoming repetition detection algorithm base on Stockfish implementation
// Algorithm by Marcel van Kervinck:
// http://www.open-chess.org/viewtopic.php?f=5&t=2300

constexpr uint32_t CuckooTableSize = 8192;

// First and second hash functions for indexing the cuckoo tables
INLINE uint32_t CuckooIndex1(uint64_t h) { return h % CuckooTableSize; }
INLINE uint32_t CuckooIndex2(uint64_t h) { return (h >> 16) % CuckooTableSize; }

// Cuckoo tables with Zobrist hashes of valid reversible moves, and the corresponding moves
static uint64_t gCuckooTable[CuckooTableSize];
static PackedMove gCuckooMoves[CuckooTableSize];

void SearchUtils::Init()
{
    memset(gCuckooTable, 0, sizeof(gCuckooTable));
    memset(gCuckooMoves, 0, sizeof(gCuckooMoves));

    uint32_t count = 0;
    constexpr Color colors[] = { White, Black };
    for (const Color color : colors)
    {
        // Note: pawn moves not included as they are not reversible
        constexpr Piece pieces[] = { Piece::Knight, Piece::Bishop, Piece::Rook, Piece::Queen, Piece::King };
        for (const Piece piece : pieces)
        {
            for (uint32_t squareA = 0; squareA < 64; ++squareA)
            {
                for (uint32_t squareB = squareA + 1; squareB < 64; ++squareB)
                {
                    Bitboard attacks;
                    switch (piece)
                    {
                    case Piece::Pawn:   attacks = Bitboard::GetPawnAttacks(Square(squareA), color);     break;
                    case Piece::Knight: attacks = Bitboard::GetKnightAttacks(Square(squareA));          break;
                    case Piece::Bishop: attacks = Bitboard::GetBishopAttacks(Square(squareA));          break;
                    case Piece::Rook:   attacks = Bitboard::GetRookAttacks(Square(squareA));            break;
                    case Piece::Queen:  attacks = Bitboard::GetQueenAttacks(Square(squareA));           break;
                    case Piece::King:   attacks = Bitboard::GetKingAttacks(Square(squareA));            break;
                    }

                    if (attacks & Square(squareB).GetBitboard())
                    {
                        PackedMove move{ Square(squareA), Square(squareB) };
                        uint64_t key = GetPieceZobristHash(color, piece, squareA) ^ GetPieceZobristHash(color, piece, squareB) ^ c_SideToMoveZobristHash;
                        uint32_t index = CuckooIndex1(key);
                        for (;;)
                        {
                            Swap(gCuckooTable[index], key);
                            Swap(gCuckooMoves[index], move);
                            if (!move.IsValid()) break;
                            index = (index == CuckooIndex1(key)) ? CuckooIndex2(key) : CuckooIndex1(key);
                        }
                        count++;
                    }
                }
            }
        }
    }
    ASSERT(count == 3668);
    UNUSED(count);
}

bool SearchUtils::CanReachGameCycle(const NodeInfo& node)
{
    if (node.position.GetHalfMoveCount() < 3)
        return false;

    if (node.isNullMove || node.previousMove.IsIrreversible())
        return false;

    const uint64_t originalKey = node.position.GetHash();
    const NodeInfo* currNode = &node - 1;
    ASSERT(currNode);

    for (;;)
    {
        // go up the tree, abort on any null move or irreversible move
        if (currNode->ply < 2) break;

        if (currNode->isNullMove || currNode->previousMove.IsIrreversible()) break;
        currNode = currNode - 1;
        if (currNode->isNullMove || currNode->previousMove.IsIrreversible()) break;
        currNode = currNode - 1;

        ASSERT(node.position.GetSideToMove() != currNode->position.GetSideToMove());
        const uint64_t moveKey = originalKey ^ currNode->position.GetHash();

        uint32_t index = UINT32_MAX;
        if (gCuckooTable[CuckooIndex1(moveKey)] == moveKey) index = CuckooIndex1(moveKey);
        else if (gCuckooTable[CuckooIndex2(moveKey)] == moveKey) index = CuckooIndex2(moveKey);

        // no move found in the table for given hash difference
        if (index >= CuckooTableSize)
            continue;

        const PackedMove move = gCuckooMoves[index];
        ASSERT(move.IsValid());

        // move is not legal
        if (Bitboard::GetBetween(move.FromSquare(), move.ToSquare()) & node.position.Occupied())
            continue;

        const Bitboard occupied = node.position.GetCurrentSide().Occupied();
        if (occupied & (move.FromSquare().GetBitboard() | move.ToSquare().GetBitboard()))
            return true;
    }

    return false;
}

uint32_t SearchUtils::GetPvLine(const NodeInfo& rootNode, uint32_t maxLength, Move* outLine)
{
    uint32_t length = 0;

    if (maxLength > 0)
    {
        Position iteratedPosition = rootNode.position;

        // reconstruct PV line using PV array
        for (; length < Min<uint32_t>(maxLength, rootNode.pvLength); ++length)
        {
            ASSERT(rootNode.pvLine[length].IsValid());
            const Move move = iteratedPosition.MoveFromPacked(rootNode.pvLine[length]);

            ASSERT(move.IsValid());
            if (!move.IsValid()) break;
            if (!iteratedPosition.DoMove(move)) break;

            outLine[length] = move;
        }
    }

    return length;
}
