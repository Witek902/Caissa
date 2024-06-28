#include "NodeCache.hpp"

#include <algorithm>
#include <iomanip>

void NodeCacheEntry::PrintMoves() const
{
    std::vector<MoveInfo> sortedMoves;

    uint64_t totalNodesSearched = 0;
    for (const MoveInfo& info : moves)
    {
        if (!info.move.IsValid()) continue;
        sortedMoves.push_back(info);
        totalNodesSearched += info.nodesSearched;
    }

    std::sort(sortedMoves.begin(), sortedMoves.end(), [](const MoveInfo& a, const MoveInfo& b)
    {
        return a.nodesSearched >= b.nodesSearched;
    });

    for (const MoveInfo& info : sortedMoves)
    {
        std::cout
            << info.move.ToString() << " "
            << std::setw(10) << info.nodesSearched
            << " (" << std::setprecision(4) << (100.0f * static_cast<float>(info.nodesSearched) / static_cast<float>(totalNodesSearched)) << "%)"
            << std::endl;
    }
}

void NodeCacheEntry::ScaleDown()
{
    nodesSum = 0;

    for (MoveInfo& moveInfo : moves)
    {
        moveInfo.nodesSearched /= 2;
        nodesSum += moveInfo.nodesSearched;
    }
}

const NodeCacheEntry::MoveInfo* NodeCacheEntry::GetMove(const Move move) const
{
    for (const MoveInfo& moveInfo : moves)
    {
        if (moveInfo.move == move)
        {
            return &moveInfo;
        }
    }
    return nullptr;
}

void NodeCacheEntry::AddMoveStats(const Move& move, uint64_t numNodes)
{
    // for replacing least-visited move
    uint64_t minNodes = UINT64_MAX;
    uint32_t minNodesMoveIndex = UINT32_MAX;

    bool moveFound = false;

    for (uint32_t i = 0; i < MaxMoves; ++i)
    {
        MoveInfo& moveInfo = moves[i];

        if (moveInfo.move == move)
        {
            moveInfo.nodesSearched += numNodes;
            nodesSum += numNodes;
            moveFound = true;

            // scale down to avoid overflow
            if (moveInfo.nodesSearched >= UINT64_MAX / MaxMoves)
            {
                ScaleDown();
            }

            break;
        }

        if (!moveInfo.move.IsValid() ||
            (moveInfo.nodesSearched < minNodes && moveInfo.nodesSearched < numNodes))
        {
            minNodes = moveInfo.nodesSearched;
            minNodesMoveIndex = i;
        }
    }

    // if no existing move found, replace move with least amount of visits
    if (!moveFound && minNodesMoveIndex < MaxMoves)
    {
        MoveInfo& moveInfo = moves[minNodesMoveIndex];

        nodesSum -= moveInfo.nodesSearched;
        nodesSum += numNodes;
        
        moveInfo.move = move;
        moveInfo.nodesSearched = numNodes;
    }
}

void NodeCache::Reset()
{
    generation = 0;
    for (NodeCacheEntry& entry : entries)
    {
        entry = NodeCacheEntry{};
    }
}

void NodeCache::OnNewSearch()
{
    generation++;
}

const NodeCacheEntry* NodeCache::TryGetEntry(const Position& pos) const
{
    const uint32_t index = pos.GetHash() % Size;
    const NodeCacheEntry* entry = entries + index;

    if (entry->position == pos)
    {
        return entry;
    }

    return nullptr;
}


NodeCacheEntry* NodeCache::GetEntry(const Position& pos, uint32_t distanceFromRoot)
{
    const uint32_t index = pos.GetHash() % Size;
    NodeCacheEntry* entry = entries + index;

    // return existing entry
    if (entry->position == pos)
    {
        entry->generation = generation;
        entry->distanceFromRoot = distanceFromRoot;

        return entry;
    }

    // allocate new entry
    if (entry->generation < generation)
    {
        *entry = NodeCacheEntry{};
        entry->position = pos;
        entry->generation = generation;
        entry->distanceFromRoot = distanceFromRoot;

        return entry;
    }

    // allocation failed
    // TODO try other entry index
    return nullptr;
}
