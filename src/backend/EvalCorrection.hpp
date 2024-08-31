#pragma once

#include "Common.hpp"
#include "Material.hpp"

class EvalCorrection
{
private:
    static constexpr int32_t Grain = 256;
    static constexpr int32_t BlendFactor = 256;

    static constexpr uint32_t MaterialTableSize = 2048;
    static constexpr uint32_t PawnStructureTableSize = 1024;

    int16_t matScoreTable[2][MaterialTableSize];
    int16_t pawnStructureTable[2][PawnStructureTableSize];

public:

    void Clear()
    {
        memset(matScoreTable, 0, sizeof(matScoreTable));
        memset(pawnStructureTable, 0, sizeof(pawnStructureTable));
    }

    ScoreType Apply(ScoreType rawScore, const Position& pos) const
    {
        const int32_t stm = pos.GetSideToMove();

        const int32_t matIndex = Murmur3(pos.GetMaterialKey().value) % MaterialTableSize;
        const int32_t pawnIndex = pos.GetPawnsHash() % PawnStructureTableSize;
        
        const int16_t matScore = matScoreTable[stm][matIndex];
        const int16_t pawnScore = pawnStructureTable[stm][pawnIndex];

        return rawScore + (matScore + pawnScore) / Grain;
    }

    void Update(const Position& pos, ScoreType rawScore, ScoreType trueScore)
    {
        const int32_t stm = pos.GetSideToMove();

        const int32_t matIndex = Murmur3(pos.GetMaterialKey().value) % MaterialTableSize;
        const int32_t pawnIndex = pos.GetPawnsHash() % PawnStructureTableSize;

        int16_t& matScore = matScoreTable[stm][matIndex];
        int16_t& pawnScore = pawnStructureTable[stm][pawnIndex];

        // adjusted = rawScore + (matScore + pawnScore) / Grain
        // diff = Grain * (trueScore - evalScore)
        const int32_t diff = std::clamp(Grain * (trueScore - rawScore) - (matScore + pawnScore), -127 * Grain, 127 * Grain);

        matScore = static_cast<int16_t>((matScore * (BlendFactor - 1) + diff) / BlendFactor);
        pawnScore = static_cast<int16_t>((pawnScore * (BlendFactor - 1) + diff) / BlendFactor);
    }
};
