#pragma once

#include "Common.hpp"
#include "Material.hpp"

class EvalCorrection
{
private:
    static constexpr int32_t Grain = 512;
    static constexpr int32_t LearningRateFactor = 256;

    static constexpr uint32_t MaterialTableSize = 2048;
    static constexpr uint32_t PawnStructureTableSize = 2048;

    int16_t matScoreTable[2][MaterialTableSize];
    int16_t pawnStructureTable[2][PawnStructureTableSize];

public:

    void Clear()
    {
        memset(matScoreTable, 0, sizeof(matScoreTable));
        memset(pawnStructureTable, 0, sizeof(pawnStructureTable));
    }

    INLINE ScoreType Apply(ScoreType rawScore, const Position& pos) const
    {
        const int32_t stm = pos.GetSideToMove();

        const int32_t matIndex = Murmur3(pos.GetMaterialKey().value) % MaterialTableSize;
        const int32_t pawnIndex = pos.GetPawnsHash() % PawnStructureTableSize;
        
        const int16_t matEntry = matScoreTable[stm][matIndex];
        const int16_t pawnEntry = pawnStructureTable[stm][pawnIndex];

        return rawScore + (matEntry + pawnEntry) / Grain;
    }

    INLINE void Update(const Position& pos, ScoreType rawScore, ScoreType trueScore)
    {
        const int32_t stm = pos.GetSideToMove();

        const int32_t matIndex = Murmur3(pos.GetMaterialKey().value) % MaterialTableSize;
        const int32_t pawnIndex = pos.GetPawnsHash() % PawnStructureTableSize;

        int16_t& matEntry = matScoreTable[stm][matIndex];
        int16_t& pawnEntry = pawnStructureTable[stm][pawnIndex];

        // error = Grain * (trueScore - adjusted)
        const int32_t error = Grain * (trueScore - rawScore) - (matEntry + pawnEntry);

        matEntry = (int16_t)std::clamp(matEntry + error / LearningRateFactor, -63 * Grain, 63 * Grain);
        pawnEntry = (int16_t)std::clamp(pawnEntry + error / LearningRateFactor, -63 * Grain, 63 * Grain);
    }
};
