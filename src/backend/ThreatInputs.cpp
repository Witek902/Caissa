#include "ThreatInputs.hpp"

#include "containers/Sort.hpp"

namespace {

static constexpr uint32_t NumPieceCodes = 16;
static constexpr uint32_t NumPieceTypes = 7;

// Slot of each victim type inside an attacker's feature block, -1 if the pair is not a feature.
// A pawn attacking a bishop or a queen, or a bishop or a rook attacking a queen,
// always implies the threat back. Kings are neither attackers nor victims.
static constexpr int8_t c_victimSlot[NumPieceTypes][NumPieceTypes] =
{
    // -   P   N   B   R   Q   K        victim
    { -1, -1, -1, -1, -1, -1, -1 },  // none
    { -1,  0,  1, -1,  2, -1, -1 },  // pawn
    { -1,  0,  1,  2,  3,  4, -1 },  // knight
    { -1,  0,  1,  2,  3, -1, -1 },  // bishop
    { -1,  0,  1,  2,  3, -1, -1 },  // rook
    { -1,  0,  1,  2,  3,  4, -1 },  // queen
    { -1, -1, -1, -1, -1, -1, -1 },  // king
};

static constexpr uint32_t c_numVictimSlots[NumPieceTypes] = { 0, 3, 5, 4, 4, 5, 0 };

// All tables are indexed by perspective-relative piece codes and oriented squares.

// first feature of an (attacker, victim) block, by "from < to"; NumThreatInputs if the pair is not a feature
uint32_t s_featureBase[NumPieceCodes][NumPieceCodes][2];

// number of empty-board attacks of the attacker from all lower squares
uint16_t s_fromOffset[NumPieceCodes][64];

// empty-board attacks; pawns of the perspective's side attack towards rank 8
Bitboard s_emptyBoardAttacks[NumPieceCodes][64];

INLINE Piece CodeToPiece(const uint8_t code)
{
    return static_cast<Piece>(code & 7);
}

INLINE Color CodeToColor(const uint8_t code)
{
    return static_cast<Color>(code >> 3);
}

Bitboard GetEmptyBoardAttacks(const Piece piece, const Color color, const Square square)
{
    switch (piece)
    {
    case Piece::Pawn:
        // pawns never stand on the first or the last rank
        if (square.Rank() == 0 || square.Rank() == 7)
            return 0;
        return Bitboard::GetPawnAttacks(square, color);
    case Piece::Knight: return Bitboard::GetKnightAttacks(square);
    case Piece::Bishop: return Bitboard::GenerateBishopAttacks(square, 0);
    case Piece::Rook:   return Bitboard::GenerateRookAttacks(square, 0);
    case Piece::Queen:  return Bitboard::GenerateQueenAttacks(square, 0);
    default:            return 0;
    }
}

INLINE Bitboard GetPieceAttacks(const Piece piece, const Color color, const Square square, const Bitboard occupied)
{
    switch (piece)
    {
    case Piece::Pawn:   return Bitboard::GetPawnAttacks(square, color);
    case Piece::Knight: return Bitboard::GetKnightAttacks(square);
    case Piece::Bishop: return Bitboard::GenerateBishopAttacks(square, occupied);
    case Piece::Rook:   return Bitboard::GenerateRookAttacks(square, occupied);
    case Piece::Queen:  return Bitboard::GenerateQueenAttacks(square, occupied);
    default:            return 0;
    }
}

INLINE uint8_t GetPieceCodeAt(const Position& pos, const Square square)
{
    const Color color = (pos.Whites().Occupied() & square) ? White : Black;
    const Piece piece = pos.GetSide(color).GetPieceAtSquare(square);
    ASSERT(piece != Piece::None);
    return ThreatPieceCode(piece, color);
}

// squares of the pieces each attacker type has features against
void GetVictimMasks(const Position& pos, Bitboard outMasks[NumPieceTypes])
{
    for (uint32_t attacker = 0; attacker < NumPieceTypes; ++attacker)
    {
        Bitboard mask = 0;
        for (uint32_t victim = (uint32_t)Piece::Pawn; victim < (uint32_t)Piece::King; ++victim)
        {
            if (c_victimSlot[attacker][victim] >= 0)
                mask |= pos.Whites().GetPieceBitBoard((Piece)victim) | pos.Blacks().GetPieceBitBoard((Piece)victim);
        }
        outMasks[attacker] = mask;
    }
}

// non-king pieces attacking the square that have a feature against the victim type
Bitboard GetThreatAttackers(const Position& pos, const Square square, const Bitboard occupied, const Piece victim)
{
    const SidePosition& whites = pos.Whites();
    const SidePosition& blacks = pos.Blacks();
    const uint32_t v = (uint32_t)victim;

    const Bitboard rookRays = Bitboard::GenerateRookAttacks(square, occupied);
    const Bitboard bishopRays = Bitboard::GenerateBishopAttacks(square, occupied);

    Bitboard attackers = 0;
    if (c_victimSlot[(uint32_t)Piece::Pawn][v] >= 0)
        attackers |= (Bitboard::GetPawnAttacks(square, Black) & whites.pawns) | (Bitboard::GetPawnAttacks(square, White) & blacks.pawns);
    if (c_victimSlot[(uint32_t)Piece::Knight][v] >= 0)
        attackers |= Bitboard::GetKnightAttacks(square) & (whites.knights | blacks.knights);
    if (c_victimSlot[(uint32_t)Piece::Bishop][v] >= 0)
        attackers |= bishopRays & (whites.bishops | blacks.bishops);
    if (c_victimSlot[(uint32_t)Piece::Rook][v] >= 0)
        attackers |= rookRays & (whites.rooks | blacks.rooks);
    if (c_victimSlot[(uint32_t)Piece::Queen][v] >= 0)
        attackers |= (rookRays | bishopRays) & (whites.queens | blacks.queens);
    return attackers;
}

INLINE void AppendThreat(ThreatRecord* records, uint32_t& numRecords, const uint8_t attacker, const Square from, const uint8_t victim, const Square to)
{
    ASSERT(numRecords < nn::MaxChangedThreats);
    records[numRecords++] = { attacker, from.Index(), victim, to.Index() };
}

// threats with the attacker on a changed square, and threats onto a changed square from an unchanged one
void AppendChangedSquareThreats(const Position& pos, const Bitboard changed, ThreatRecord* records, uint32_t& numRecords)
{
    const Bitboard occupied = pos.Occupied();
    const Bitboard nonKings = pos.Whites().OccupiedExcludingKing() | pos.Blacks().OccupiedExcludingKing();

    Bitboard victimMasks[NumPieceTypes];
    GetVictimMasks(pos, victimMasks);

    (changed & nonKings).Iterate([&](const Square square) INLINE_LAMBDA
    {
        const uint8_t code = GetPieceCodeAt(pos, square);
        const Piece piece = CodeToPiece(code);

        const Bitboard targets = GetPieceAttacks(piece, CodeToColor(code), square, occupied) & victimMasks[(uint32_t)piece];
        targets.Iterate([&](const Square to) INLINE_LAMBDA
        {
            AppendThreat(records, numRecords, code, square, GetPieceCodeAt(pos, to), to);
        });

        const Bitboard attackers = GetThreatAttackers(pos, square, occupied, piece) & ~changed;
        attackers.Iterate([&](const Square from) INLINE_LAMBDA
        {
            AppendThreat(records, numRecords, GetPieceCodeAt(pos, from), from, code, square);
        });
    });
}

// Threats between pieces on unchanged squares change only for sliders whose ray passes a changed square.
void AppendSliderThreatChanges(const Position& before, const Position& after, const Bitboard changed, ThreatDeltas& deltas)
{
    const Bitboard occupiedBefore = before.Occupied();
    const Bitboard occupiedAfter = after.Occupied();

    // unchanged squares hold the same pieces in both positions
    const SidePosition& whites = after.Whites();
    const SidePosition& blacks = after.Blacks();
    const Bitboard queens = (whites.queens | blacks.queens) & ~changed;
    const Bitboard rookLike = ((whites.rooks | blacks.rooks) & ~changed) | queens;
    const Bitboard bishopLike = ((whites.bishops | blacks.bishops) & ~changed) | queens;

    Bitboard sliders = 0;
    changed.Iterate([&](const Square square) INLINE_LAMBDA
    {
        sliders |= (Bitboard::GenerateRookAttacks(square, occupiedBefore) | Bitboard::GenerateRookAttacks(square, occupiedAfter)) & rookLike;
        sliders |= (Bitboard::GenerateBishopAttacks(square, occupiedBefore) | Bitboard::GenerateBishopAttacks(square, occupiedAfter)) & bishopLike;
    });

    if (!sliders)
        return;

    Bitboard victimMasks[NumPieceTypes];
    GetVictimMasks(after, victimMasks);

    sliders.Iterate([&](const Square from) INLINE_LAMBDA
    {
        const uint8_t code = GetPieceCodeAt(after, from);
        const Piece piece = CodeToPiece(code);
        const Color color = CodeToColor(code);
        const Bitboard victims = victimMasks[(uint32_t)piece] & ~changed;

        const Bitboard targetsBefore = GetPieceAttacks(piece, color, from, occupiedBefore) & victims;
        const Bitboard targetsAfter = GetPieceAttacks(piece, color, from, occupiedAfter) & victims;

        (targetsBefore & ~targetsAfter).Iterate([&](const Square to) INLINE_LAMBDA
        {
            AppendThreat(deltas.removed, deltas.numRemoved, code, from, GetPieceCodeAt(after, to), to);
        });

        (targetsAfter & ~targetsBefore).Iterate([&](const Square to) INLINE_LAMBDA
        {
            AppendThreat(deltas.added, deltas.numAdded, code, from, GetPieceCodeAt(after, to), to);
        });
    });
}

// sorted feature lists: the features only in the first list
uint32_t SubtractSortedFeatures(const uint16_t* a, uint32_t numA, const uint16_t* b, uint32_t numB, uint16_t* outFeatures)
{
    uint32_t numFeatures = 0;
    uint32_t j = 0;
    for (uint32_t i = 0; i < numA; ++i)
    {
        while (j < numB && b[j] < a[i])
            ++j;
        if (j == numB || b[j] != a[i])
            outFeatures[numFeatures++] = a[i];
    }
    return numFeatures;
}

bool EqualFeatures(uint16_t* a, uint32_t numA, uint16_t* b, uint32_t numB)
{
    if (numA != numB)
        return false;

    const auto less = [](const uint16_t x, const uint16_t y) { return x < y; };
    Sort(a, a + numA, less);
    Sort(b, b + numB, less);

    for (uint32_t i = 0; i < numA; ++i)
    {
        if (a[i] != b[i])
            return false;
    }
    return true;
}

} // namespace

void InitThreatInputs()
{
    for (uint32_t attacker = 0; attacker < NumPieceCodes; ++attacker)
    {
        for (uint32_t victim = 0; victim < NumPieceCodes; ++victim)
        {
            s_featureBase[attacker][victim][0] = nn::NumThreatInputs;
            s_featureBase[attacker][victim][1] = nn::NumThreatInputs;
        }
    }

    // blocks ordered by attacker color (perspective's side first) and type, then victim color and type
    uint32_t blockStart = 0;
    for (Color attackerColor = 0; attackerColor < 2; ++attackerColor)
    {
        for (uint32_t a = (uint32_t)Piece::Pawn; a < (uint32_t)Piece::King; ++a)
        {
            const Piece attacker = (Piece)a;
            const uint8_t attackerCode = ThreatPieceCode(attacker, attackerColor);

            uint32_t numPairs = 0;
            for (uint32_t square = 0; square < 64; ++square)
            {
                const Bitboard attacks = GetEmptyBoardAttacks(attacker, attackerColor, Square(square));
                s_emptyBoardAttacks[attackerCode][square] = attacks;
                s_fromOffset[attackerCode][square] = (uint16_t)numPairs;
                numPairs += attacks.Count();
            }

            for (Color victimColor = 0; victimColor < 2; ++victimColor)
            {
                for (uint32_t v = (uint32_t)Piece::Pawn; v < (uint32_t)Piece::King; ++v)
                {
                    const int32_t slot = c_victimSlot[a][v];
                    if (slot < 0)
                        continue;

                    // Pieces of the same type threaten each other both ways (except a pawn defending a pawn),
                    // so only the threat from the higher square is a feature.
                    const bool mutual = a == v && (attackerColor != victimColor || attacker != Piece::Pawn);

                    const uint32_t base = blockStart + (victimColor * c_numVictimSlots[a] + (uint32_t)slot) * numPairs;
                    const uint8_t victimCode = ThreatPieceCode((Piece)v, victimColor);
                    s_featureBase[attackerCode][victimCode][0] = base;
                    s_featureBase[attackerCode][victimCode][1] = mutual ? nn::NumThreatInputs : base;
                }
            }

            blockStart += 2 * c_numVictimSlots[a] * numPairs;
        }
    }

    ASSERT(blockStart == nn::NumThreatInputs);
}

uint32_t GetThreatFeatureIndex(const ThreatRecord& record, const ThreatOrientation orientation)
{
    const uint32_t attacker = record.attacker ^ orientation.pieceXor;
    const uint32_t victim = record.victim ^ orientation.pieceXor;
    const uint32_t from = record.from ^ orientation.squareXor;
    const uint32_t to = record.to ^ orientation.squareXor;

    const Bitboard lowerSquares = (1ull << to) - 1;
    return s_featureBase[attacker][victim][from < to] + s_fromOffset[attacker][from] + (s_emptyBoardAttacks[attacker][from] & lowerSquares).Count();
}

uint32_t ThreatRecordsToFeatures(const ThreatRecord* records, uint32_t numRecords, const ThreatOrientation orientation, uint16_t* outFeatures)
{
    uint32_t numFeatures = 0;
    for (uint32_t i = 0; i < numRecords; ++i)
    {
        // always written, kept only if it is a feature
        const uint32_t index = GetThreatFeatureIndex(records[i], orientation);
        outFeatures[numFeatures] = (uint16_t)index;
        numFeatures += index < nn::NumThreatInputs;
    }
    return numFeatures;
}

uint32_t GetThreatRecords(const Position& pos, ThreatRecord* outRecords)
{
    const Bitboard occupied = pos.Occupied();

    Bitboard victimMasks[NumPieceTypes];
    GetVictimMasks(pos, victimMasks);

    uint32_t numRecords = 0;
    for (Color color = 0; color < 2; ++color)
    {
        const SidePosition& side = pos.GetSide(color);
        for (uint32_t p = (uint32_t)Piece::Pawn; p < (uint32_t)Piece::King; ++p)
        {
            const Piece piece = (Piece)p;
            const uint8_t attackerCode = ThreatPieceCode(piece, color);

            side.GetPieceBitBoard(piece).Iterate([&](const Square from) INLINE_LAMBDA
            {
                const Bitboard targets = GetPieceAttacks(piece, color, from, occupied) & victimMasks[p];
                targets.Iterate([&](const Square to) INLINE_LAMBDA
                {
                    ASSERT(numRecords < nn::MaxThreatRecords);
                    outRecords[numRecords++] = { attackerCode, from.Index(), GetPieceCodeAt(pos, to), to.Index() };
                });
            });
        }
    }
    return numRecords;
}

void GetThreatDeltas(const Position& before, const Position& after, ThreatDeltas& outDeltas)
{
    outDeltas.numAdded = 0;
    outDeltas.numRemoved = 0;

    Bitboard changed = 0;
    for (Color color = 0; color < 2; ++color)
    {
        for (uint32_t p = (uint32_t)Piece::Pawn; p <= (uint32_t)Piece::King; ++p)
        {
            changed |= before.GetSide(color).GetPieceBitBoard((Piece)p) ^ after.GetSide(color).GetPieceBitBoard((Piece)p);
        }
    }
    ASSERT(changed.Count() <= 4);

    if (!changed)
        return;

    AppendChangedSquareThreats(before, changed, outDeltas.removed, outDeltas.numRemoved);
    AppendChangedSquareThreats(after, changed, outDeltas.added, outDeltas.numAdded);
    AppendSliderThreatChanges(before, after, changed, outDeltas);
}

uint32_t PositionToThreatFeatures(const Position& pos, const Color perspective, uint16_t* outFeatures)
{
    ThreatRecord records[nn::MaxThreatRecords];
    const uint32_t numRecords = GetThreatRecords(pos, records);
    return ThreatRecordsToFeatures(records, numRecords, GetThreatOrientation(pos, perspective), outFeatures);
}

bool ValidateThreatDeltas(const Position& before, const Position& after, const Color perspective)
{
    // a mirror change refreshes all features instead
    if (IsThreatMirrored(before, perspective) != IsThreatMirrored(after, perspective))
        return true;

    uint16_t featuresBefore[nn::MaxThreatRecords];
    uint16_t featuresAfter[nn::MaxThreatRecords];
    const uint32_t numBefore = PositionToThreatFeatures(before, perspective, featuresBefore);
    const uint32_t numAfter = PositionToThreatFeatures(after, perspective, featuresAfter);

    const auto less = [](const uint16_t x, const uint16_t y) { return x < y; };
    Sort(featuresBefore, featuresBefore + numBefore, less);
    Sort(featuresAfter, featuresAfter + numAfter, less);

    uint16_t expectedAdded[nn::MaxThreatRecords];
    uint16_t expectedRemoved[nn::MaxThreatRecords];
    const uint32_t numExpectedAdded = SubtractSortedFeatures(featuresAfter, numAfter, featuresBefore, numBefore, expectedAdded);
    const uint32_t numExpectedRemoved = SubtractSortedFeatures(featuresBefore, numBefore, featuresAfter, numAfter, expectedRemoved);

    ThreatDeltas deltas;
    GetThreatDeltas(before, after, deltas);

    const ThreatOrientation orientation = GetThreatOrientation(after, perspective);
    uint16_t added[nn::MaxChangedThreats];
    uint16_t removed[nn::MaxChangedThreats];
    const uint32_t numAdded = ThreatRecordsToFeatures(deltas.added, deltas.numAdded, orientation, added);
    const uint32_t numRemoved = ThreatRecordsToFeatures(deltas.removed, deltas.numRemoved, orientation, removed);

    return
        EqualFeatures(added, numAdded, expectedAdded, numExpectedAdded) &&
        EqualFeatures(removed, numRemoved, expectedRemoved, numExpectedRemoved);
}
