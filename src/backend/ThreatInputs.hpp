#pragma once

#include "PackedNeuralNetwork.hpp"
#include "Position.hpp"

// neural net threat inputs: one feature per non-king piece attacking or defending another non-king piece,
// indexed by (attacker piece, attacker square, victim piece, victim square).
// Squares and colors are seen from a perspective, the same way as the piece-square inputs.

namespace nn {

// every non-king piece has at most 8 threats (one per ray or knight jump)
static constexpr uint32_t MaxThreatRecords = 8 * 62;

// A move changes at most 4 squares. Per changed square and sign, at most 8 threats from it, 16 threats
// onto it and 8 slider threats through it change.
static constexpr uint32_t MaxChangedThreats = 4 * (8 + 16 + 8);

} // namespace nn

INLINE constexpr uint8_t ThreatPieceCode(const Piece piece, const Color color)
{
    return static_cast<uint8_t>((color << 3) | static_cast<uint32_t>(piece));
}

// threat in board coordinates, the same for both perspectives
struct ThreatRecord
{
    uint8_t attacker;   // ThreatPieceCode
    uint8_t from;
    uint8_t victim;     // ThreatPieceCode
    uint8_t to;
};

// threat changes between two positions
struct ThreatDeltas
{
    uint32_t numAdded = 0;
    uint32_t numRemoved = 0;
    ThreatRecord added[nn::MaxChangedThreats];
    ThreatRecord removed[nn::MaxChangedThreats];
};

// Square and piece transform of a perspective:
// - rank flip for black
// - file flip when the perspective's king is on files e-h, piece colors relative to the perspective
struct ThreatOrientation
{
    uint8_t squareXor;
    uint8_t pieceXor;
};

INLINE bool IsThreatMirrored(const Position& pos, const Color perspective)
{
    return pos.GetSide(perspective).GetKingSquare().File() >= 4;
}

INLINE ThreatOrientation GetThreatOrientation(const Position& pos, const Color perspective)
{
    ThreatOrientation orientation;
    orientation.squareXor = static_cast<uint8_t>((perspective == Black ? 0b111000 : 0) | (IsThreatMirrored(pos, perspective) ? 0b000111 : 0));
    orientation.pieceXor = static_cast<uint8_t>(perspective << 3);
    return orientation;
}

// called on engine init
void InitThreatInputs();

// all threats of a position, returns their number
uint32_t GetThreatRecords(const Position& pos, ThreatRecord* outRecords);

// threats present in only one of two positions that differ by a single move
void GetThreatDeltas(const Position& before, const Position& after, ThreatDeltas& outDeltas);

// feature index of a threat; NumThreatInputs or more if the threat is not a feature in this orientation
uint32_t GetThreatFeatureIndex(const ThreatRecord& record, const ThreatOrientation orientation);

// feature indices of the threats that are features; outFeatures must have room for numRecords entries
uint32_t ThreatRecordsToFeatures(const ThreatRecord* records, uint32_t numRecords, const ThreatOrientation orientation, uint16_t* outFeatures);

// the same for both perspectives in one pass over the records; orientations and outputs are indexed by color
void ThreatRecordsToFeatures(const ThreatRecord* records, uint32_t numRecords, const ThreatOrientation orientations[2],
    uint16_t* outWhiteFeatures, uint16_t* outBlackFeatures, uint32_t outNumFeatures[2]);

// threat features of a position; outFeatures must have room for MaxThreatRecords entries
uint32_t PositionToThreatFeatures(const Position& pos, const Color perspective, uint16_t* outFeatures);

// checks that the threat deltas of a move match the difference of the full feature sets (debugging)
bool ValidateThreatDeltas(const Position& before, const Position& after, const Color perspective);
