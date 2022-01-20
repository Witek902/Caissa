#include "Material.hpp"

static_assert(sizeof(MaterialKey) == sizeof(uint64_t), "Invalid material key size");

std::string MaterialKey::ToString() const
{
	std::string str = "K";

	if (numWhiteQueens)		str += std::string(numWhiteQueens, 'Q');
	if (numWhiteRooks)		str += std::string(numWhiteRooks, 'R');
	if (numWhiteBishops)	str += std::string(numWhiteBishops, 'B');
	if (numWhiteKnights)	str += std::string(numWhiteKnights, 'N');
	if (numWhitePawns)		str += std::string(numWhitePawns, 'P');

	str += "vK";

	if (numBlackQueens)		str += std::string(numBlackQueens, 'Q');
	if (numBlackRooks)		str += std::string(numBlackRooks, 'R');
	if (numBlackBishops)	str += std::string(numBlackBishops, 'B');
	if (numBlackKnights)	str += std::string(numBlackKnights, 'N');
	if (numBlackPawns)		str += std::string(numBlackPawns, 'P');

	return str;
}

uint32_t MaterialKey::GetNeuralNetworkInputsNumber() const
{
    uint32_t inputs = 0;

    if (numWhitePawns > 0 || numBlackPawns > 0)
    {
        // has pawns, so can't exploit vertical/diagonal symmetry
        inputs += 32; // white king on left files
        inputs += 64; // black king on any file
    }
    else
    {
        // pawnless position, can exploit vertical/horizonal/diagonal symmetry
        inputs += 16; // white king on files A-D, ranks 1-4
        inputs += 36; // black king on bottom-right triangle (a1, b1, b2, c1, c2, c3, ...)
    }

    // knights/bishops/rooks/queens on any square
    if (numWhiteQueens)     inputs += 64;
    if (numBlackQueens)     inputs += 64;
    if (numWhiteRooks)      inputs += 64;
    if (numBlackRooks)      inputs += 64;
    if (numWhiteBishops)    inputs += 64;
    if (numBlackBishops)    inputs += 64;
    if (numWhiteKnights)    inputs += 64;
    if (numBlackKnights)    inputs += 64;

    // pawns on ranks 2-7
    if (numWhitePawns)      inputs += 48;
    if (numBlackPawns)      inputs += 48;

    return inputs;
}
