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
