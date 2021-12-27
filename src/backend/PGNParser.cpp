#include "PGNParser.hpp"

#include <iostream>

PGNParser::PGNParser(std::istream& stream)
	: mStream(stream)
{
}

bool PGNParser::GetGame(PGNGame& outGame)
{
	if (!ParseTags(outGame))
	{
		std::cerr << "Failed to parse tags from PGN" << std::endl;
		return false;
	}

	if (!ParseMoves(outGame))
	{
		std::cerr << "Failed to parse moves from PGN" << std::endl;
		return false;
	}

	return true;
}

void PGNParser::SkipBlank()
{
    while (!mStream.eof())
    {
        const auto next = mStream.get();

        if (!isspace(next))
        {
            mStream.unget();
            return;
        }
    }
}

bool PGNParser::ParseTags(PGNGame& outGame)
{
	// TODO
	(void)outGame;

	SkipBlank();

	if (!mStream.eof())
	{

	}

	return false;
}

bool PGNParser::ParseMoves(PGNGame& outGame)
{
	// TODO
	(void)outGame;

	return false;
}