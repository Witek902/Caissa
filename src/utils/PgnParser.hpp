#pragma once

#include "../backend/Game.hpp"

#include <functional>
#include <istream>
#include <string>

// Parse PGN from stream. Callback is invoked for each successfully parsed game.
// Return false from callback to stop parsing early.
// Returns number of games successfully parsed.
uint64_t ParsePgn(std::istream& stream, const std::function<bool(Game&)>& callback);
uint64_t ParsePgn(const std::string& path, const std::function<bool(Game&)>& callback);
