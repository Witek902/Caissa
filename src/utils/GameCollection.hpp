#pragma once

#include "Stream.hpp"

#include "../backend/PositionUtils.hpp"
#include "../backend/Move.hpp"
#include "../backend/Game.hpp"

#include <string>
#include <mutex>

namespace GameCollection
{
    struct Header
    {
        uint32_t magic;
    };

#pragma pack(push, 1)
    struct GameHeader
    {
        PackedPosition initialPosition;
        // Note: this is not final game score. It's to handle resignation, agreed draw, etc.
        Game::Score forcedScore;
        uint8_t hasMoveScores : 1;
        uint16_t numMoves;
    };
#pragma pack(pop)

    struct MoveAndScore
    {
        PackedMove move;
        int16_t score;
    };

    bool ReadGame(InputStream& stream, Game& game, std::vector<Move>& decodedMoves);

    class Writer
    {
    public:
        Writer(OutputStream& stream) : mStream(stream) { }

        bool WriteGame(const Game& game);
        bool IsOK() const { return mStream.IsOK(); }

    private:
        OutputStream& mStream;
        std::mutex mMutex;
    };

} // namespace GameCollection
