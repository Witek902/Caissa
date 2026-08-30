#pragma once

#include "../backend/Position.hpp"

#include <memory>

// Plays self-play games from a fixed root position in order to measure search quality
// at a given nodes-per-move limit. Games are played to a real terminal state, without
// any eval- or tablebase-based adjudication.
class PlayoutRunner
{
public:
    PlayoutRunner();
    ~PlayoutRunner();

    // start playing games in the background, returns false if the root position is already terminal
    bool Start(const Position& rootPosition, uint64_t softNodeLimit, uint64_t maxGames);

    // abort all games in progress and print the final report
    void Stop();

    bool IsRunning() const;

private:
    PlayoutRunner(const PlayoutRunner&) = delete;
    PlayoutRunner& operator = (const PlayoutRunner&) = delete;

    struct Impl;
    std::unique_ptr<Impl> mImpl;
};
