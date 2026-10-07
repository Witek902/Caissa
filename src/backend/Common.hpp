#pragma once

#include <cstdint>

#if defined(_WIN32)
    #define PLATFORM_WINDOWS
#elif defined(__GNUC__) || defined(__clang__)
    #define PLATFORM_LINUX
#endif

#if defined(PLATFORM_LINUX)
    #include <csignal>
#endif

#if defined(PLATFORM_WINDOWS)
    #define DEBUG_BREAK() __debugbreak()
#elif defined(PLATFORM_LINUX)
    #define DEBUG_BREAK() std::raise(SIGINT)
#endif

#define UNUSED(x) (void)(x)

#ifndef CONFIGURATION_FINAL
    void AssertionFailed(const char* expression, const char* file, int line);
    #define ASSERT(x) do { if (!(x)) [[unlikely]] { AssertionFailed(#x, __FILE__, __LINE__); DEBUG_BREAK(); } } while (0)
    #define VERIFY(x) do { if (!(x)) [[unlikely]] { AssertionFailed(#x, __FILE__, __LINE__); DEBUG_BREAK(); } } while (0)
#else
    #define ASSERT(x) do { } while (0)
    #define VERIFY(x) (x)
#endif

#define CACHELINE_SIZE 64u

#define USE_SYZYGY_TABLEBASES
// #define USE_GAVIOTA_TABLEBASES

#if defined(_MSC_VER) && !defined(__clang__)

    // "C++ nonstandard extension: nameless struct"
    #pragma warning(disable : 4201)

    // "unreferenced local function"
    #pragma warning(disable : 4505)

    // "structure was padded due to alignment specifier"
    #pragma warning(disable : 4324)

    #define INLINE __forceinline
    #define INLINE_LAMBDA [[msvc::forceinline]]
    #define NO_INLINE __declspec(noinline)

#elif defined(__GNUC__) || defined(__clang__)

    #define INLINE __attribute__((always_inline)) inline
    #define INLINE_LAMBDA
    #define NO_INLINE __attribute__((noinline))

#else //  defined(__GNUC__) || defined(__clang__)

#error "Unsupported compiler"

#endif

#if defined(__GNUC__)
    #define UNNAMED_STRUCT __extension__
#else
    #define UNNAMED_STRUCT
#endif

#if defined(__GNUC__) || defined(__clang__)
    #define EXPORT __attribute__((visibility("default")))
#else
    #define EXPORT
#endif

union MaterialKey;
class Position;
struct TTEntry;
struct Move;
struct PackedMove;
class Game;
class TranspositionTable;
struct NodeInfo;
struct NNEvaluatorContext;

static constexpr uint32_t MaxAllowedMoves = 280;

template<uint32_t MaxSize> class TMoveList;
using MoveList = TMoveList<MaxAllowedMoves>;

using Color = uint8_t;
static constexpr Color White = 0;
static constexpr Color Black = 1;

using ScoreType = int16_t;

static constexpr ScoreType InfValue             = 32767;
static constexpr ScoreType InvalidValue         = INT16_MAX;
static constexpr ScoreType CheckmateValue       = 32000;
static constexpr ScoreType TablebaseWinValue    = 31000;
static constexpr ScoreType KnownWinValue        = 20000;

static constexpr uint16_t MaxSearchDepth    = 256;

static constexpr ScoreType DrawScoreRandomness = 2;

// initialize all engine subsystems
EXPORT void InitEngine();
