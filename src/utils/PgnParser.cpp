#include "PgnParser.hpp"

#include "../backend/Move.hpp"
#include "../backend/Position.hpp"
#include "../backend/Score.hpp"

#include <fstream>
#include <algorithm>
#include <cstring>

static constexpr size_t kPgnReadBufferSize = 65536;

// Buffered reader
struct PgnReader
{
    explicit PgnReader(std::istream& s) : stream(s) {}

    std::istream& stream;
    char buf[kPgnReadBufferSize];
    size_t pos = 0;
    size_t end = 0;

    bool refill()
    {
        stream.read(buf, kPgnReadBufferSize);
        end = static_cast<size_t>(stream.gcount());
        pos = 0;
        return end > 0;
    }

    // Returns current char without consuming; '\0' at EOF.
    char peek()
    {
        while (pos >= end)
        {
            if (!refill()) return '\0';
        }
        return buf[pos];
    }

    // Consumes and returns current char; '\0' at EOF.
    char get()
    {
        const char c = peek();
        if (c) ++pos;
        return c;
    }

    bool atEof() { return peek() == '\0'; }
};

// Score parsing

// Parse the first token of a PGN comment as a score in centipawns.
// Handles: "+1.24", "-1.08", "+M5", "-M3", "1.08", "+1.50/0" (strips /...).
static bool ParseCommentScore(const char* token, size_t len, ScoreType& outScore)
{
    if (len == 0) return false;

    size_t i = 0;
    int sign = 1;

    if (token[i] == '+')      { ++i; }
    else if (token[i] == '-') { sign = -1; ++i; }

    if (i >= len) return false;

    // Mate score: M<N>
    if (token[i] == 'M' || token[i] == 'm')
    {
        ++i;
        int mateIn = 0;
        while (i < len && token[i] >= '0' && token[i] <= '9')
            mateIn = mateIn * 10 + (token[i++] - '0');
        if (mateIn <= 0) return false;
        // Matches ScoreToStr inverse: +MN → CheckmateValue-2N+1, -MN → -CheckmateValue+2N-1
        outScore = (sign > 0)
            ? ScoreType(CheckmateValue - 2 * mateIn + 1)
            : ScoreType(-CheckmateValue + 2 * mateIn - 1);
        return true;
    }

    // Numeric score
    if (token[i] < '0' || token[i] > '9') return false;

    int32_t intPart = 0;
    while (i < len && token[i] >= '0' && token[i] <= '9')
        intPart = intPart * 10 + (token[i++] - '0');

    int32_t fracCp = 0;
    if (i < len && token[i] == '.')
    {
        ++i;
        // Support up to 2 decimal places → centipawn precision
        if (i < len && token[i] >= '0' && token[i] <= '9')
            fracCp += (token[i++] - '0') * 10;
        if (i < len && token[i] >= '0' && token[i] <= '9')
            fracCp += (token[i++] - '0');
    }

    const int32_t total = sign * (intPart * 100 + fracCp);
    outScore = ScoreType(std::max<int32_t>(-32767, std::min<int32_t>(32767, total)));
    return true;
}

// Main parser implementation
class PgnParserImpl
{
public:
    explicit PgnParserImpl(std::istream& s) : reader(s) {}

    uint64_t readGames(const std::function<bool(Game&)>& cb)
    {
        callback = &cb;
        numParsed = 0;

        if (!reader.refill()) return 0;

        while (!reader.atEof())
        {
            if (inHeader)
            {
                if (reader.peek() == '[')
                {
                    processHeader();
                }
            }
            else if (inBody)
            {
                processBody();
            }

            if (!dontAdvanceAfterBody) reader.get();
            dontAdvanceAfterBody = false;
        }

        // Finish last game if body was still open
        if (!gameEnded)
            onGameEnd();

        return numParsed;
    }

private:
    // Header processing
    void processHeader()
    {
        // Entry: peek() == '['. Called once per game; loops until body transition.
        // Design: parseOneTag() consumes the trailing '\n' of each tag line.
        // So when this loop sees '\n', it is the blank line separating headers from body.
        // We do NOT consume that '\n' — the outer loop's unconditional advance does it.
        while (!reader.atEof())
        {
            const char c = reader.peek();

            if (c == '[')
            {
                reader.get(); // consume '['
                parseOneTag(); // consumes everything up to and including the tag's trailing '\n'
            }
            else if (c == '\n')
            {
                // Blank separator line — transition to body. Leave '\n' for outer loop to advance.
                inHeader = false;
                inBody = true;
                beginGame();
                return;
            }
            else if (c == '\r')
            {
                reader.get();
            }
            else
            {
                // Non-tag content after headers (no blank line) — switch to body.
                // Set dontAdvanceAfterBody so outer loop doesn't consume this char.
                inHeader = false;
                inBody = true;
                dontAdvanceAfterBody = true;
                beginGame();
                return;
            }
        }
    }

    void parseOneTag()
    {
        // Read tag name
        tagKey.clear();
        while (!reader.atEof())
        {
            const char c = reader.peek();
            if (c == ' ' || c == '\t' || c == '"' || c == ']' || c == '\n') break;
            tagKey += reader.get();
        }

        // Skip whitespace before '"'
        while (!reader.atEof() && (reader.peek() == ' ' || reader.peek() == '\t'))
            reader.get();

        // Read tag value between quotes
        tagVal.clear();
        if (reader.peek() == '"')
        {
            reader.get(); // consume '"'
            bool esc = false;
            while (!reader.atEof())
            {
                const char ch = reader.get();
                if (ch == '\n') break; // malformed
                if (esc) { tagVal += ch; esc = false; }
                else if (ch == '\\') esc = true;
                else if (ch == '"') break;
                else tagVal += ch;
            }
        }

        // Skip to end of line and consume the '\n' so processHeader() sees clean state
        while (!reader.atEof())
        {
            const char ch = reader.get();
            if (ch == '\n') break;
        }

        // Store relevant tags
        if (tagKey == "FEN")
        {
            pendingFen = tagVal;
        }
        else if (tagKey == "Result")
        {
            if (tagVal == "1-0")        pendingResult = Game::Score::WhiteWins;
            else if (tagVal == "0-1")   pendingResult = Game::Score::BlackWins;
            else if (tagVal == "1/2-1/2") pendingResult = Game::Score::Draw;
            else                        pendingResult = Game::Score::Unknown;
        }
        else if (tagKey == "Round")
        {
            pendingRound = 0;
            for (const char ch : tagVal)
            {
                if (ch < '0' || ch > '9') break;
                pendingRound = pendingRound * 10 + static_cast<uint32_t>(ch - '0');
            }
        }
    }

    // Body (moves) processing
    void processBody()
    {
        while (!reader.atEof())
        {
            const char c = reader.peek();

            // Next game starts
            if (c == '[')
            {
                onGameEnd();
                dontAdvanceAfterBody = true;
                return;
            }

            // Whitespace
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r')
            {
                reader.get();
                continue;
            }

            // Skip dots (move number suffix or "..." for black)
            if (c == '.')
            {
                while (!reader.atEof() && reader.peek() == '.')
                    reader.get();
                continue;
            }

            // Digit — could be move number, "1-0", "0-1", "1/2-1/2", or "0-0" castling
            if (c >= '0' && c <= '9')
            {
                if (handleDigitToken()) return;
                continue;
            }

            // Game termination: '*'
            if (c == '*')
            {
                reader.get();
                onGameEnd();
                return;
            }

            // Skip variations: ( ... )
            if (c == '(')
            {
                reader.get();
                skipVariation();
                continue;
            }

            // Skip NAG: $n
            if (c == '$')
            {
                reader.get();
                while (!reader.atEof() && reader.peek() >= '0' && reader.peek() <= '9')
                    reader.get();
                continue;
            }

            // Comment: { ... }
            if (c == '{')
            {
                reader.get();
                ScoreType s = 0;
                if (readCommentAndExtractScore(s))
                    pendingScore = s;
                continue;
            }

            // Move token (SAN move or termination like "O-O")
            if (readMoveToken()) return;
        }

        // EOF while in body
        onGameEnd();
    }

    // Handle a token that starts with a digit.
    // Returns true if the game ended (termination marker found).
    bool handleDigitToken()
    {
        char tmp[16];
        size_t n = 0;
        while (!reader.atEof() && reader.peek() >= '0' && reader.peek() <= '9' && n < 15)
            tmp[n++] = reader.get();
        tmp[n] = '\0';

        const char next = reader.peek();

        // Move number: digits followed by dot(s) like "1." or "23..."
        if (next == '.')
        {
            while (!reader.atEof() && reader.peek() == '.')
                reader.get();
            return false;
        }

        if (next == '-' && n == 1)
        {
            reader.get(); // consume '-'
            if (tmp[0] == '1')
            {
                // "1-0"
                if (!reader.atEof() && reader.peek() == '0') reader.get();
                onGameEnd();
                return true;
            }
            if (tmp[0] == '0')
            {
                const char after = reader.peek();
                if (after == '1') { reader.get(); onGameEnd(); return true; } // "0-1"
                // "0-0" or "0-0-0": castling — assemble and apply as move
                moveStr = "0-";
                while (!reader.atEof())
                {
                    const char ch = reader.peek();
                    if (ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r' || ch == '{' || ch == '(' || ch == '$')
                        break;
                    moveStr += reader.get();
                }
                parseMoveToBuffer();
                return false;
            }
        }

        if (next == '/' && n == 1 && tmp[0] == '1')
        {
            // "1/2-1/2"
            for (int i = 0; i < 6 && !reader.atEof(); ++i) reader.get();
            onGameEnd();
            return true;
        }

        // Otherwise just a move number without dot — skip
        return false;
    }

    // Read one SAN move token. Returns true if game ended.
    bool readMoveToken()
    {
        moveStr.clear();

        while (!reader.atEof())
        {
            const char c = reader.peek();
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '{' || c == '(' || c == '$')
                break;
            moveStr += reader.get();
        }

        if (moveStr.empty()) return false;

        parseMoveToBuffer();
        return false;
    }

    // Parse the move string and buffer it; flush the previous buffered move to the game first.
    // Comments come AFTER the move in PGN, so the flow is:
    //   move → comment (score for that move) → next move → ...
    // We flush each buffered move once we see the next move (or game end), by which time
    // the comment score has been read into pendingScore.
    void parseMoveToBuffer()
    {
        if (gameError || !gameActive) return;

        // Flush the previously buffered move (now we have its comment score)
        flushBufferedMove();

        // Parse the new move against current board position
        const bool isWhite = (currentPos.GetSideToMove() == White);
        const Move move = currentPos.MoveFromString(moveStr, MoveNotation::SAN);
        if (!move.IsValid())
        {
            gameError = true;
            return;
        }

        // Advance the position immediately so subsequent SAN moves parse correctly
        currentPos.DoMove(move);

        // Buffer this move; its score will arrive in the next comment
        bufferedMove = move;
        bufferedMoveIsWhite = isWhite;
        hasPendingMove = true;
        pendingScore = kNoScore;
    }

    // Commit the buffered move (with its score) to the Game object.
    void flushBufferedMove()
    {
        if (!hasPendingMove) return;
        hasPendingMove = false;

        if (pendingScore != kNoScore)
        {
            // Scores in PGN comments are from the side-to-move's perspective.
            // mMoveScores stores them from White's perspective.
            const ScoreType stored = bufferedMoveIsWhite ? pendingScore : ScoreType(-pendingScore);
            currentGame.DoMove(bufferedMove, stored);
        }
        else
        {
            currentGame.DoMove(bufferedMove);
        }
        pendingScore = kNoScore;
    }

    // Comment: read score from first token, discard rest
    bool readCommentAndExtractScore(ScoreType& outScore)
    {
        // Skip leading whitespace inside comment
        while (!reader.atEof() && reader.peek() != '}')
        {
            const char c = reader.peek();
            if (c != ' ' && c != '\t' && c != '\n' && c != '\r') break;
            reader.get();
        }

        // Read first token (until whitespace, '/', or '}')
        char tokenBuf[64];
        size_t tokenLen = 0;

        while (!reader.atEof() && tokenLen < sizeof(tokenBuf) - 1)
        {
            const char c = reader.peek();
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '/' || c == '}')
                break;
            tokenBuf[tokenLen++] = reader.get();
        }
        tokenBuf[tokenLen] = '\0';

        // Consume rest of comment until '}'
        while (!reader.atEof() && reader.get() != '}') {}

        return ParseCommentScore(tokenBuf, tokenLen, outScore);
    }

    // Skip variation: ( ... ) with nesting
    void skipVariation()
    {
        int depth = 1;
        char c;
        while ((c = reader.get()))
        {
            if (c == '(') ++depth;
            else if (c == ')') { if (--depth == 0) break; }
        }
    }

    // Game lifecycle
    void beginGame()
    {
        gameError = false;
        gameActive = false;
        gameEnded = false;
        hasPendingMove = false;
        pendingScore = kNoScore;

        const char* fen = pendingFen.empty() ? Position::InitPositionFEN : pendingFen.c_str();
        if (!currentPos.FromFEN(fen))
        {
            gameError = true;
            return;
        }

        currentGame.Reset(currentPos);
        GameMetadata meta;
        meta.roundNumber = pendingRound;
        currentGame.SetMetadata(meta);
        gameActive = true;
    }

    void onGameEnd()
    {
        gameEnded = true;
        inHeader = true;
        inBody = false;

        // Flush the last buffered move before ending the game
        if (!gameError && gameActive)
            flushBufferedMove();

        // Capture result before resetting per-game state
        const Game::Score capturedResult = pendingResult;
        resetPerGameState();

        if (gameError || !gameActive) return;

        // Apply forced result from Result header if position has no natural result
        if (currentGame.GetScore() == Game::Score::Unknown && capturedResult != Game::Score::Unknown)
            currentGame.SetScore(capturedResult);

        ++numParsed;
        if (!(*callback)(currentGame))
        {
            // Caller wants to stop — drain the stream
            while (!reader.atEof()) reader.get();
        }
    }

    void resetPerGameState()
    {
        pendingFen.clear();
        pendingResult = Game::Score::Unknown;
        pendingRound = 1;
    }

    PgnReader reader;
    const std::function<bool(Game&)>* callback = nullptr;
    uint64_t numParsed = 0;

    static constexpr ScoreType kNoScore = std::numeric_limits<ScoreType>::min();

    // Parser state
    bool inHeader = true;
    bool inBody = false;
    bool gameEnded = true;
    bool dontAdvanceAfterBody = false;

    // Accumulated from headers (valid between tags and beginGame())
    std::string pendingFen;
    Game::Score pendingResult = Game::Score::Unknown;
    uint32_t pendingRound = 1;

    // Per-game
    Game currentGame;
    Position currentPos;
    bool gameActive = false;
    bool gameError = false;

    // Buffered move: parsed but not yet committed to currentGame (waiting for its comment score)
    Move bufferedMove;
    bool hasPendingMove = false;
    bool bufferedMoveIsWhite = false;
    ScoreType pendingScore = kNoScore;

    // Reused buffers
    std::string tagKey;
    std::string tagVal;
    std::string moveStr;
};

// ---------------------------------------------------------------------------

uint64_t ParsePgn(std::istream& stream, const std::function<bool(Game&)>& callback)
{
    PgnParserImpl parser(stream);
    return parser.readGames(callback);
}

uint64_t ParsePgn(const std::string& path, const std::function<bool(Game&)>& callback)
{
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) return 0;
    return ParsePgn(file, callback);
}
