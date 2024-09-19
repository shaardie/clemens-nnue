#ifndef C_CHESS_CLI_HPP
#define C_CHESS_CLI_HPP

#include <cassert>
#include <cstdint>
#include <exception>
#include <fstream>

namespace c_chess_cli
{
    enum PieceType
    {
        KNIGHT,
        BISHOP,
        ROOK,
        QUEEN,
        KING,
        PAWN,
        ROOK_WITH_CASTLING_RIGHT,
        PAWN_CAPTURABLE_ENPASSANT
    };

    enum Color
    {
        WHITE,
        BLACK
    };

    struct Piece
    {
        std::uint8_t square;
        PieceType type;
        Color color;
    };

    struct Pos
    {
    private:
        std::uint8_t packed_pieces[16]; // 4 bits per piece, max 16 bytes
    public:
        std::uint64_t occ;   // occupied squares (bitboard)
        std::uint8_t turn;   // 0=WHITE, 1=BLACK
        std::uint8_t rule50; // half-move clock for 50-move rule
        // This is an upstream bug
        // upstream bug https://github.com/lucasart/c-chess-cli/issues/63
        // So size is different
        std::int32_t score;   // score in cp; mating scores INT16_MAX - dtm; mated scores INT16_MIN + dtm
        std::uint32_t result; // 0=loss, 1=draw, 2=win
        std::uint8_t number_of_pieces;
        Piece pieces[32];

    private:
        void unpack_packed_pieces();

    public:
        Pos(std::ifstream &stream);
        ~Pos();
    };
};

#endif