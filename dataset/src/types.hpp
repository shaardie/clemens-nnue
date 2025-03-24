#ifndef CONSTS_HPP
#define CONSTS_HPP

#include <cassert>
#include <cstdint>
#include <ostream>
#include <sys/types.h>

namespace types {

constexpr int MAX_ACTIVE_FEATURES = 32;

enum PieceType {
  PAWN,
  KNIGHT,
  BISHOP,
  ROOK,
  QUEEN,
  KING,
};

enum Color { WHITE, BLACK };

enum Square : int {
  SQ_A1,
  SQ_B1,
  SQ_C1,
  SQ_D1,
  SQ_E1,
  SQ_F1,
  SQ_G1,
  SQ_H1,
  SQ_A2,
  SQ_B2,
  SQ_C2,
  SQ_D2,
  SQ_E2,
  SQ_F2,
  SQ_G2,
  SQ_H2,
  SQ_A3,
  SQ_B3,
  SQ_C3,
  SQ_D3,
  SQ_E3,
  SQ_F3,
  SQ_G3,
  SQ_H3,
  SQ_A4,
  SQ_B4,
  SQ_C4,
  SQ_D4,
  SQ_E4,
  SQ_F4,
  SQ_G4,
  SQ_H4,
  SQ_A5,
  SQ_B5,
  SQ_C5,
  SQ_D5,
  SQ_E5,
  SQ_F5,
  SQ_G5,
  SQ_H5,
  SQ_A6,
  SQ_B6,
  SQ_C6,
  SQ_D6,
  SQ_E6,
  SQ_F6,
  SQ_G6,
  SQ_H6,
  SQ_A7,
  SQ_B7,
  SQ_C7,
  SQ_D7,
  SQ_E7,
  SQ_F7,
  SQ_G7,
  SQ_H7,
  SQ_A8,
  SQ_B8,
  SQ_C8,
  SQ_D8,
  SQ_E8,
  SQ_F8,
  SQ_G8,
  SQ_H8,
  SQ_NONE,

  SQUARE_ZERO = 0,
  SQUARE_NB = 64
};

inline Square &operator+=(Square &sq, int offset) {
  return sq = static_cast<Square>(static_cast<int>(sq) + offset);
}

enum Direction : int {
  NORTH = 8,
  EAST = 1,
  SOUTH = -NORTH,
  WEST = -EAST,

  NORTH_EAST = NORTH + EAST,
  SOUTH_EAST = SOUTH + EAST,
  SOUTH_WEST = SOUTH + WEST,
  NORTH_WEST = NORTH + WEST
};

struct Piece {
  Square square;
  PieceType type;
  Color color;
  friend std::ostream &operator<<(std::ostream &os, const Piece &p) {
    os << "(" << p.square << "," << p.type << "," << p.color << ")";
    return os;
  }
};
struct Pos {
private:
  std::uint8_t turn;   // 0=WHITE, 1=BLACK
  std::uint8_t rule50; // half-move clock for 50-move rule
  std::int16_t score;  // score in cp; mating scores INT16_MAX - dtm; mated
                       // scores INT16_MIN + dtm
  std::uint8_t result; // 0=loss, 1=draw, 2=win
  std::uint8_t number_of_pieces;

public:
  Piece pieces[32];

  void set_turn(std::uint8_t t) {
    assert(t <= 1);
    turn = t;
  };
  std::uint8_t get_turn() const { return turn; };

  void set_rule50(std::uint8_t r) {
    assert(r <= 100);
    rule50 = r;
  };
  std::uint8_t get_rule50() const { return rule50; };

  void set_number_of_pieces(std::uint8_t n) {
    assert(n <= 32);
    number_of_pieces = n;
  };
  std::uint8_t get_number_of_pieces() const { return number_of_pieces; };

  void set_score(std::int16_t s) { score = s; };
  std::int16_t get_score() const { return score; };

  void set_result(std::uint8_t r) {
    assert(r <= 2);
    result = r;
  };
  std::uint8_t get_result() const { return result; };

  Pos() = default;
  ~Pos() = default;

  friend std::ostream &operator<<(std::ostream &os, const Pos &pos) {
    os << "Turn: " << (pos.turn == 0 ? "White" : "Black") << "\n";
    os << "50-move rule: " << (int)pos.rule50 << "\n";
    os << "Score: " << pos.score << " cp\n";
    os << "Result: "
       << (pos.result == 0 ? "Loss" : (pos.result == 1 ? "Draw" : "Win"))
       << "\n";
    os << "Number of pieces: " << (int)pos.number_of_pieces << "\n";

    os << "Pieces: \n";
    for (int i = 0; i < pos.number_of_pieces; ++i) {
      os << "  Piece " << i + 1 << ": " << pos.pieces[i] << "\n";
    }

    return os;
  }
};

} // namespace types

#endif
