#ifndef CONSTS_HPP
#define CONSTS_HPP

namespace dataset {

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
} // namespace dataset

#endif
