#ifndef C_CHESS_CLI_HPP
#define C_CHESS_CLI_HPP

#include <cassert>
#include <cstdint>
#include <fstream>
#include <sys/types.h>

namespace c_chess_cli {
enum PieceType {
  KNIGHT,
  BISHOP,
  ROOK,
  QUEEN,
  KING,
  PAWN,
  ROOK_WITH_CASTLING_RIGHT,
  PAWN_CAPTURABLE_ENPASSANT
};

enum Color { WHITE, BLACK };

struct Piece {
  std::uint8_t square;
  PieceType type;
  Color color;
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
  std::uint8_t get_rule50() const { return turn; };

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
};

struct PosReader {
public:
  virtual ~PosReader() = default;
  virtual Pos read_pos() = 0;
};

struct BinPosReader : public PosReader {
private:
  std::ifstream stream;
  void unpack_packed_pieces(Pos &pos, u_int64_t occ,
                            std::uint8_t packed_pieces[16]);

public:
  BinPosReader(std::string filename);
  ~BinPosReader();

  Pos read_pos();
};

struct CSVPosReader : public PosReader {};

}; // namespace c_chess_cli

#endif
