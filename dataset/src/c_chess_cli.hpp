#ifndef C_CHESS_CLI_HPP
#define C_CHESS_CLI_HPP

#include "types.hpp"
#include <cassert>
#include <cstdint>
#include <fstream>
#include <sys/types.h>

namespace c_chess_cli {

struct PosReader {
public:
  virtual ~PosReader() = default;
  virtual types::Pos read_pos() = 0;
};

struct BinPosReader : public PosReader {
private:
  std::ifstream stream;
  void unpack_packed_pieces(types::Pos &pos, u_int64_t occ,
                            std::uint8_t packed_pieces[16]);
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
  types::PieceType fromExtType(PieceType extPieceType);

public:
  BinPosReader(const std::string filename);
  ~BinPosReader();

  types::Pos read_pos();
};

struct CSVPosReader : public PosReader {
private:
  std::ifstream stream;

public:
  CSVPosReader(const std::string &filename);
  ~CSVPosReader();
  types::Pos read_pos();
};

}; // namespace c_chess_cli

#endif
