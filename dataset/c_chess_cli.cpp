#include <cassert>
#include <cstdint>
#include <fstream>

#include "c_chess_cli.hpp"

namespace c_chess_cli {

BinPosReader::BinPosReader(std::string filename) {
  stream.open(filename);
  if (!stream) {
    throw std::runtime_error("unable to open file");
  }
};
BinPosReader::~BinPosReader() { stream.close(); };

void BinPosReader::unpack_packed_pieces(Pos &pos, uint64_t occ,
                                        std::uint8_t packed_pieces[16]) {

  int i = 0;
  while (occ) {
    int packed_piece = i % 2 ? packed_pieces[(i + 1) / 2] >> 4
                             : packed_pieces[(i + 1) / 2] & 0x0F;
    Piece *piece = &pos.pieces[i];

    // get square via builtin least significat bit function
    piece->square = __builtin_ctzll(occ & -occ);

    // reduce occupation
    occ &= occ - 1;

    // get piece and get proper piece type and color from it
    piece->type = c_chess_cli::PieceType((packed_piece & 0xFE) / 2);
    piece->color = c_chess_cli::Color(packed_piece & 1);
    i++;
  }
}

Pos BinPosReader::read_pos() {
  Pos pos;

  // read occupation
  std::uint64_t occ; // occupied squares (bitboard)
  stream.read(reinterpret_cast<char *>(&occ), sizeof(occ));
  if (stream.gcount() != sizeof(occ)) {
    throw std::runtime_error("unable to read occ");
  }

  // read turn and rule50
  std::uint8_t turn_and_rule50;
  stream.read(reinterpret_cast<char *>(&turn_and_rule50),
              sizeof(turn_and_rule50));
  if (stream.gcount() != sizeof(turn_and_rule50)) {
    throw std::runtime_error("unable to read turn and rule50");
  }
  pos.set_turn(turn_and_rule50 & 1);
  pos.set_rule50(turn_and_rule50 >> 1);

  // calculate number of pieces (number of 1s)
  pos.set_number_of_pieces(__builtin_popcountll(occ));

  // read packed pieces
  std::uint8_t packed_pieces[16]; // 4 bits per piece, max 16 bytes
  int packed_pieces_size = (pos.get_number_of_pieces() + 1) / 2;
  stream.read(reinterpret_cast<char *>(packed_pieces), packed_pieces_size);
  if (stream.gcount() != packed_pieces_size) {
    throw std::runtime_error("unable to read packed pieces");
  }

  unpack_packed_pieces(pos, occ, packed_pieces);

  // This is an upstream bug
  // upstream bug https://github.com/lucasart/c-chess-cli/issues/63
  // So size is different
  std::int32_t score = 0;
  stream.read(reinterpret_cast<char *>(&score), sizeof(score));
  if (stream.gcount() != sizeof(score)) {
    throw std::runtime_error("unable to read score");
  }
  pos.set_score(static_cast<std::int16_t>(score));

  std::uint32_t result = 0;
  stream.read(reinterpret_cast<char *>(&result), sizeof(result));
  if (stream.gcount() != sizeof(result)) {
    throw std::runtime_error("unable to read result");
  }
  pos.set_result(static_cast<std::uint8_t>(result));
  return pos;
}
}; // namespace c_chess_cli
