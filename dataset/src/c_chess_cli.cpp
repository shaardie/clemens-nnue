#include <cassert>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "c_chess_cli.hpp"
#include "types.hpp"

namespace c_chess_cli {

BinPosReader::BinPosReader(const std::string filename) {
  stream.open(filename);
  if (!stream) {
    throw std::runtime_error("unable to open file");
  }
};
BinPosReader::~BinPosReader() { stream.close(); };

types::PieceType BinPosReader::fromExtType(PieceType extPieceType) {
  switch (extPieceType) {
  case PAWN:
    return types::PAWN;
  case PAWN_CAPTURABLE_ENPASSANT:
    return types::PAWN;
  case KNIGHT:
    return types::KNIGHT;
  case ROOK:
    return types::ROOK;
  case ROOK_WITH_CASTLING_RIGHT:
    return types::ROOK;
  case QUEEN:
    return types::QUEEN;
  case KING:
    return types::KING;
  case BISHOP:
    return types::BISHOP;
  default:
    throw std::runtime_error("unable to open file");
  }
}
void BinPosReader::unpack_packed_pieces(types::Pos &pos, uint64_t occ,
                                        std::uint8_t packed_pieces[16]) {

  int i = 0;
  while (occ) {
    int packed_piece = i % 2 ? packed_pieces[(i + 1) / 2] >> 4
                             : packed_pieces[(i + 1) / 2] & 0x0F;
    types::Piece *piece = &pos.pieces[i];

    // get square via builtin least significat bit function
    piece->square = static_cast<types::Square>(__builtin_ctzll(occ & -occ));

    // reduce occupation
    occ &= occ - 1;

    // get piece and get proper piece type and color from it
    piece->type = fromExtType(PieceType((packed_piece & 0xFE) / 2));
    piece->color = types::Color(packed_piece & 1);
    i++;
  }
}

types::Pos BinPosReader::read_pos() {
  types::Pos pos;

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

CSVPosReader::CSVPosReader(const std::string &filename) {
  stream.open(filename);
  if (!stream) {
    throw std::runtime_error("unable to open file");
  }
};

CSVPosReader::~CSVPosReader() { stream.close(); };

types::Pos CSVPosReader::read_pos() {
  std::string line;
  if (!std::getline(stream, line)) {
    throw std::runtime_error("unable to read line");
  }
  std::istringstream iss_line(line);
  std::string fen, eval_str, result_str;
  if (!std::getline(iss_line, fen, ',') ||
      !std::getline(iss_line, eval_str, ',') ||
      !std::getline(iss_line, result_str, ',')) {
    throw std::runtime_error("Error reading CSV line");
  }
  types::Pos pos;
  pos.set_score(static_cast<std::int16_t>(std::stoi(eval_str)));
  pos.set_result(static_cast<std::uint8_t>(std::stoi(result_str)));

  std::istringstream iss_fen(fen);
  std::string piece_str, turn_str, rochade_str, en_passant_str, rule50_str,
      halfmove_str;
  if (!(iss_fen >> piece_str >> turn_str >> rochade_str >> en_passant_str >>
        rule50_str >> halfmove_str)) {
    throw std::runtime_error("unable to split fen");
  }

  // Pieces
  std::istringstream iss_pieces(piece_str);
  unsigned char token;
  types::Square square = types::SQ_A8;
  types::Piece *piece;
  pos.set_number_of_pieces(0);
  while (iss_pieces >> token) {
    if (std::isdigit(token)) {
      square += (token - '0') * types::EAST;
    } else if (token == '/') {
      square += 2 * types::SOUTH;
    } else {
      piece = &pos.pieces[pos.get_number_of_pieces()];
      pos.set_number_of_pieces(pos.get_number_of_pieces() + 1);
      piece->square = square;
      square += 1;
      switch (token) {
      case 'r':
        piece->color = types::BLACK;
        piece->type = types::ROOK;
        break;
      case 'n':
        piece->color = types::BLACK;
        piece->type = types::KNIGHT;
        break;
      case 'b':
        piece->color = types::BLACK;
        piece->type = types::BISHOP;
        break;
      case 'q':
        piece->color = types::BLACK;
        piece->type = types::QUEEN;
        break;
      case 'k':
        piece->color = types::BLACK;
        piece->type = types::KING;
        break;
      case 'p':
        piece->color = types::BLACK;
        piece->type = types::PAWN;
        break;
      case 'R':
        piece->color = types::WHITE;
        piece->type = types::ROOK;
        break;
      case 'N':
        piece->color = types::WHITE;
        piece->type = types::KNIGHT;
        break;
      case 'B':
        piece->color = types::WHITE;
        piece->type = types::BISHOP;
        break;
      case 'Q':
        piece->color = types::WHITE;
        piece->type = types::QUEEN;
        break;
      case 'K':
        piece->color = types::WHITE;
        piece->type = types::KING;
        break;
      case 'P':
        piece->color = types::WHITE;
        piece->type = types::PAWN;
        break;
      }
    }
  }

  if (turn_str == "w") {
    pos.set_turn(types::WHITE);
  } else if (turn_str == "b") {
    pos.set_turn(types::BLACK);
  } else {
    throw std::runtime_error("unknown color");
  }

  pos.set_rule50(static_cast<std::uint8_t>(std::stoi(rule50_str)));

  return pos;
}

}; // namespace c_chess_cli
