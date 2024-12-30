#include "batchstream.hpp"
#include "c_chess_cli.hpp"
#include "trainingdataset.hpp"
#include <iostream>
#include <thread>
#include <vector>

namespace dataset {

BatchStream::BatchStream(std::string filename, std::uint16_t batch_size,
                         std::uint16_t cache_size)
    : filename(filename), batch_size(batch_size), cache_size(cache_size),
      chan(1024) {
  stream.open(filename);
  if (!stream) {
    throw std::runtime_error("unable to open file");
  }

  thread = std::thread(&BatchStream::run, this);
};

BatchStream::~BatchStream() {
  thread.join();
  stream.close();
};

float convert_result(int turn, int result) {
  switch (result) {
  // turn looses
  case 0:
    return turn == WHITE ? 0 : 1;
  // draw
  case 1:
    return 0.5;
  // turn wins
  case 2:
    return turn == WHITE ? 1 : 0;
    break;
  }
  throw std::runtime_error("strange result");
  return 0;
}

SparseBatch *BatchStream::GetBatch() {
  SparseBatch *sp;
  if (!chan.pop(sp)) {
    return NULL;
  }
  return sp;
};

void BatchStream::addPos(std::vector<trainingDataEntry> &v) {
  c_chess_cli::Pos pos(stream);

  // find kings, I guess this could be done better
  int kings_found = 0;
  int king_squares[2] = {0};
  for (int i = 0; i < pos.number_of_pieces; ++i) {
    c_chess_cli::Piece *piece = &pos.pieces[i];
    if (piece->type != c_chess_cli::KING) {
      continue;
    }
    king_squares[piece->color] = piece->square;
    kings_found++;
    if (2 == kings_found) {
      break;
    }
  }

  // generate indices for all pieces except the kings
  int number_active_features = 0;
  int white_features_indices[MAX_ACTIVE_FEATURES] = {0};
  int black_features_indices[MAX_ACTIVE_FEATURES] = {0};
  for (int i = 0; i < pos.number_of_pieces; ++i) {
    c_chess_cli::Piece *piece = &pos.pieces[i];
    if (piece->type == c_chess_cli::KING) {
      continue;
    }
    PieceType piece_type = fromExtType(piece->type);
    int p_idx = piece_type * 2 + piece->color;
    white_features_indices[number_active_features] =
        piece->square + (p_idx + king_squares[WHITE] * 10) * 64;
    black_features_indices[number_active_features] =
        piece->square + (p_idx + king_squares[BLACK] * 10) * 64;
    number_active_features++;
  }

  trainingDataEntry tde(number_active_features, white_features_indices,
                        black_features_indices, pos.turn, pos.score,
                        convert_result(pos.turn, pos.result));
  v.push_back(tde);
}

void BatchStream::run() {
  while (true) {
    std::vector<trainingDataEntry> v;
    for (int x = 0; x < batch_size; ++x) {
      try {
        this->addPos(v);
      } catch (const std::exception &) {
        chan.close();
        return;
      }
      chan.push(new SparseBatch(v));
    }
  }
};

PieceType fromExtType(c_chess_cli::PieceType extPieceType) {
  switch (extPieceType) {
  case c_chess_cli::PAWN:
    return PAWN;
  case c_chess_cli::PAWN_CAPTURABLE_ENPASSANT:
    return PAWN;
  case c_chess_cli::KNIGHT:
    return KNIGHT;
  case c_chess_cli::ROOK:
    return ROOK;
  case c_chess_cli::ROOK_WITH_CASTLING_RIGHT:
    return ROOK;
  case c_chess_cli::QUEEN:
    return QUEEN;
  case c_chess_cli::KING:
    return KING;
  case c_chess_cli::BISHOP:
    return BISHOP;
  default:
    throw std::runtime_error("unable to open file");
  }
}

} // namespace dataset
dataset::BatchStream *CreateBatchStream(char *filename,
                                        std::uint16_t batch_size,
                                        std::uint16_t cache_size) {
  return new dataset::BatchStream(filename, batch_size, cache_size);
}
