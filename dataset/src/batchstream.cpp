#include "batchstream.hpp"
#include "c_chess_cli.hpp"
#include "trainingdataset.hpp"
#include "types.hpp"
#include <algorithm>
#include <mutex>
#include <thread>
#include <vector>

#define MAX_INT16 32767

namespace dataset {

bool ends_with_csv(const std::string &filename) {
  return filename.size() >= 4 && filename.substr(filename.size() - 4) == ".csv";
}

BatchStream::BatchStream(std::string filename, std::uint16_t batch_size,
                         std::uint16_t cache_size)
    : batch_size(batch_size), cache_size(cache_size), chan(1024),
      stopped(false) {

  if (ends_with_csv(filename)) {
    pos_reader = new c_chess_cli::CSVPosReader(filename);
  } else {
    pos_reader = new c_chess_cli::BinPosReader(filename);
  }
  thread = std::thread(&BatchStream::run, this);
};

BatchStream::~BatchStream() {
  {
    std::lock_guard<std::mutex> lock(mtx);
    stopped = true;
  }

  if (thread.joinable()) {
    thread.join();
  }

  SparseBatch *sp;
  while (chan.pop(sp)) {
    delete sp;
  }

  delete pos_reader;
};

float convert_result(int turn, int result) {
  switch (result) {
  // turn looses
  case 0:
    return turn == types::WHITE ? 0 : 1;
  // draw
  case 1:
    return 0.5;
  // turn wins
  case 2:
    return turn == types::WHITE ? 1 : 0;
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
  // Get next position, which is not a forced mate
  types::Pos pos;
  while (true) {
    pos = pos_reader->read_pos();
    if (pos.get_score() < MAX_INT16 - 1000 &&
        pos.get_score() > -MAX_INT16 + 1000) {
      break;
    }
  }

  // find kings, I guess this could be done better
  int kings_found = 0;
  int king_squares[2] = {0};
  for (int i = 0; i < pos.get_number_of_pieces(); ++i) {
    types::Piece *piece = &pos.pieces[i];
    if (piece->type != types::KING) {
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
  int white_features_indices[types::MAX_ACTIVE_FEATURES] = {0};
  int black_features_indices[types::MAX_ACTIVE_FEATURES] = {0};
  for (int i = 0; i < pos.get_number_of_pieces(); ++i) {
    types::Piece *piece = &pos.pieces[i];
    if (piece->type == types::KING) {
      continue;
    }
    types::PieceType piece_type = piece->type;
    int p_idx = piece_type * 2 + piece->color;
    white_features_indices[number_active_features] =
        piece->square + (p_idx + king_squares[types::WHITE] * 10) * 64;
    black_features_indices[number_active_features] =
        piece->square + (p_idx + king_squares[types::BLACK] * 10) * 64;
    number_active_features++;

    // sort the indices
    std::sort(white_features_indices,
              white_features_indices + number_active_features);
    std::sort(black_features_indices,
              black_features_indices + number_active_features);
  }

  trainingDataEntry tde(number_active_features, white_features_indices,
                        black_features_indices, pos.get_turn(), pos.get_score(),
                        convert_result(pos.get_turn(), pos.get_result()));
  v.push_back(tde);
}

void BatchStream::run() {
  while (true) {
    {
      std::lock_guard<std::mutex> lock(mtx);
      if (stopped) {
        chan.close();
        return;
      }
    }
    std::vector<trainingDataEntry> v;
    for (int x = 0; x < batch_size; ++x) {
      try {
        this->addPos(v);
      } catch (const std::exception &) {
        chan.close();
        return;
      }
    }
    chan.push(new SparseBatch(v));
  }
};

} // namespace dataset
