#ifndef BATCHSTREAM_HPP
#define BATCHSTREAM_HPP

#include "c_chess_cli.hpp"
#include "channel.hpp"
#include "sparsebatch.hpp"
#include "trainingdataset.hpp"
#include <cstdint>
#include <string>
#include <thread>
#include <vector>

namespace dataset {

class BatchStream {
private:
  void run();
  void addPos(std::vector<trainingDataEntry> &v);

  std::uint16_t batch_size;
  std::uint16_t cache_size;
  c_chess_cli::PosReader *pos_reader;

  channel::Channel<SparseBatch *> chan;
  std::thread thread;
  bool stopped;
  std::mutex mtx;

public:
  BatchStream(std::string filename, std::uint16_t batch_size,
              std::uint16_t cache_size);
  ~BatchStream();
  SparseBatch *GetBatch();
};

PieceType fromExtType(c_chess_cli::PieceType extPieceType);
} // namespace dataset

#endif
