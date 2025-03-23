#include "batchstream.hpp"
extern "C" {
dataset::BatchStream *CreateBatchStream(char *filename, uint batch_size,
                                        uint cache_size) {
  return new dataset::BatchStream(filename, batch_size, cache_size);
}

void DestroyBatchStream(dataset::BatchStream *batchStream) {
  delete batchStream;
}

dataset::SparseBatch *GetNextBatch(dataset::BatchStream *batchstream) {
  return batchstream->GetBatch();
}

void DestroyBatch(dataset::SparseBatch *sparseBatch) { delete sparseBatch; }
}
