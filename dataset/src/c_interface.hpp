#ifndef C_INTERFACE_HPP
#define C_INTERFACE_HPP

#include "batchstream.hpp"

#define EXPORT __attribute__((visibility("default")))

extern "C" {
dataset::BatchStream *CreateBatchStream(char *filename, uint batch_size,
                                        uint cache_size);
void DestroyBatchStream(dataset::BatchStream *batchStream);
dataset::SparseBatch *GetNextBatch(dataset::BatchStream *batchstream);
void DestroyBatch(dataset::SparseBatch *sparseBatch);
}
#endif
