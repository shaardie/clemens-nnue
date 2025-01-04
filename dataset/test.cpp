#include "c_interface.hpp"
#include <iostream>

#define BATCH_SIZE 1024
#define CACHE_SIZE 1024
#define MAX_ITERATIONS 10000

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cout << "no filename given" << std::endl;
    return 1;
  }
  auto batchstream = CreateBatchStream(argv[1], BATCH_SIZE, CACHE_SIZE);
  auto start = std::chrono::high_resolution_clock::now();
  int i;
  for (i = 0; i < MAX_ITERATIONS; i++) {
    auto sparsebatch = GetNextBatch(batchstream);
    if (sparsebatch == NULL) {
      break;
    }
    DestroyBatch(sparsebatch);
  }
  auto end = std::chrono::high_resolution_clock::now();

  DestroyBatchStream(batchstream);

  std::chrono::duration<double> elapsed = end - start;
  std::cout << "Read " << i * BATCH_SIZE << " positions in " << i << " batches "
            << elapsed.count() << " seconds" << std::endl;
  return 0;
}
