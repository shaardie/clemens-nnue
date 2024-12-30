#include "trainingdataset.hpp"
#include <vector>

namespace dataset {

struct SparseBatch {
  SparseBatch(const std::vector<trainingDataEntry> &entries);
  void fill(const std::vector<trainingDataEntry> &entries);

  int size;
  // int num_active_features;

  int *stm;
  int *score;
  float *result;
  int *white_features_indices;
  int *black_features_indices;

  ~SparseBatch();
};
} // namespace dataset
