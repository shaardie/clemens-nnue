
#include "batchstream.hpp"
namespace dataset {

SparseBatch::SparseBatch(const std::vector<trainingDataEntry> &entries) {

  // The number of positions in the batch
  size = entries.size();

  // The total number of white/black active features in the whole batch.
  // I do not get this one
  // num_active_features = 0;

  // The side to move for each position. 1 for white, 0 for black.
  // Required for ordering the accumulator slices in the forward pass.
  stm = new int[size];

  // The score for each position. This is the value that we will be teaching the
  // network.
  score = new int[size];

  result = new float[size];

  // The indices of the active features.
  // Why is the size * 2?! The answer is that the indices are 2 dimensional
  // (position_index, feature_index). It's effectively a matrix of size
  // (num_active_*_features, 2).
  // IMPORTANT: We must make sure that the indices are in ascending order.
  // That is first comes the first position, then second, then third,
  // and so on. And within features for one position the feature indices
  // are also in ascending order. Why this is needed will be apparent later.

  // I do not get why this should be size * MAX_ACTIVE_FEATURES * 2, so I
  // removed it. Let's see if this break.
  white_features_indices = new int[size * MAX_ACTIVE_FEATURES];
  black_features_indices = new int[size * MAX_ACTIVE_FEATURES];

  fill(entries);
}

void SparseBatch::fill(const std::vector<trainingDataEntry> &entries) {
  for (int i = 0; i < size; ++i) {
    stm[i] = entries[i].turn;
    score[i] = entries[i].score;
    result[i] = entries[i].result;
    int offset = i * MAX_ACTIVE_FEATURES;
    for (int j = 0; j < MAX_ACTIVE_FEATURES; ++j) {
      int idx = offset + j;
      if (j >= entries[i].number_active_features) {
        white_features_indices[idx] = -1;
        black_features_indices[idx] = -1;
        continue;
      }
      white_features_indices[idx] = entries[i].black_features_indices[j];
      black_features_indices[idx] = entries[i].black_features_indices[j];
      continue;
    }
  }
}

SparseBatch::~SparseBatch() {
  // RAII! Or use std::unique_ptr<T[]>, but remember that only raw pointers
  // should be passed through language boundaries as std::unique_ptr doesn't
  // have stable ABI
  delete[] stm;
  delete[] score;
  delete[] white_features_indices;
  delete[] black_features_indices;
}
} // namespace dataset
