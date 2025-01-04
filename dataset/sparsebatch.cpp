
#include "batchstream.hpp"
namespace dataset {

SparseBatch::SparseBatch(const std::vector<trainingDataEntry> &entries) {

  // The number of positions in the batch
  size = entries.size();

  // The total number of white/black active features in the whole batch.
  num_active_features = 0;

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

  white_features_indices = new int[size * MAX_ACTIVE_FEATURES * 2];
  black_features_indices = new int[size * MAX_ACTIVE_FEATURES * 2];

  fill(entries);
}

void SparseBatch::fill(const std::vector<trainingDataEntry> &entries) {
  for (int position_index = 0; position_index < size; ++position_index) {
    const trainingDataEntry *entry = &entries[position_index];
    stm[position_index] = entry->turn;
    score[position_index] = entry->score;
    result[position_index] = entry->result;

    for (int j = 0; j < entry->number_active_features; ++j) {
      white_features_indices[2 * num_active_features] = position_index;
      black_features_indices[2 * num_active_features] = position_index;
      white_features_indices[2 * num_active_features + 1] =
          entry->white_features_indices[j];
      black_features_indices[2 * num_active_features + 1] =
          entry->black_features_indices[j];
      num_active_features++;
    }
  }
}

SparseBatch::~SparseBatch() {
  // RAII! Or use std::unique_ptr<T[]>, but remember that only raw pointers
  // should be passed through language boundaries as std::unique_ptr doesn't
  // have stable ABI
  delete[] stm;
  delete[] score;
  delete[] result;
  delete[] white_features_indices;
  delete[] black_features_indices;
}
} // namespace dataset
