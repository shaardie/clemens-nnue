#ifndef TRAININGDATAENTRY_HPP
#define TRAININGDATAENTRY_HPP

#include "consts.hpp"
#include <iostream>

namespace dataset {

struct trainingDataEntry {
  int number_active_features;
  int white_features_indices[MAX_ACTIVE_FEATURES];
  int black_features_indices[MAX_ACTIVE_FEATURES];
  int turn;
  int score;
  float result;

  trainingDataEntry(int number_active_features,
                    const int (&wfi)[MAX_ACTIVE_FEATURES],
                    const int (&bfi)[MAX_ACTIVE_FEATURES], int turn, int score,
                    float result);
};
std::ostream &operator<<(std::ostream &os, const trainingDataEntry &tde);

} // namespace dataset

#endif
