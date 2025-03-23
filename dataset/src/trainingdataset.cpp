#include "trainingdataset.hpp"
namespace dataset {

trainingDataEntry::trainingDataEntry(
    int number_active_features, const int (&wfi)[types::MAX_ACTIVE_FEATURES],
    const int (&bfi)[types::MAX_ACTIVE_FEATURES], int turn, int score,
    float result)
    : number_active_features(number_active_features), turn(turn), score(score),
      result(result) {
  for (int i = 0; i < number_active_features; ++i) {
    white_features_indices[i] = wfi[i];
    black_features_indices[i] = bfi[i];
  }
}

} // namespace dataset
