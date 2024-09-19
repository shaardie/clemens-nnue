#include "dataset.hpp"
#include <iostream>

int main(int argc, char *argv[])
{
    BatchStream batchstream(argv[1], 2);
    SparseBatch *sparsebatch = batchstream.GetBatch();
    return 0;
    // std::cout << "size: " << sparsebatch->size << std::endl;
    // std::cout << "num_active_features: " << sparsebatch->num_active_features << std::endl;
    for (int i = 0; i < sparsebatch->size * MAX_ACTIVE_FEATURES * 2; ++i)
    {
        // std::cout << sparsebatch->white_features_indices[i] << std::endl;
    }
    // std::cout << *(sparsebatch->score) << std::endl;
    return 0;
}
