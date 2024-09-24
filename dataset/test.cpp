#include "dataset.hpp"
#include <iostream>

template <typename T>
void printArray(T arr[], int size)
{
    std::cout << "[";
    for (int i = 0; i < size; ++i)
    {
        std::cout << arr[i];
        if (i < size - 1)
        {
            std::cout << ", ";
        }
    }
    std::cout << "]";
}

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        std::cout << "no filename given" << std::endl;
        return 1;
    }

    BatchStream *batchStream = CreateBatchStream(argv[1], 16384);
    int i = 0;
    while (true)
    {
        if (i % 100 == 0)
        {
            std::cout << i << " Batches read" << std::endl;
        }
        i++;
        SparseBatch *sparsebatch = GetNextBatch(batchStream);
        if (NULL == sparsebatch) {
            return 0;
        }
        DestroyBatch(sparsebatch);
        // std::cout << "size: " << sparsebatch->size << std::endl;
        // std::cout << "num_active_features: " << sparsebatch->num_active_features << std::endl;
        // std::cout << "score: ";
        // printArray(sparsebatch->score, sparsebatch->size);
        // std::cout << std::endl;
        // std::cout << "result: ";
        // printArray(sparsebatch->result, sparsebatch->size);
        // std::cout << std::endl;
        // std::cout << "stm: ";
        // printArray(sparsebatch->stm, sparsebatch->size);
        // std::cout << std::endl;
        // std::cout << "white features: ";
        // printArray(sparsebatch->white_features_indices, sparsebatch->size * MAX_ACTIVE_FEATURES * 2);
        // std::cout << std::endl;
        // std::cout << "black features: ";
        // printArray(sparsebatch->black_features_indices, sparsebatch->size * MAX_ACTIVE_FEATURES * 2);
    }
    DestroyBatchStream(batchStream);
    return 0;
}
