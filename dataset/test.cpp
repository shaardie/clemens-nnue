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
    BatchStream batchstream(argv[1], 2);
    SparseBatch *sparsebatch = batchstream.GetBatch();
    std::cout << "size: " << sparsebatch->size << std::endl;
    std::cout << "score: ";
    printArray(sparsebatch->score, sparsebatch->size);
    std::cout << std::endl;
    std::cout << "result: ";
    printArray(sparsebatch->result, sparsebatch->size);
    std::cout << std::endl;
    std::cout << "stm: ";
    printArray(sparsebatch->stm, sparsebatch->size);
    std::cout << std::endl;
    std::cout << "white features: " << std::endl;
    for (int i = 0; i < sparsebatch->size; ++i)
    {
        std::cout << "  ";
        printArray(sparsebatch->white_features_indices, sparsebatch->size * MAX_ACTIVE_FEATURES);
        std::cout << std::endl;
    }
        std::cout << "black features: " << std::endl;
    for (int i = 0; i < sparsebatch->size; ++i)
    {
        std::cout << "  ";
        printArray(sparsebatch->black_features_indices, sparsebatch->size * MAX_ACTIVE_FEATURES);
        std::cout << std::endl;
    }

    return 0;
}
