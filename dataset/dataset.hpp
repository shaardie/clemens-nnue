#ifndef DATASET_HPP
#define DATASET_HPP

#include <vector>
#include <string>
#include <cstdint>
#include <iostream>
#include <bit>
#include <fstream>
#include <cassert>
#include <map>

#include "c_chess_cli.hpp"

#define MAX_ACTIVE_FEATURES 32

enum PieceType
{
    PAWN,
    KNIGHT,
    BISHOP,
    ROOK,
    QUEEN,
    KING,
};

enum Color
{
    WHITE,
    BLACK
};

struct trainingDataEntry
{
    int number_active_features;
    int white_features_indices[MAX_ACTIVE_FEATURES];
    int black_features_indices[MAX_ACTIVE_FEATURES];
    int turn;
    int score;
    float result;

    trainingDataEntry(
        int number_active_features,
        const int (&wfi)[MAX_ACTIVE_FEATURES],
        const int (&bfi)[MAX_ACTIVE_FEATURES],
        int turn, int score, float result);
};
std::ostream &operator<<(std::ostream &os, const trainingDataEntry &tde);

struct SparseBatch
{
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

struct BatchStream
{
    BatchStream(std::string filename, std::uint16_t batch_size);
    ~BatchStream();
    SparseBatch *GetBatch();

    std::string filename;
    std::ifstream stream;
    std::uint16_t batch_size;
};

extern "C"
{
    BatchStream *CreateBatchStream(char *filename, std::uint16_t batch_size);
}

#endif