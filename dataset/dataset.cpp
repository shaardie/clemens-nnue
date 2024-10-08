#include <vector>
#include <string>
#include <cstdint>
#include <iostream>
#include <bit>
#include <fstream>
#include <cassert>
#include <map>

#include "c_chess_cli.hpp"
#include "dataset.hpp"

PieceType fromExtType(c_chess_cli::PieceType extPieceType)
{
    switch (extPieceType)
    {
    case c_chess_cli::PAWN:
        return PAWN;
    case c_chess_cli::PAWN_CAPTURABLE_ENPASSANT:
        return PAWN;
    case c_chess_cli::KNIGHT:
        return KNIGHT;
    case c_chess_cli::ROOK:
        return ROOK;
    case c_chess_cli::ROOK_WITH_CASTLING_RIGHT:
        return ROOK;
    case c_chess_cli::QUEEN:
        return QUEEN;
    case c_chess_cli::KING:
        return KING;
    case c_chess_cli::BISHOP:
        return BISHOP;
    default:
        throw std::runtime_error("unable to open file");
    }
}

trainingDataEntry::trainingDataEntry(
    int number_active_features,
    const int (&wfi)[MAX_ACTIVE_FEATURES],
    const int (&bfi)[MAX_ACTIVE_FEATURES],
    int turn, int score, float result)
    : number_active_features(number_active_features),
      turn(turn), score(score), result(result)
{
    for (int i = 0; i < number_active_features; ++i)
    {
        white_features_indices[i] = wfi[i];
        black_features_indices[i] = bfi[i];
    }
}

SparseBatch::SparseBatch(const std::vector<trainingDataEntry> &entries)
{

    // The number of positions in the batch
    size = entries.size();

    // The total number of white/black active features in the whole batch.
    // I do not get this one
    // num_active_features = 0;

    // The side to move for each position. 1 for white, 0 for black.
    // Required for ordering the accumulator slices in the forward pass.
    stm = new int[size];

    // The score for each position. This is the value that we will be teaching the network.
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

    // I do not get why this should be size * MAX_ACTIVE_FEATURES * 2, so I removed it.
    // Let's see if this break.
    white_features_indices = new int[size * MAX_ACTIVE_FEATURES];
    black_features_indices = new int[size * MAX_ACTIVE_FEATURES];

    fill(entries);
}

void SparseBatch::fill(const std::vector<trainingDataEntry> &entries)
{
    for (int i = 0; i < size; ++i)
    {
        stm[i] = entries[i].turn;
        score[i] = entries[i].score;
        result[i] = entries[i].result;
        int offset = i * MAX_ACTIVE_FEATURES;
        for (int j = 0; j < MAX_ACTIVE_FEATURES; ++j)
        {
            int idx = offset + j;
            if (j >= entries[i].number_active_features)
            {
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

SparseBatch::~SparseBatch()
{
    // RAII! Or use std::unique_ptr<T[]>, but remember that only raw pointers should
    // be passed through language boundaries as std::unique_ptr doesn't have stable ABI
    delete[] stm;
    delete[] score;
    delete[] white_features_indices;
    delete[] black_features_indices;
}

BatchStream::BatchStream(std::string filename, std::uint16_t batch_size) : filename(filename), batch_size(batch_size)
{
    stream.open(filename);
    if (!stream)
    {
        throw std::runtime_error("unable to open file");
    }
};

BatchStream::~BatchStream()
{
    stream.close();
};

float convert_result(int turn, int result)
{
    switch (result)
    {
    // turn looses
    case 0:
        return turn == WHITE ? 0 : 1;
    // draw
    case 1:
        return 0.5;
    // turn wins
    case 2:
        return turn == WHITE ? 1 : 0;
        break;
    }
    throw std::runtime_error("strange result");
    return 0;
}

SparseBatch *BatchStream::GetBatch()
{
    std::vector<trainingDataEntry> v;
    for (int x = 0; x < batch_size; ++x)
    {
        c_chess_cli::Pos pos(stream);

        // find kings, I guess this could be done better
        int kings_found = 0;
        int king_squares[2] = {0};
        for (int i = 0; i < pos.number_of_pieces; ++i)
        {
            c_chess_cli::Piece *piece = &pos.pieces[i];
            if (piece->type != c_chess_cli::KING)
            {
                continue;
            }
            king_squares[piece->color] = piece->square;
            kings_found++;
            if (2 == kings_found)
            {
                break;
            }
        }

        // generate indices for all pieces except the kings
        int number_active_features = 0;
        int white_features_indices[MAX_ACTIVE_FEATURES] = {0};
        int black_features_indices[MAX_ACTIVE_FEATURES] = {0};
        for (int i = 0; i < pos.number_of_pieces; ++i)
        {
            c_chess_cli::Piece *piece = &pos.pieces[i];
            if (piece->type == c_chess_cli::KING)
            {
                continue;
            }
            PieceType piece_type = fromExtType(piece->type);
            int p_idx = piece_type * 2 + piece->color;
            white_features_indices[number_active_features] = piece->square + (p_idx + king_squares[WHITE] * 10) * 64;
            black_features_indices[number_active_features] = piece->square + (p_idx + king_squares[BLACK] * 10) * 64;
            number_active_features++;
        }

        trainingDataEntry tde(
            number_active_features, white_features_indices, black_features_indices,
            pos.turn, pos.score, convert_result(pos.turn, pos.result));
        v.push_back(tde);
    };

    return new SparseBatch(v);
};

BatchStream *CreateBatchStream(char *filename, std::uint16_t batch_size)
{
    return new BatchStream(filename, batch_size);
}
