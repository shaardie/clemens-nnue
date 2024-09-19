#include <cassert>
#include <cstdint>
#include <exception>
#include <fstream>

#include "c_chess_cli.hpp"

namespace c_chess_cli
{

    void Pos::unpack_packed_pieces()
    {
        // shadow occ
        uint64_t occ = occ;

        int i = 0;
        while (occ)
        {
            int packed_piece = i % 2 ? packed_pieces[(i+1)/2] >> 4 : packed_pieces[(i+1)/2] & 0x0F;
            Piece *piece = &pieces[i];

            // get square via builtin least significat bit function
            piece->square = __builtin_ctzll(occ & -occ);

            // reduce occupation
            occ &= occ - 1;

            // get piece and get proper piece type and color from it
            piece->type = c_chess_cli::PieceType((packed_piece & 0xFE) / 2);
            piece->color = c_chess_cli::Color(packed_piece & 1);
            i++;
        }
    }

    Pos::Pos(std::ifstream &stream)
    {
        // read occupation
        stream.read(reinterpret_cast<char *>(&occ), sizeof(occ));
        if (stream.gcount() != sizeof(occ))
        {
            throw std::exception();
        }

        // read turn and rule50
        std::uint8_t turn_and_rule50;
        stream.read(reinterpret_cast<char *>(&turn_and_rule50), sizeof(turn_and_rule50));
        if (stream.gcount() != sizeof(turn_and_rule50))
        {
            throw std::exception();
        }
        std::uint8_t turn = turn_and_rule50 & 1;
        assert(turn <= 1);
        std::uint8_t rule50 = turn_and_rule50 >> 1;
        assert(rule50 <= 100);

        // calculate number of pieces (number of 1s)
        this->number_of_pieces = __builtin_popcountll(occ);
        assert(number_of_pieces <= 32);

        // read packed pieces
        int packed_pieces_size = (number_of_pieces + 1) / 2;
        stream.read(reinterpret_cast<char *>(packed_pieces), packed_pieces_size);
        if (stream.gcount() != packed_pieces_size)
        {
            throw std::exception();
        }

        unpack_packed_pieces();
        stream.read(reinterpret_cast<char *>(&score), sizeof(score));
        if (stream.gcount() != sizeof(score))
        {
            throw std::exception();
        }

        stream.read(reinterpret_cast<char *>(&result), sizeof(result));
        if (stream.gcount() != sizeof(result))
        {
            throw std::exception();
        }
        assert(result <= 2);
    }

    Pos::~Pos()
    {
    }
};
