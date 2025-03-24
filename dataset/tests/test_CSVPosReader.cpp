#include "../src/c_chess_cli.hpp"
#include <cstdio>
#include <gtest/gtest.h>

const std::string filename = "Chae5hei.csv";

class CSVPosReaderTest : public ::testing::Test {
protected:
  void SetUp() override {
    const std::string csv_input =
        R"(rnbq1rk1/pp2ppbp/6p1/3pP3/3P2n1/6P1/PP2NPBP/RNBQK2R w KQ - 1 9,108,2                                                                                                                                                                           
rnbq1rk1/pp2ppbp/6p1/3pP3/3P2n1/6PP/PP2NPB1/RNBQK2R b KQ - 0 9,-133,0                                                                                                                                                                          
rnbq1rk1/pp2ppbp/6pn/3pP3/3P4/6PP/PP2NPB1/RNBQK2R w KQ - 1 10,131,2                                                                                                                                                                            
rnbq1rk1/pp2ppbp/6pn/3pP3/3P2P1/7P/PP2NPB1/RNBQK2R b KQ - 0 10,-137,0                                                                                                                                                                          
rnbq1rk1/pp2p1bp/5ppn/3pP3/3P2P1/7P/PP2NPB1/RNBQK2R w KQ - 0 11,133,2                                                                                                                                                                          
rnbq1rk1/pp2p1bp/5Ppn/3p4/3P2P1/7P/PP2NPB1/RNBQK2R b KQ - 0 11,-130,0                                                                                                                                                                          
rnbq1rk1/pp4bp/5ppn/3p4/3P2P1/7P/PP2NPB1/RNBQK2R w KQ - 0 12,135,2                                                                                                                                                                             
rnbq1rk1/pp4bp/5ppn/3p4/3P2P1/2N4P/PP2NPB1/R1BQK2R b KQ - 1 12,-126,0                                                                                                                                                                          
rnbq1rk1/pp3nbp/5pp1/3p4/3P2P1/2N4P/PP2NPB1/R1BQK2R w KQ - 2 13,132,2                                                                                                                                                                          
rnbq1rk1/pp3nbp/5pp1/3p4/3P2P1/2N1B2P/PP2NPB1/R2QK2R b KQ - 3 13,-122,0
)";
    std::ofstream stream(filename);
    stream << csv_input;
    stream.close();
  }

  void TearDown() override { std::remove(filename.c_str()); }
};

TEST_F(CSVPosReaderTest, ReadPosWorks) {
  c_chess_cli::CSVPosReader reader(filename);
  types::Pos pos = reader.read_pos();

  EXPECT_EQ(pos.get_score(), 108);
  EXPECT_EQ(pos.get_result(), 2);
  EXPECT_EQ(pos.get_turn(), types::WHITE);
  EXPECT_EQ(pos.get_rule50(), 0);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
