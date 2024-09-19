cmake_minimum_required(VERSION 3.5.0)
project(dataset VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O2")

add_library(dataset dataset/c_chess_cli.cpp dataset/dataset.cpp)

add_executable(test dataset/test.cpp)
target_link_libraries(test dataset)
