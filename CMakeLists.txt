cmake_minimum_required(VERSION 3.10.0)
project(dataset VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O2")

file(GLOB CPP_FILES "dataset/*.cpp")

add_library(dataset SHARED ${CPP_FILES})

add_executable(test dataset/test.cpp)
target_link_libraries(test dataset)
