cmake_minimum_required(VERSION 3.10.0)
project(dataset VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O2")

# Aktivieren von AddressSanitizer für Debug-Builds
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  add_compile_options(-fsanitize=address -g -O1)
  add_link_options(-fsanitize=address)
endif()

file(GLOB CPP_FILES "dataset/*.cpp")

add_library(dataset SHARED ${CPP_FILES})

add_executable(test dataset/test.cpp)
target_link_libraries(test dataset)
