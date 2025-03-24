cmake_minimum_required(VERSION 3.31)
project(dataset VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O2")

# Aktivieren von AddressSanitizer für Debug-Builds
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  add_compile_options(-fsanitize=address -g -O1)
  add_link_options(-fsanitize=address)
endif()

file(GLOB CPP_FILES "src/*.cpp")

add_library(dataset SHARED ${CPP_FILES})

add_executable(run src/run.cpp)
target_link_libraries(run dataset)

# optional tests
option(ENABLE_TESTS "Enable unit tests" ON)
if (ENABLE_TESTS)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.16.0
    )
    FetchContent_MakeAvailable(googletest)
    enable_testing()
    add_subdirectory(tests)
endif()
