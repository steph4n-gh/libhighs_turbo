cmake_minimum_required(VERSION 3.20)
project(highs_turbo_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|AppleClang")
    add_compile_options(-O3 -funroll-loops -Wall -Wextra -Wno-unused-parameter)
endif()

find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)

# GMP headers & libraries
find_path(GMP_INCLUDE_DIR NAMES gmp.h HINTS /opt/homebrew/include REQUIRED)
find_library(GMP_LIB NAMES gmp HINTS /opt/homebrew/lib REQUIRED)
find_library(GMPXX_LIB NAMES gmpxx HINTS /opt/homebrew/lib REQUIRED)

# Extension module
python_add_library(_compiled_engine MODULE WITH_SOABI
    bit_graph.cpp
    rational_verifier.cpp
    cut_engine.cpp
    solver_callback.cpp
    bindings.cpp
)

target_include_directories(_compiled_engine PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${GMP_INCLUDE_DIR}
)

target_link_libraries(_compiled_engine PRIVATE
    ${GMPXX_LIB}
    ${GMP_LIB}
    "-framework Security"
)
