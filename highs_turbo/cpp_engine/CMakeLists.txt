cmake_minimum_required(VERSION 3.20)
project(highs_turbo_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|AppleClang")
    add_compile_options(-O3 -funroll-loops -Wall -Wextra -Wno-unused-parameter)
endif()

find_package(Python 3 COMPONENTS Interpreter Development.Module REQUIRED)

# GMP headers & libraries
set(GMP_ROOT "/opt/homebrew/Cellar/gmp/6.3.0")
find_path(GMP_INCLUDE_DIR NAMES gmp.h PATHS ${GMP_ROOT}/include /opt/homebrew/include)
find_library(GMP_LIB NAMES gmp PATHS ${GMP_ROOT}/lib /opt/homebrew/lib)
find_library(GMPXX_LIB NAMES gmpxx PATHS ${GMP_ROOT}/lib /opt/homebrew/lib)

# Extension module
python_add_library(_compiled_engine MODULE
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
    ${GMP_LIB}
    ${GMPXX_LIB}
    "-framework Security"
)
