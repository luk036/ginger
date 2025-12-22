#!/bin/bash
set -e

# Create build directory
mkdir -p build
cd build

# Compile source files
echo "Compiling source files..."
clang++ -std=c++17 -I../include -c ../src/vector2.cpp -o vector2.o
clang++ -std=c++17 -I../include -c ../src/matrix2.cpp -o matrix2.o
clang++ -std=c++17 -I../include -c ../src/rootfinding.cpp -o rootfinding.o
clang++ -std=c++17 -I../include -c ../src/aberth.cpp -o aberth.o
clang++ -std=c++17 -I../include -c ../src/autocorr.cpp -o autocorr.o

# Create static library
echo "Creating static library..."
ar rcs libginger_cpp.a vector2.o matrix2.o rootfinding.o aberth.o autocorr.o

# Compile tests
echo "Compiling tests..."
clang++ -std=c++17 -I../include -I. -c ../tests/test_vector2.cpp -o test_vector2.o
clang++ -std=c++17 -I../include -I. -c ../tests/test_matrix2.cpp -o test_matrix2.o
clang++ -std=c++17 -I../include -I. -c ../tests/test_rootfind.cpp -o test_rootfind.o
clang++ -std=c++17 -I../include -I. -c ../tests/test_aberth.cpp -o test_aberth.o
clang++ -std=c++17 -I../include -I. -c ../tests/test_autocorr.cpp -o test_autocorr.o

# Link test executable (without doctest for now)
echo "Linking test executable..."
clang++ -std=c++17 test_vector2.o test_matrix2.o test_rootfind.o test_aberth.o test_autocorr.o libginger_cpp.a -o ginger_tests

echo "Build complete!"
echo "Run tests with: ./build/ginger_tests"