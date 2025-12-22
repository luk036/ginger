cmake_minimum_required(VERSION 3.10)
project(ginger_cpp VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Options
option(BUILD_TESTS "Build tests" ON)
option(BUILD_EXAMPLES "Build examples" OFF)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Add doctest if building tests
if(BUILD_TESTS)
    # Download doctest if not present
    include(FetchContent)
    FetchContent_Declare(
        doctest
        GIT_REPOSITORY https://github.com/doctest/doctest.git
        GIT_TAG v2.4.11
    )
    FetchContent_MakeAvailable(doctest)
endif()

# Source files
set(GINGER_SOURCES
    src/aberth.cpp
    src/autocorr.cpp
    src/matrix2.cpp
    src/rootfinding.cpp
    src/vector2.cpp
)

# Header files
set(GINGER_HEADERS
    include/ginger/aberth.hpp
    include/ginger/autocorr.hpp
    include/ginger/matrix2.hpp
    include/ginger/rootfinding.hpp
    include/ginger/vector2.hpp
)

# Create library
add_library(ginger_cpp ${GINGER_SOURCES} ${GINGER_HEADERS})
target_include_directories(ginger_cpp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Tests
if(BUILD_TESTS)
    # Test source files
    set(TEST_SOURCES
        tests/test_aberth.cpp
        tests/test_autocorr.cpp
        tests/test_matrix2.cpp
        tests/test_rootfind.cpp
        tests/test_vector2.cpp
    )
    
    # Create test executable
    add_executable(ginger_tests ${TEST_SOURCES})
    target_link_libraries(ginger_tests ginger_cpp doctest)
    
    # Add test
    add_test(NAME ginger_tests COMMAND ginger_tests)
endif()

# Examples
if(BUILD_EXAMPLES)
    add_executable(example_aberth examples/aberth_example.cpp)
    target_link_libraries(example_aberth ginger_cpp)
endif()