cmake_minimum_required(VERSION 3.14)
project(sycl-gemm LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find SYCL implementation
# Try to find Intel oneAPI DPC++
find_package(IntelSYCL QUIET)

if(NOT IntelSYCL_FOUND)
    # Try hipSYCL/OpenSYCL or other implementations
    find_package(hipSYCL QUIET)
    if(NOT hipSYCL_FOUND)
        # Use generic SYCL flags
        message(STATUS "No SYCL package found, using generic flags")
        
        # Check for common SYCL compiler flags
        if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            # For DPC++ compiler
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl")
        elseif(CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM")
            # For Intel oneAPI DPC++
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl")
        endif()
    endif()
endif()

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Library (header-only)
add_library(sycl_gemm INTERFACE)
target_include_directories(sycl_gemm INTERFACE
    ${CMAKE_SOURCE_DIR}/include
)

# Example executable
add_executable(example src/example.cpp)
target_link_libraries(example sycl_gemm)

# Benchmark executable
add_executable(benchmark src/benchmark.cpp)
target_link_libraries(benchmark sycl_gemm)

# Enable testing
enable_testing()

# Test executables
add_executable(test_gemm tests/test_gemm.cpp)
target_link_libraries(test_gemm sycl_gemm)

add_executable(test_auto_tuner tests/test_auto_tuner.cpp)
target_link_libraries(test_auto_tuner sycl_gemm)

# Add tests
add_test(NAME test_gemm COMMAND test_gemm)
add_test(NAME test_auto_tuner COMMAND test_auto_tuner)

# Installation
install(DIRECTORY include/ DESTINATION include/sycl_gemm)
install(TARGETS example benchmark DESTINATION bin)

# Print configuration info
message(STATUS "SYCL GEMM Configuration:")
message(STATUS "  C++ Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
