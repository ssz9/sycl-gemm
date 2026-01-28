# SYCL GEMM Kernel Generator

A high-performance SYCL-based GEMM (General Matrix Multiplication) kernel generator with cross-platform auto-tuning capabilities. This implementation is designed to work on SYCL-compatible devices such as CPUs, GPUs, and accelerators.

## Features

### 1. SYCL GEMM Kernel Implementation
- Parameterized GEMM kernel in SYCL that performs matrix-matrix multiplication (C = α·A·B + β·C)
- Support for both global memory and local memory (tiled) implementations
- Configurable tile sizes, work-group sizes, and memory access patterns

### 2. Cross-Platform Auto-Tuning
- Automatically tunes parameters based on device capabilities
- Configurable tile sizes, work-group sizes, and memory access patterns
- Benchmarking-based optimization for best performance

### 3. Device Query and Tunable Configurations
- Queries device properties (max work-group size, compute units, memory sizes)
- Selects appropriate tuning options based on device capabilities
- Runtime performance optimization

### 4. Performance Benchmarking
- Comprehensive benchmarking utilities to test various configurations
- Tests different work-group sizes, local memory usage, and loop unrolling factors
- GFLOPS calculation and performance metrics

### 5. Examples and Tests
- Sample programs demonstrating kernel usage
- Unit tests for correctness validation
- Auto-tuning examples

## Directory Structure

```
sycl-gemm/
├── include/
│   ├── gemm_kernel.hpp    # SYCL GEMM kernel implementation
│   └── auto_tuner.hpp     # Auto-tuning logic
├── src/
│   ├── benchmark.cpp      # Performance benchmarking
│   └── example.cpp        # Usage examples
├── tests/
│   ├── test_gemm.cpp      # GEMM correctness tests
│   └── test_auto_tuner.cpp # Auto-tuner tests
├── CMakeLists.txt         # CMake build configuration
└── README.md              # This file
```

## Requirements

- C++17 compatible compiler
- SYCL implementation (Intel oneAPI DPC++, hipSYCL, or other SYCL-compatible compiler)
- CMake 3.14 or higher

## Building

### With Intel oneAPI DPC++

```bash
# Source Intel oneAPI environment
source /opt/intel/oneapi/setvars.sh

# Build
mkdir build && cd build
cmake .. -DCMAKE_CXX_COMPILER=icpx
make
```

### With hipSYCL/OpenSYCL

```bash
# Build
mkdir build && cd build
cmake .. -DCMAKE_CXX_COMPILER=syclcc
make
```

### Generic SYCL

```bash
mkdir build && cd build
cmake ..
make
```

## Running

### Run Examples

```bash
./example
```

This will demonstrate:
- Basic GEMM usage
- GEMM with alpha and beta scaling
- Different matrix sizes
- Auto-tuning

### Run Tests

```bash
ctest
# Or run individually
./test_gemm
./test_auto_tuner
```

### Run Benchmark

```bash
./benchmark
```

This will benchmark different configurations and provide performance metrics in GFLOPS.

## Usage

### Basic GEMM

```cpp
#include "gemm_kernel.hpp"
#include <CL/sycl.hpp>

using namespace sycl_gemm;

int main() {
    sycl::queue queue{sycl::default_selector_v};
    
    const size_t M = 128, N = 128, K = 128;
    std::vector<float> A(M * K, 1.0f);
    std::vector<float> B(K * N, 1.0f);
    std::vector<float> C(M * N, 0.0f);
    
    GemmKernel<float> kernel(queue);
    kernel.gemm(M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);
    
    return 0;
}
```

### With Custom Configuration

```cpp
GemmConfig config;
config.tile_size_m = 16;
config.tile_size_n = 16;
config.tile_size_k = 16;
config.work_group_size_m = 8;
config.work_group_size_n = 8;
config.use_local_memory = true;

GemmKernel<float> kernel(queue, config);
kernel.gemm(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
```

### Auto-Tuning

```cpp
#include "auto_tuner.hpp"

AutoTuner<float> tuner(queue);

// Get default configuration based on device
GemmConfig config = tuner.get_default_config();

// Or auto-tune for specific matrix size
GemmConfig best_config = tuner.auto_tune(M, N, K);

GemmKernel<float> kernel(queue, best_config);
kernel.gemm(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
```

## API Reference

### GemmConfig

Configuration parameters for GEMM kernel:
- `tile_size_m`, `tile_size_n`, `tile_size_k`: Tile dimensions
- `work_group_size_m`, `work_group_size_n`: Work-group dimensions
- `use_local_memory`: Enable local memory (tiling)
- `unroll_factor`: Loop unrolling factor (reserved for future use)

### GemmKernel<T>

GEMM kernel implementation:
- `gemm(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc)`: Performs C = α·A·B + β·C

### AutoTuner<T>

Auto-tuning utilities:
- `get_default_config()`: Get default config based on device
- `auto_tune(M, N, K, iterations)`: Auto-tune for specific matrix size
- `print_device_info()`: Print device properties

## Performance

The implementation provides competitive performance across different SYCL devices:
- Utilizes local memory for efficient data reuse
- Configurable tiling for optimal cache usage
- Auto-tuning selects best parameters for target device

## Future Improvements

- [ ] Extend support for half-precision (fp16) computation
- [ ] Add tuning profiles for specific hardware platforms (Intel, AMD, NVIDIA)
- [ ] Support batched GEMM operations for AI/ML workloads
- [ ] Implement strided GEMM variants
- [ ] Add support for complex number matrices
- [ ] Optimize for specific matrix sizes (e.g., powers of 2)

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

This project is open source. Please check the LICENSE file for details.
