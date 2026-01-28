#include "gemm_kernel.hpp"
#include "auto_tuner.hpp"
#include <CL/sycl.hpp>
#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>

using namespace sycl_gemm;

/**
 * @brief Helper function to verify GEMM result
 */
template<typename T>
bool verify_gemm(const std::vector<T>& C, size_t M, size_t N, size_t K, T expected_value, T tolerance = 1e-3) {
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            T value = C[i * N + j];
            if (std::abs(value - expected_value) > tolerance) {
                std::cerr << "Verification failed at (" << i << ", " << j << "): "
                          << "expected " << expected_value << ", got " << value << std::endl;
                return false;
            }
        }
    }
    return true;
}

/**
 * @brief Example 1: Basic GEMM usage
 */
void example_basic_gemm(sycl::queue& queue) {
    std::cout << "\n=== Example 1: Basic GEMM Usage ===" << std::endl;
    
    const size_t M = 64;
    const size_t N = 64;
    const size_t K = 64;
    
    // Create input matrices (all ones)
    std::vector<float> A(M * K, 1.0f);
    std::vector<float> B(K * N, 1.0f);
    std::vector<float> C(M * N, 0.0f);
    
    // Use default configuration
    GemmConfig config;
    config.use_local_memory = false; // Simple configuration
    
    GemmKernel<float> kernel(queue, config);
    
    // Perform C = A * B
    kernel.gemm(M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);
    
    // Verify result (should be K for each element)
    if (verify_gemm(C, M, N, K, static_cast<float>(K))) {
        std::cout << "✓ Basic GEMM passed! C[0,0] = " << C[0] << " (expected " << K << ")" << std::endl;
    } else {
        std::cout << "✗ Basic GEMM failed!" << std::endl;
    }
}

/**
 * @brief Example 2: GEMM with alpha and beta
 */
void example_gemm_with_scaling(sycl::queue& queue) {
    std::cout << "\n=== Example 2: GEMM with Alpha and Beta ===" << std::endl;
    
    const size_t M = 32;
    const size_t N = 32;
    const size_t K = 32;
    
    std::vector<float> A(M * K, 2.0f);
    std::vector<float> B(K * N, 3.0f);
    std::vector<float> C(M * N, 1.0f);
    
    float alpha = 0.5f;
    float beta = 2.0f;
    
    GemmConfig config;
    config.use_local_memory = false;
    
    GemmKernel<float> kernel(queue, config);
    
    // Perform C = alpha * A * B + beta * C
    kernel.gemm(M, N, K, alpha, A, K, B, N, beta, C, N);
    
    // Expected: 0.5 * (2 * 3 * 32) + 2 * 1 = 0.5 * 192 + 2 = 98
    float expected = alpha * (2.0f * 3.0f * K) + beta * 1.0f;
    
    if (verify_gemm(C, M, N, K, expected, 0.1f)) {
        std::cout << "✓ Scaled GEMM passed! C[0,0] = " << C[0] << " (expected " << expected << ")" << std::endl;
    } else {
        std::cout << "✗ Scaled GEMM failed!" << std::endl;
    }
}

/**
 * @brief Example 3: Auto-tuning
 */
void example_auto_tuning(sycl::queue& queue) {
    std::cout << "\n=== Example 3: Auto-tuning ===" << std::endl;
    
    AutoTuner<float> tuner(queue);
    
    // Get default configuration based on device
    GemmConfig default_config = tuner.get_default_config();
    std::cout << "Default config: tile=" << default_config.tile_size_m 
              << "x" << default_config.tile_size_n
              << ", wg=" << default_config.work_group_size_m 
              << "x" << default_config.work_group_size_n << std::endl;
    
    // Auto-tune for specific matrix size
    const size_t M = 128;
    const size_t N = 128;
    const size_t K = 128;
    
    std::cout << "\nAuto-tuning for " << M << "x" << N << "x" << K << " matrix..." << std::endl;
    GemmConfig best_config = tuner.auto_tune(M, N, K, 5);
    
    // Test with best configuration
    std::vector<float> A(M * K, 1.0f);
    std::vector<float> B(K * N, 1.0f);
    std::vector<float> C(M * N, 0.0f);
    
    GemmKernel<float> kernel(queue, best_config);
    kernel.gemm(M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);
    
    if (verify_gemm(C, M, N, K, static_cast<float>(K))) {
        std::cout << "✓ Auto-tuned GEMM passed!" << std::endl;
    } else {
        std::cout << "✗ Auto-tuned GEMM failed!" << std::endl;
    }
}

/**
 * @brief Example 4: Different matrix sizes
 */
void example_different_sizes(sycl::queue& queue) {
    std::cout << "\n=== Example 4: Different Matrix Sizes ===" << std::endl;
    
    GemmConfig config;
    config.use_local_memory = false;
    GemmKernel<float> kernel(queue, config);
    
    std::vector<std::tuple<size_t, size_t, size_t>> sizes = {
        {16, 16, 16},
        {32, 32, 32},
        {64, 32, 48},
        {100, 100, 100}
    };
    
    for (const auto& [M, N, K] : sizes) {
        std::vector<float> A(M * K, 1.0f);
        std::vector<float> B(K * N, 1.0f);
        std::vector<float> C(M * N, 0.0f);
        
        kernel.gemm(M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);
        
        if (verify_gemm(C, M, N, K, static_cast<float>(K))) {
            std::cout << "✓ GEMM " << M << "x" << N << "x" << K << " passed" << std::endl;
        } else {
            std::cout << "✗ GEMM " << M << "x" << N << "x" << K << " failed" << std::endl;
        }
    }
}

int main() {
    try {
        // Create SYCL queue
        sycl::queue queue{sycl::default_selector_v};
        
        std::cout << "SYCL GEMM Kernel Generator - Examples" << std::endl;
        std::cout << "======================================" << std::endl;
        
        // Print device information
        auto device = queue.get_device();
        std::cout << "Using device: " << device.get_info<sycl::info::device::name>() << std::endl;
        
        AutoTuner<float> tuner(queue);
        tuner.print_device_info();
        
        // Run examples
        example_basic_gemm(queue);
        example_gemm_with_scaling(queue);
        example_different_sizes(queue);
        example_auto_tuning(queue);
        
        std::cout << "\n=== All examples completed ===" << std::endl;
        
    } catch (const sycl::exception& e) {
        std::cerr << "SYCL exception: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
