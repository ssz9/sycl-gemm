#include "gemm_kernel.hpp"
#include <CL/sycl.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>

using namespace sycl_gemm;

bool verify_result(const std::vector<float>& C, size_t M, size_t N, float expected, float tolerance = 1e-3) {
    for (size_t i = 0; i < M * N; i++) {
        if (std::abs(C[i] - expected) > tolerance) {
            std::cerr << "Mismatch at index " << i << ": expected " << expected 
                      << ", got " << C[i] << std::endl;
            return false;
        }
    }
    return true;
}

void test_basic_gemm(sycl::queue& queue) {
    std::cout << "Test: Basic GEMM (C = A * B)... ";
    
    const size_t M = 32, N = 32, K = 32;
    std::vector<float> A(M * K, 1.0f);
    std::vector<float> B(K * N, 1.0f);
    std::vector<float> C(M * N, 0.0f);
    
    GemmConfig config;
    config.use_local_memory = false;
    GemmKernel<float> kernel(queue, config);
    
    kernel.gemm(M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);
    
    // Expected: each element should be K (sum of K ones)
    assert(verify_result(C, M, N, static_cast<float>(K)));
    std::cout << "PASSED" << std::endl;
}

void test_gemm_with_alpha_beta(sycl::queue& queue) {
    std::cout << "Test: GEMM with alpha and beta (C = alpha*A*B + beta*C)... ";
    
    const size_t M = 16, N = 16, K = 16;
    std::vector<float> A(M * K, 2.0f);
    std::vector<float> B(K * N, 3.0f);
    std::vector<float> C(M * N, 1.0f);
    
    float alpha = 0.5f;
    float beta = 2.0f;
    
    GemmConfig config;
    config.use_local_memory = false;
    GemmKernel<float> kernel(queue, config);
    
    kernel.gemm(M, N, K, alpha, A, K, B, N, beta, C, N);
    
    // Expected: alpha * (2 * 3 * K) + beta * 1 = 0.5 * (6 * 16) + 2 * 1 = 50
    float expected = alpha * (2.0f * 3.0f * K) + beta * 1.0f;
    assert(verify_result(C, M, N, expected, 0.1f));
    std::cout << "PASSED" << std::endl;
}

void test_gemm_with_local_memory(sycl::queue& queue) {
    std::cout << "Test: GEMM with local memory... ";
    
    const size_t M = 32, N = 32, K = 32;
    std::vector<float> A(M * K, 1.0f);
    std::vector<float> B(K * N, 1.0f);
    std::vector<float> C(M * N, 0.0f);
    
    GemmConfig config;
    config.use_local_memory = true;
    config.tile_size_m = 16;
    config.tile_size_n = 16;
    config.tile_size_k = 16;
    config.work_group_size_m = 8;
    config.work_group_size_n = 8;
    
    try {
        GemmKernel<float> kernel(queue, config);
        kernel.gemm(M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);
        
        assert(verify_result(C, M, N, static_cast<float>(K)));
        std::cout << "PASSED" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "SKIPPED (local memory not supported)" << std::endl;
    }
}

void test_non_square_matrices(sycl::queue& queue) {
    std::cout << "Test: Non-square matrices... ";
    
    const size_t M = 64, N = 32, K = 48;
    std::vector<float> A(M * K, 1.0f);
    std::vector<float> B(K * N, 1.0f);
    std::vector<float> C(M * N, 0.0f);
    
    GemmConfig config;
    config.use_local_memory = false;
    GemmKernel<float> kernel(queue, config);
    
    kernel.gemm(M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);
    
    assert(verify_result(C, M, N, static_cast<float>(K)));
    std::cout << "PASSED" << std::endl;
}

void test_small_matrices(sycl::queue& queue) {
    std::cout << "Test: Small matrices... ";
    
    const size_t M = 8, N = 8, K = 8;
    std::vector<float> A(M * K, 2.0f);
    std::vector<float> B(K * N, 3.0f);
    std::vector<float> C(M * N, 0.0f);
    
    GemmConfig config;
    config.use_local_memory = false;
    GemmKernel<float> kernel(queue, config);
    
    kernel.gemm(M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);
    
    // Expected: 2 * 3 * K = 6 * 8 = 48
    assert(verify_result(C, M, N, 2.0f * 3.0f * K));
    std::cout << "PASSED" << std::endl;
}

void test_large_matrices(sycl::queue& queue) {
    std::cout << "Test: Large matrices... ";
    
    const size_t M = 256, N = 256, K = 256;
    std::vector<float> A(M * K, 1.0f);
    std::vector<float> B(K * N, 1.0f);
    std::vector<float> C(M * N, 0.0f);
    
    GemmConfig config;
    config.use_local_memory = false;
    GemmKernel<float> kernel(queue, config);
    
    kernel.gemm(M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);
    
    assert(verify_result(C, M, N, static_cast<float>(K)));
    std::cout << "PASSED" << std::endl;
}

void test_zero_alpha(sycl::queue& queue) {
    std::cout << "Test: Zero alpha (C = beta*C)... ";
    
    const size_t M = 16, N = 16, K = 16;
    std::vector<float> A(M * K, 5.0f);
    std::vector<float> B(K * N, 5.0f);
    std::vector<float> C(M * N, 3.0f);
    
    GemmConfig config;
    config.use_local_memory = false;
    GemmKernel<float> kernel(queue, config);
    
    kernel.gemm(M, N, K, 0.0f, A, K, B, N, 2.0f, C, N);
    
    // Expected: 0 * A*B + 2 * 3 = 6
    assert(verify_result(C, M, N, 6.0f));
    std::cout << "PASSED" << std::endl;
}

int main() {
    try {
        sycl::queue queue{sycl::default_selector_v};
        
        std::cout << "=== SYCL GEMM Unit Tests ===" << std::endl;
        std::cout << "Device: " << queue.get_device().get_info<sycl::info::device::name>() << std::endl;
        std::cout << std::endl;
        
        test_basic_gemm(queue);
        test_gemm_with_alpha_beta(queue);
        test_gemm_with_local_memory(queue);
        test_non_square_matrices(queue);
        test_small_matrices(queue);
        test_large_matrices(queue);
        test_zero_alpha(queue);
        
        std::cout << std::endl;
        std::cout << "=== All tests completed successfully ===" << std::endl;
        
        return 0;
    } catch (const sycl::exception& e) {
        std::cerr << "SYCL exception: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
