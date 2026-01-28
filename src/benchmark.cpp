#include "gemm_kernel.hpp"
#include "auto_tuner.hpp"
#include <CL/sycl.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cstdlib>

using namespace sycl_gemm;

/**
 * @brief Benchmark GEMM with different configurations
 */
template<typename T>
void benchmark_gemm(sycl::queue& queue, size_t M, size_t N, size_t K, size_t iterations = 100) {
    std::cout << "\n=== Benchmarking GEMM " << M << "x" << N << "x" << K << " ===" << std::endl;
    
    // Initialize matrices
    std::vector<T> A(M * K);
    std::vector<T> B(K * N);
    std::vector<T> C(M * N);
    
    // Fill with test data
    for (size_t i = 0; i < M * K; i++) A[i] = static_cast<T>(rand()) / RAND_MAX;
    for (size_t i = 0; i < K * N; i++) B[i] = static_cast<T>(rand()) / RAND_MAX;
    
    T alpha = 1.0;
    T beta = 0.0;
    
    // Test different configurations
    std::vector<GemmConfig> configs;
    
    // Config 1: Small tiles, no local memory
    GemmConfig config1;
    config1.tile_size_m = 8;
    config1.tile_size_n = 8;
    config1.tile_size_k = 8;
    config1.work_group_size_m = 4;
    config1.work_group_size_n = 4;
    config1.use_local_memory = false;
    configs.push_back(config1);
    
    // Config 2: Medium tiles, no local memory
    GemmConfig config2;
    config2.tile_size_m = 16;
    config2.tile_size_n = 16;
    config2.tile_size_k = 16;
    config2.work_group_size_m = 8;
    config2.work_group_size_n = 8;
    config2.use_local_memory = false;
    configs.push_back(config2);
    
    // Config 3: Medium tiles, with local memory
    GemmConfig config3;
    config3.tile_size_m = 16;
    config3.tile_size_n = 16;
    config3.tile_size_k = 16;
    config3.work_group_size_m = 8;
    config3.work_group_size_n = 8;
    config3.use_local_memory = true;
    configs.push_back(config3);
    
    std::cout << std::fixed << std::setprecision(3);
    
    for (size_t idx = 0; idx < configs.size(); idx++) {
        const auto& config = configs[idx];
        
        try {
            GemmKernel<T> kernel(queue, config);
            
            // Warm-up
            std::fill(C.begin(), C.end(), 0.0);
            kernel.gemm(M, N, K, alpha, A, K, B, N, beta, C, N);
            
            // Benchmark
            auto start = std::chrono::high_resolution_clock::now();
            
            for (size_t i = 0; i < iterations; i++) {
                std::fill(C.begin(), C.end(), 0.0);
                kernel.gemm(M, N, K, alpha, A, K, B, N, beta, C, N);
            }
            
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> elapsed = end - start;
            double avg_time = elapsed.count() / iterations;
            
            // Calculate GFLOPS
            double gflops = (2.0 * M * N * K) / (avg_time * 1e6);
            
            std::cout << "Config " << (idx + 1) << ": "
                      << "tile=" << config.tile_size_m << "x" << config.tile_size_n
                      << ", wg=" << config.work_group_size_m << "x" << config.work_group_size_n
                      << ", local=" << (config.use_local_memory ? "yes" : "no")
                      << " -> " << avg_time << " ms, " << gflops << " GFLOPS" << std::endl;
        } catch (const std::exception& e) {
            std::cout << "Config " << (idx + 1) << " failed: " << e.what() << std::endl;
        }
    }
}

int main(int argc, char** argv) {
    try {
        // Create SYCL queue
        sycl::queue queue{sycl::default_selector_v};
        
        std::cout << "SYCL GEMM Benchmark" << std::endl;
        std::cout << "===================" << std::endl;
        
        // Print device info
        AutoTuner<float> tuner(queue);
        tuner.print_device_info();
        
        // Benchmark different matrix sizes
        benchmark_gemm<float>(queue, 128, 128, 128, 50);
        benchmark_gemm<float>(queue, 256, 256, 256, 20);
        benchmark_gemm<float>(queue, 512, 512, 512, 10);
        
        // Auto-tuning example
        std::cout << "\n=== Auto-tuning Example ===" << std::endl;
        GemmConfig best_config = tuner.auto_tune(256, 256, 256, 5);
        
        std::cout << "\n=== Testing with best configuration ===" << std::endl;
        std::vector<float> A(256 * 256, 1.0f);
        std::vector<float> B(256 * 256, 1.0f);
        std::vector<float> C(256 * 256, 0.0f);
        
        GemmKernel<float> kernel(queue, best_config);
        auto start = std::chrono::high_resolution_clock::now();
        kernel.gemm(256, 256, 256, 1.0f, A, 256, B, 256, 0.0f, C, 256);
        auto end = std::chrono::high_resolution_clock::now();
        
        std::chrono::duration<double, std::milli> elapsed = end - start;
        double gflops = (2.0 * 256 * 256 * 256) / (elapsed.count() * 1e6);
        
        std::cout << "Time: " << elapsed.count() << " ms" << std::endl;
        std::cout << "Performance: " << gflops << " GFLOPS" << std::endl;
        
    } catch (const sycl::exception& e) {
        std::cerr << "SYCL exception: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
