#ifndef AUTO_TUNER_HPP
#define AUTO_TUNER_HPP

#include "gemm_kernel.hpp"
#include <CL/sycl.hpp>
#include <chrono>
#include <vector>
#include <algorithm>
#include <iostream>

namespace sycl_gemm {

/**
 * @brief Auto-tuner for GEMM kernel configuration
 */
template<typename T>
class AutoTuner {
public:
    AutoTuner(sycl::queue& q) : queue_(q) {
        query_device_properties();
    }

    /**
     * @brief Auto-tune GEMM configuration for given matrix dimensions
     * @param M Number of rows in A and C
     * @param N Number of columns in B and C
     * @param K Number of columns in A and rows in B
     * @param num_iterations Number of iterations for benchmarking
     * @return Best configuration found
     */
    GemmConfig auto_tune(size_t M, size_t N, size_t K, size_t num_iterations = 10) {
        std::cout << "Auto-tuning GEMM configuration for size " 
                  << M << "x" << N << "x" << K << std::endl;
        
        std::vector<GemmConfig> configs = generate_configs();
        
        GemmConfig best_config;
        double best_time = std::numeric_limits<double>::max();

        // Test data
        std::vector<T> A(M * K, 1.0);
        std::vector<T> B(K * N, 1.0);
        std::vector<T> C(M * N, 0.0);

        for (const auto& config : configs) {
            try {
                double avg_time = benchmark_config(M, N, K, config, A, B, C, num_iterations);
                
                std::cout << "Config: tile=" << config.tile_size_m << "x" << config.tile_size_n
                          << ", wg=" << config.work_group_size_m << "x" << config.work_group_size_n
                          << ", local_mem=" << config.use_local_memory
                          << " -> " << avg_time << " ms" << std::endl;

                if (avg_time < best_time) {
                    best_time = avg_time;
                    best_config = config;
                }
            } catch (const std::exception& e) {
                // Configuration not supported, skip it
                continue;
            }
        }

        std::cout << "Best configuration: tile=" << best_config.tile_size_m 
                  << "x" << best_config.tile_size_n
                  << ", wg=" << best_config.work_group_size_m 
                  << "x" << best_config.work_group_size_n
                  << ", local_mem=" << best_config.use_local_memory
                  << " (" << best_time << " ms)" << std::endl;

        return best_config;
    }

    /**
     * @brief Get default configuration based on device properties
     */
    GemmConfig get_default_config() const {
        GemmConfig config;
        
        // Adjust based on device capabilities
        if (max_work_group_size_ >= 256) {
            config.work_group_size_m = 16;
            config.work_group_size_n = 16;
            config.tile_size_m = 32;
            config.tile_size_n = 32;
            config.tile_size_k = 16;
        } else if (max_work_group_size_ >= 64) {
            config.work_group_size_m = 8;
            config.work_group_size_n = 8;
            config.tile_size_m = 16;
            config.tile_size_n = 16;
            config.tile_size_k = 16;
        } else {
            config.work_group_size_m = 4;
            config.work_group_size_n = 4;
            config.tile_size_m = 8;
            config.tile_size_n = 8;
            config.tile_size_k = 8;
        }
        
        config.use_local_memory = has_local_memory_;
        
        return config;
    }

    /**
     * @brief Print device information
     */
    void print_device_info() const {
        auto device = queue_.get_device();
        std::cout << "Device: " << device.get_info<sycl::info::device::name>() << std::endl;
        std::cout << "Max work-group size: " << max_work_group_size_ << std::endl;
        std::cout << "Max compute units: " << max_compute_units_ << std::endl;
        std::cout << "Local memory size: " << local_memory_size_ << " bytes" << std::endl;
        std::cout << "Global memory size: " << global_memory_size_ << " bytes" << std::endl;
    }

private:
    sycl::queue& queue_;
    size_t max_work_group_size_;
    size_t max_compute_units_;
    size_t local_memory_size_;
    size_t global_memory_size_;
    bool has_local_memory_;

    void query_device_properties() {
        auto device = queue_.get_device();
        
        max_work_group_size_ = device.get_info<sycl::info::device::max_work_group_size>();
        max_compute_units_ = device.get_info<sycl::info::device::max_compute_units>();
        local_memory_size_ = device.get_info<sycl::info::device::local_mem_size>();
        global_memory_size_ = device.get_info<sycl::info::device::global_mem_size>();
        
        has_local_memory_ = (local_memory_size_ > 0);
    }

    std::vector<GemmConfig> generate_configs() {
        std::vector<GemmConfig> configs;
        
        // Tile sizes to try
        std::vector<size_t> tile_sizes = {8, 16, 32};
        // Work-group sizes to try
        std::vector<size_t> wg_sizes = {4, 8, 16};
        
        for (size_t tile : tile_sizes) {
            for (size_t wg : wg_sizes) {
                // Skip if work-group size exceeds device limit
                if (wg * wg > max_work_group_size_) {
                    continue;
                }
                
                GemmConfig config;
                config.tile_size_m = tile;
                config.tile_size_n = tile;
                config.tile_size_k = 16;
                config.work_group_size_m = wg;
                config.work_group_size_n = wg;
                
                // Try with and without local memory
                if (has_local_memory_) {
                    config.use_local_memory = true;
                    configs.push_back(config);
                }
                
                config.use_local_memory = false;
                configs.push_back(config);
            }
        }
        
        return configs;
    }

    double benchmark_config(size_t M, size_t N, size_t K,
                           const GemmConfig& config,
                           const std::vector<T>& A,
                           const std::vector<T>& B,
                           std::vector<T>& C,
                           size_t num_iterations) {
        GemmKernel<T> kernel(queue_, config);
        
        // Warm-up run
        std::fill(C.begin(), C.end(), 0.0);
        kernel.gemm(M, N, K, 1.0, A, K, B, N, 0.0, C, N);
        
        // Benchmark runs
        auto start = std::chrono::high_resolution_clock::now();
        
        for (size_t i = 0; i < num_iterations; i++) {
            std::fill(C.begin(), C.end(), 0.0);
            kernel.gemm(M, N, K, 1.0, A, K, B, N, 0.0, C, N);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;
        
        return elapsed.count() / num_iterations;
    }
};

} // namespace sycl_gemm

#endif // AUTO_TUNER_HPP
