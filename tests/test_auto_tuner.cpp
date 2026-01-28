#include "auto_tuner.hpp"
#include <CL/sycl.hpp>
#include <iostream>
#include <cassert>

using namespace sycl_gemm;

void test_device_query(sycl::queue& queue) {
    std::cout << "Test: Device query... ";
    
    AutoTuner<float> tuner(queue);
    
    // This should not throw
    tuner.print_device_info();
    
    std::cout << "PASSED" << std::endl;
}

void test_default_config(sycl::queue& queue) {
    std::cout << "Test: Default configuration generation... ";
    
    AutoTuner<float> tuner(queue);
    GemmConfig config = tuner.get_default_config();
    
    // Verify config has reasonable values
    assert(config.tile_size_m > 0);
    assert(config.tile_size_n > 0);
    assert(config.tile_size_k > 0);
    assert(config.work_group_size_m > 0);
    assert(config.work_group_size_n > 0);
    
    std::cout << "PASSED" << std::endl;
}

void test_auto_tuning(sycl::queue& queue) {
    std::cout << "Test: Auto-tuning (small scale)... ";
    
    AutoTuner<float> tuner(queue);
    
    // Run auto-tuning with small number of iterations
    GemmConfig best_config = tuner.auto_tune(64, 64, 64, 3);
    
    // Verify best config has valid values
    assert(best_config.tile_size_m > 0);
    assert(best_config.tile_size_n > 0);
    assert(best_config.work_group_size_m > 0);
    assert(best_config.work_group_size_n > 0);
    
    std::cout << "PASSED" << std::endl;
}

void test_config_validation(sycl::queue& queue) {
    std::cout << "Test: Configuration validation... ";
    
    AutoTuner<float> tuner(queue);
    
    // Test that auto-tuning works for different sizes
    tuner.auto_tune(32, 32, 32, 2);
    tuner.auto_tune(128, 128, 128, 2);
    
    std::cout << "PASSED" << std::endl;
}

int main() {
    try {
        sycl::queue queue{sycl::default_selector_v};
        
        std::cout << "=== Auto-Tuner Unit Tests ===" << std::endl;
        std::cout << "Device: " << queue.get_device().get_info<sycl::info::device::name>() << std::endl;
        std::cout << std::endl;
        
        test_device_query(queue);
        test_default_config(queue);
        test_auto_tuning(queue);
        test_config_validation(queue);
        
        std::cout << std::endl;
        std::cout << "=== All auto-tuner tests completed successfully ===" << std::endl;
        
        return 0;
    } catch (const sycl::exception& e) {
        std::cerr << "SYCL exception: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
