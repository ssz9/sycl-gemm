#ifndef GEMM_KERNEL_HPP
#define GEMM_KERNEL_HPP

#include <CL/sycl.hpp>
#include <vector>
#include <stdexcept>

namespace sycl_gemm {

/**
 * @brief Configuration parameters for GEMM kernel
 */
struct GemmConfig {
    size_t tile_size_m = 16;      // Tile size in M dimension
    size_t tile_size_n = 16;      // Tile size in N dimension
    size_t tile_size_k = 16;      // Tile size in K dimension
    size_t work_group_size_m = 8; // Work-group size in M dimension
    size_t work_group_size_n = 8; // Work-group size in N dimension
    bool use_local_memory = true; // Use local memory for tiling
    size_t unroll_factor = 1;     // Loop unrolling factor
};

/**
 * @brief SYCL GEMM kernel implementation
 * Performs C = alpha * A * B + beta * C
 */
template<typename T>
class GemmKernel {
public:
    GemmKernel(sycl::queue& q, const GemmConfig& config = GemmConfig())
        : queue_(q), config_(config) {}

    /**
     * @brief Execute GEMM operation
     * @param M Number of rows in A and C
     * @param N Number of columns in B and C
     * @param K Number of columns in A and rows in B
     * @param alpha Scalar multiplier for A*B
     * @param A Input matrix A (M x K)
     * @param lda Leading dimension of A
     * @param B Input matrix B (K x N)
     * @param ldb Leading dimension of B
     * @param beta Scalar multiplier for C
     * @param C Input/output matrix C (M x N)
     * @param ldc Leading dimension of C
     */
    void gemm(size_t M, size_t N, size_t K,
              T alpha,
              const std::vector<T>& A, size_t lda,
              const std::vector<T>& B, size_t ldb,
              T beta,
              std::vector<T>& C, size_t ldc) {
        
        // Validate inputs
        if (A.size() < M * lda || B.size() < K * ldb || C.size() < M * ldc) {
            throw std::invalid_argument("Matrix dimensions do not match buffer sizes");
        }

        // Allocate device buffers
        sycl::buffer<T, 1> buf_A(A.data(), sycl::range<1>(M * lda));
        sycl::buffer<T, 1> buf_B(B.data(), sycl::range<1>(K * ldb));
        sycl::buffer<T, 1> buf_C(C.data(), sycl::range<1>(M * ldc));

        if (config_.use_local_memory) {
            gemm_local_memory(M, N, K, alpha, buf_A, lda, buf_B, ldb, beta, buf_C, ldc);
        } else {
            gemm_global_memory(M, N, K, alpha, buf_A, lda, buf_B, ldb, beta, buf_C, ldc);
        }
    }

private:
    sycl::queue& queue_;
    GemmConfig config_;

    void gemm_local_memory(size_t M, size_t N, size_t K,
                          T alpha,
                          sycl::buffer<T, 1>& buf_A, size_t lda,
                          sycl::buffer<T, 1>& buf_B, size_t ldb,
                          T beta,
                          sycl::buffer<T, 1>& buf_C, size_t ldc) {
        
        const size_t tile_m = config_.tile_size_m;
        const size_t tile_n = config_.tile_size_n;
        const size_t tile_k = config_.tile_size_k;
        const size_t wg_m = config_.work_group_size_m;
        const size_t wg_n = config_.work_group_size_n;

        sycl::range<2> global_size((M + tile_m - 1) / tile_m * wg_m,
                                   (N + tile_n - 1) / tile_n * wg_n);
        sycl::range<2> local_size(wg_m, wg_n);

        queue_.submit([&](sycl::handler& cgh) {
            auto acc_A = buf_A.template get_access<sycl::access::mode::read>(cgh);
            auto acc_B = buf_B.template get_access<sycl::access::mode::read>(cgh);
            auto acc_C = buf_C.template get_access<sycl::access::mode::read_write>(cgh);

            sycl::local_accessor<T, 2> tile_A({tile_m, tile_k}, cgh);
            sycl::local_accessor<T, 2> tile_B({tile_k, tile_n}, cgh);

            cgh.parallel_for(sycl::nd_range<2>(global_size, local_size),
                [=](sycl::nd_item<2> item) {
                    const size_t local_m = item.get_local_id(0);
                    const size_t local_n = item.get_local_id(1);
                    const size_t group_m = item.get_group(0);
                    const size_t group_n = item.get_group(1);
                    
                    const size_t global_m = group_m * tile_m + local_m * (tile_m / wg_m);
                    const size_t global_n = group_n * tile_n + local_n * (tile_n / wg_n);

                    T sum = 0;

                    // Tile across K dimension
                    for (size_t k_tile = 0; k_tile < K; k_tile += tile_k) {
                        // Load tile_A
                        for (size_t i = local_m; i < tile_m; i += wg_m) {
                            for (size_t k = local_n; k < tile_k; k += wg_n) {
                                size_t row = global_m - local_m * (tile_m / wg_m) + i;
                                size_t col = k_tile + k;
                                if (row < M && col < K) {
                                    tile_A[i][k] = acc_A[row * lda + col];
                                } else {
                                    tile_A[i][k] = 0;
                                }
                            }
                        }

                        // Load tile_B
                        for (size_t k = local_m; k < tile_k; k += wg_m) {
                            for (size_t j = local_n; j < tile_n; j += wg_n) {
                                size_t row = k_tile + k;
                                size_t col = global_n - local_n * (tile_n / wg_n) + j;
                                if (row < K && col < N) {
                                    tile_B[k][j] = acc_B[row * ldb + col];
                                } else {
                                    tile_B[k][j] = 0;
                                }
                            }
                        }

                        item.barrier(sycl::access::fence_space::local_space);

                        // Compute partial results
                        for (size_t i = 0; i < tile_m / wg_m; i++) {
                            for (size_t j = 0; j < tile_n / wg_n; j++) {
                                for (size_t k = 0; k < tile_k; k++) {
                                    sum += tile_A[local_m * (tile_m / wg_m) + i][k] *
                                           tile_B[k][local_n * (tile_n / wg_n) + j];
                                }
                            }
                        }

                        item.barrier(sycl::access::fence_space::local_space);
                    }

                    // Write results
                    for (size_t i = 0; i < tile_m / wg_m; i++) {
                        for (size_t j = 0; j < tile_n / wg_n; j++) {
                            size_t row = global_m + i;
                            size_t col = global_n + j;
                            if (row < M && col < N) {
                                size_t idx = row * ldc + col;
                                acc_C[idx] = alpha * sum + beta * acc_C[idx];
                            }
                        }
                    }
                });
        }).wait();
    }

    void gemm_global_memory(size_t M, size_t N, size_t K,
                           T alpha,
                           sycl::buffer<T, 1>& buf_A, size_t lda,
                           sycl::buffer<T, 1>& buf_B, size_t ldb,
                           T beta,
                           sycl::buffer<T, 1>& buf_C, size_t ldc) {
        
        queue_.submit([&](sycl::handler& cgh) {
            auto acc_A = buf_A.template get_access<sycl::access::mode::read>(cgh);
            auto acc_B = buf_B.template get_access<sycl::access::mode::read>(cgh);
            auto acc_C = buf_C.template get_access<sycl::access::mode::read_write>(cgh);

            cgh.parallel_for(sycl::range<2>(M, N),
                [=](sycl::id<2> idx) {
                    const size_t m = idx[0];
                    const size_t n = idx[1];
                    
                    T sum = 0;
                    for (size_t k = 0; k < K; k++) {
                        sum += acc_A[m * lda + k] * acc_B[k * ldb + n];
                    }
                    
                    acc_C[m * ldc + n] = alpha * sum + beta * acc_C[m * ldc + n];
                });
        }).wait();
    }
};

} // namespace sycl_gemm

#endif // GEMM_KERNEL_HPP
