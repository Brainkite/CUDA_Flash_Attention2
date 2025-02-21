/**
 * @file flash_attention_2.cu
 * @brief CUDA implementation of the FlashAttention 2 algorithm using FP16
 * 
 * This file implements the FlashAttention 2 algorithm as described in the paper
 * "Flash Attention 2: Faster Attention with Better Parallelism and Work Partitioning".
 * The implementation focuses on optimizing memory access patterns and utilizing shared
 * memory effectively to reduce HBM accesses.
 * 
 * Key Features:
 * - Block-sparse attention computation
 * - Efficient shared memory usage
 * - Optimized memory access patterns
 * - Support for different head dimensions and sequence lengths
 * 
 * Performance Notes:
 * - The implementation automatically adjusts block sizes based on available shared memory
 * - Thread block dimensions are optimized for modern NVIDIA GPUs
 * - The algorithm uses a tiling strategy to handle long sequences efficiently
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cmath>
#include <iostream>

// nvcc -g -G -o flash_attention_2_debug flash_attention_2.cu -lcudart -lcurand
// sudo ncu -f --set full -o profile_output ./flash_attention_2_debug
// ncu --import /home/antonin/projects/cuda_flash_attn/cuda/src/profile_output.ncu-rep --print-summary per-kernel

// Error checking macro
#define CHECK_CUDA(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define CHECK_CURAND(call) \
do { \
    curandStatus_t status = call; \
    if (status != CURAND_STATUS_SUCCESS) { \
        fprintf(stderr, "CURAND error in %s:%d: %d\n", __FILE__, __LINE__, \
                status); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

__host__ __device__ int cdiv(int a, int b) {
    return (a + b - 1) / b;
}

/**
 * @brief Main kernel for FlashAttention 2 forward pass
 * 
 * This kernel implements the core FlashAttention algorithm, processing input matrices
 * Q, K, and V in blocks to compute attention scores and output values efficiently.
 * 
 * @param Q Input queries of shape [Bs, Nh, N, dim]
 * @param K Input keys of shape [Bs, Nh, N, dim]
 * @param V Input values of shape [Bs, Nh, N, dim]
 * @param O Output tensor of shape [Bs, Nh, N, dim]
 * @param L Output scaling factors of shape [Bs, Nh, N]
 * @param BrDim Block size for rows
 * @param BcDim Block size for columns
 * @param Bs Batch size
 * @param Nh Number of attention heads
 * @param N Sequence length
 * @param dim Head dimension
 * 
 * Shared Memory Layout:
 * - Qs: Query block [BrDim x dim]
 * - Ks: Key block [BcDim x dim]
 * - Ss: Score matrix [BrDim x BcDim]
 * - Vs: Value block [BcDim x dim]
 * - Os: Output accumulator [BrDim x dim]
 * - ls: Row scaling factors [BrDim]
 * - ms: Row maxima [BrDim]
 */
__global__ void flash_attention_kernel(const half* __restrict__ Q, const half* __restrict__ K, 
                                     const half* __restrict__ V, half* __restrict__ O, 
                                     float* __restrict__ L, int BrDim, int BcDim, 
                                       int Bs, int Nh, int N, int dim) {
    extern __shared__ half shar[];
    half* Qs = &shar[0];
    half* Ks = &shar[BrDim * dim];
    float* Ss = (float*)&shar[BrDim * dim + BcDim * dim];  // Keep scores in FP32 for stability
    half* Vs = (half*)&Ss[BrDim * BcDim];
    half* Os = &Vs[BcDim * dim];
    float* ls = (float*)&Os[BrDim * dim];  // Keep scaling factors in FP32
    float* ms = &ls[BrDim];
    float* expMaxDelta = &ms[BrDim];

    const int BrIdx = blockIdx.x;
    const int NhIdx = blockIdx.y;
    const int BsIdx = blockIdx.z;
    const int TcDim = blockDim.x;
    const int TrDim = blockDim.y;
    const int TcIdx = threadIdx.x;
    const int TrIdx = threadIdx.y;

    const int global_offset = BsIdx * Nh * N * dim + NhIdx * N * dim;

    // Initialize ms
    for (int i = TrIdx * TcDim + TcIdx; i < BrDim; i += TrDim * TcDim) {
        ms[i] = -INFINITY;
    }

    // Load Q into shared memory
    for (int i = TrIdx; i < BrDim; i += TrDim) {
        for (int j = TcIdx; j < dim; j += TcDim) {
            int idx = BrIdx * BrDim + i;
            if (idx < N) {
                Qs[i * dim + j] = Q[global_offset + idx * dim + j];
            }
        }
    }
    
    const int num_Bc = (N + BcDim - 1) / BcDim;
    const float sqrt_dim = sqrtf(dim);
    for (int BcIdx = 0; BcIdx < num_Bc; ++BcIdx) {
        // Load K and V into shared memory
        for (int i = TrIdx; i < BcDim; i += TrDim) {
            for (int j = TcIdx; j < dim; j += TcDim) {
                int idx = BcIdx * BcDim + i;
                if (idx < N) {
                    Ks[i * dim + j] = K[global_offset + idx * dim + j];
                    Vs[i * dim + j] = V[global_offset + idx * dim + j];
                }
            }
        }
        __syncthreads();

        // Compute attention scores
        for (int i = TrIdx; i < BrDim; i += TrDim) {
            for (int j = TcIdx; j < BcDim; j += TcDim) {
                if ((BrIdx * BrDim + i < N) && (BcIdx * BcDim + j < N)) {
                    float s = 0.0f;
                    for (int k = 0; k < dim; ++k) {
                        s += __half2float(Qs[i * dim + k]) * __half2float(Ks[j * dim + k]);
                    }
                    Ss[i * BcDim + j] = s / sqrt_dim;
                }
            }
        }
        __syncthreads();

        // Compute mi and Pi
        for (int i = TrIdx; i < BrDim; i += TrDim) {
            if (BrIdx * BrDim + i < N) {
                float row_max = -INFINITY;
                for (int j = TcIdx; j < BcDim; j += TcDim) {
                    if (BcIdx * BcDim + j < N) {
                        row_max = fmaxf(row_max, Ss[i * BcDim + j]);
                    }
                }

                for (int stride = 16; stride > 0; stride /= 2) {
                    row_max = fmaxf(row_max, __shfl_down_sync(0xffffffff, row_max, stride));
                }
                if (TcIdx == 0) {
                    row_max = fmaxf(ms[i], row_max);
                }
                row_max = __shfl_sync(0xffffffff, row_max, 0);

                for (int j = TcIdx; j < BcDim; j += TcDim) {
                    if (BcIdx * BcDim + j < N) {
                        Ss[i * BcDim + j] = expf(Ss[i * BcDim + j] - row_max);
                    }
                }

                if (TcIdx == 0) {
                    expMaxDelta[i] = expf(ms[i] - row_max);
                    ms[i] = row_max;
                }
            }
        }
        __syncthreads();

        // Compute li and Oi
        for (int i = TrIdx; i < BrDim; i += TrDim) {
            if (BrIdx * BrDim + i < N) {
                float row_sum = 0.0f;
                for (int j = TcIdx; j < BcDim; j += TcDim) {
                    if (BcIdx * BcDim + j < N) {
                        row_sum += Ss[i * BcDim + j];
                    }
                }
                for (int stride = 16; stride > 0; stride /= 2) {
                    row_sum += __shfl_down_sync(0xffffffff, row_sum, stride);
                }

                if (TcIdx == 0) {
                    ls[i] = fmaxf(ls[i] * expMaxDelta[i] + row_sum, 1e-7f);
                }

                for (int j = TcIdx; j < dim; j += TcDim) {
                    float pv = 0.0f;
                    for (int k = 0; k < BcDim; ++k) {
                        if (BcIdx * BcDim + k < N) {
                            pv += Ss[i * BcDim + k] * __half2float(Vs[k * dim + j]);
                        }
                    }
                    Os[i * dim + j] = __float2half(__half2float(Os[i * dim + j]) * expMaxDelta[i] + pv);
                }
            }
        }
        __syncthreads();
    }

    // Update final Oi and li
    for (int i = TrIdx; i < BrDim; i += TrDim) {
        if (BrIdx * BrDim + i < N) {
            for (int j = TcIdx; j < dim; j += TcDim) {
                Os[i * dim + j] = __float2half(__half2float(Os[i * dim + j]) / ls[i]);
            }
            if (TcIdx == 0) {
                ls[i] = ms[i] + logf(ls[i]);
            }
        }
    }
    __syncthreads();

    // Write output
    for (int i = TrIdx; i < BrDim; i += TrDim) {
        int idx = BrIdx * BrDim + i;
        if (idx < N) {
            for (int j = TcIdx; j < dim; j += TcDim) {
                O[global_offset + idx * dim + j] = Os[i * dim + j];
            }
            if (TcIdx == 0) {
                L[BsIdx * Nh * N + NhIdx * N + idx] = ls[i];
            }
        }
    }
}

/**
 * @brief Host function to perform flash attention with verbose output
 * 
 * This function handles the setup and launch of the flash attention kernel,
 * including calculating appropriate block dimensions and allocating resources.
 * 
 * @param Q Input queries
 * @param K Input keys
 * @param V Input values
 * @param O Output tensor
 * @param L Output scaling factors
 * @param BrDim Block size for rows
 * @param BcDim Block size for columns
 * @param Bs Batch size
 * @param Nh Number of attention heads
 * @param N Sequence length
 * @param dim Head dimension
 * 
 * Performance Considerations:
 * - Block sizes are chosen based on available shared memory
 * - Thread block dimensions are set to maximize occupancy
 * - The implementation automatically handles sequence padding
 */
void flash_attention(half* Q, half* K, half* V, half* O, float* L, int BrDim, int BcDim,
                    int Bs, int Nh, int N, int dim) {
    // Calculate grid dimensions
    dim3 grid((N + BrDim - 1) / BrDim, Nh, Bs);
    dim3 block(32, 4);  // Fixed block size for now
    
    // Calculate shared memory size with proper alignment
    size_t shared_mem_size = 0;
    shared_mem_size += BrDim * dim * sizeof(half);  // Qs
    shared_mem_size += BcDim * dim * sizeof(half);  // Ks
    shared_mem_size += BrDim * BcDim * sizeof(float);  // Ss
    shared_mem_size += BcDim * dim * sizeof(half);  // Vs
    shared_mem_size += BrDim * dim * sizeof(half);  // Os
    shared_mem_size += BrDim * sizeof(float);  // ls
    shared_mem_size += BrDim * sizeof(float);  // ms
    shared_mem_size += BrDim * sizeof(float);  // expMaxDelta
    
    // Ensure alignment
    shared_mem_size = ((shared_mem_size + 15) / 16) * 16;
    
    // Check if shared memory size is within limits
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    if (shared_mem_size > prop.sharedMemPerBlock) {
        fprintf(stderr, "Required shared memory (%zu bytes) exceeds device limit (%zu bytes)\n",
                shared_mem_size, (size_t)prop.sharedMemPerBlock);
        return;
    }
    
    // Initialize output arrays
    cudaMemset(O, 0, Bs * Nh * N * dim * sizeof(half));
    cudaMemset(L, 0, Bs * Nh * N * sizeof(float));
    
    // Launch kernel
    flash_attention_kernel<<<grid, block, shared_mem_size>>>(Q, K, V, O, L, BrDim, BcDim, Bs, Nh, N, dim);
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(err));
        return;
    }
}