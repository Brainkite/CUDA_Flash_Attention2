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

__global__ void flash_attention_kernel(const float* __restrict__ Q, const float* __restrict__ K, const float* __restrict__ V, 
                                       float* __restrict__ O, float* __restrict__ L, int BrDim, int BcDim, 
                                       int Bs, int Nh, int N, int dim) {
    bool is_main_thread = (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0 && threadIdx.y == 0);
    
    // consider  declaring as volatile
    extern __shared__ float shar[];
    float* Qs = &shar[0];
    float* Ks = &shar[BrDim * dim];
    float* Ss = &shar[BrDim * dim + BcDim * dim];
    float* Vs = &shar[BrDim * dim + BcDim * dim + BrDim * BcDim];
    float* Os = &shar[BrDim * dim + 2 * BcDim * dim + BrDim * BcDim];
    float* ls = &shar[2 * BrDim * dim + 2 * BcDim * dim + BrDim * BcDim];
    float* ms = &shar[2 * BrDim * dim + 2 * BcDim * dim + BrDim * BcDim + BrDim];
    float* expMaxDelta = &shar[2 * BrDim * dim + 2 * BcDim * dim + BrDim * BcDim + 2 * BrDim];

    const int BrIdx = blockIdx.x;
    const int NhIdx = blockIdx.y;
    const int BsIdx = blockIdx.z;
    const int TcDim = blockDim.x;
    const int TrDim = blockDim.y;
    const int TcIdx = threadIdx.x;
    const int TrIdx = threadIdx.y;

    const int global_offset = BsIdx * Nh * N * dim + NhIdx * N * dim;

    if (is_main_thread) {
        printf("Kernel started. Block dimensions: (%d, %d, %d), Thread dimensions: (%d, %d)\n", 
               gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y);
    }

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

    if (is_main_thread) {
        printf("Q loaded into shared memory\n");
    }
    
    const int num_Bc = (N + BcDim - 1) / BcDim;
    const float sqrt_dim = sqrtf(dim);
    for (int BcIdx = 0; BcIdx < num_Bc; ++BcIdx) {
        if (is_main_thread) {
            printf("Computing partial output for BcIdx: %d\n", BcIdx);
        }
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

        if (is_main_thread) {
            printf("K and V loaded into shared memory for BcIdx: %d\n", BcIdx);
        }

        // Compute attention scores
        for (int i = TrIdx; i < BrDim; i += TrDim) {
            for (int j = TcIdx; j < BcDim; j += TcDim) {
                if ((BrIdx * BrDim + i < N) && (BcIdx * BcDim + j < N)) {
                    float s = 0.0f;
                    for (int k = 0; k < dim; ++k) {
                        s += Qs[i * dim + k] * Ks[j * dim + k];
                    }
                    Ss[i * BcDim + j] = s / sqrt_dim;
                }
            }
        }
        __syncthreads();

        if (is_main_thread) {
            printf("Attention scores computed for BcIdx: %d\n", BcIdx);
        }

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

        if (is_main_thread) {
            printf("mi and Pi computed for BcIdx: %d\n", BcIdx);
        }

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
                            pv += Ss[i * BcDim + k] * Vs[k * dim + j];
                        }
                    }
                    Os[i * dim + j] = Os[i * dim + j] * expMaxDelta[i] + pv;
                }
            }
        }
        __syncthreads();

        if (is_main_thread) {
            printf("li and Oi computed for BcIdx: %d\n", BcIdx);
        }

    }

    // Update final Oi and li
    for (int i = TrIdx; i < BrDim; i += TrDim) {
        if (BrIdx * BrDim + i < N) {
            for (int j = TcIdx; j < dim; j += TcDim) {
                Os[i * dim + j] /= ls[i];
            }
            if (TcIdx == 0) {
                ls[i] = ms[i] + logf(ls[i]);
            }
        }
    }
    __syncthreads();

    if (is_main_thread) {
        printf("Final Oi and li updated\n");
    }

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

    if (is_main_thread) {
        printf("Output written to global memory\n");
    }
}

// Host function to perform flash attention with verbose output
void flash_attention(const float* Q, const float* K, const float* V, float* O, float* L,
                     int BrDim, int BcDim, int Bs, int Nh, int N, int dim) {
    std::cout << "Starting flash attention with parameters:" << std::endl;
    std::cout << "Batch Size (Bs): " << Bs << std::endl;
    std::cout << "Number of Heads (Nh): " << Nh << std::endl;
    std::cout << "Sequence Length (N): " << N << std::endl;
    std::cout << "Dimension (dim): " << dim << std::endl;
    std::cout << "Block Dimension for Rows (BrDim): " << BrDim << std::endl;
    std::cout << "Block Dimension for Columns (BcDim): " << BcDim << std::endl;

    dim3 threadsPerBlock(32, 4);
    dim3 numBlocks(cdiv(N, BrDim), Nh, Bs);

    size_t sharedMemSize = (2 * (BrDim * dim) + 2 * (BcDim * dim) + (BrDim * BcDim) + 3 * BrDim) * sizeof(float);

    std::cout << "### Launching kernel with " << numBlocks.x << " x " << numBlocks.y << " x " << numBlocks.z << " blocks and " << threadsPerBlock.x << " x " << threadsPerBlock.y << " threads per block." << std::endl;
    std::cout << "Shared memory size per block: " << sharedMemSize << " bytes." << std::endl;

    flash_attention_kernel<<<numBlocks, threadsPerBlock, sharedMemSize>>>(
        Q, K, V, O, L, BrDim, BcDim, Bs, Nh, N, dim);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    std::cout << "Flash attention kernel launched successfully." << std::endl;
}

// Main function for testing with verbose output
int main() {
    // Test parameters
    int Bs = 2;
    int Nh = 2;
    int N = 1024;
    int dim = 32;

    // Calculate BrDim and BcDim based on shared memory constraints
    int max_shared_mem_bytes;
    CHECK_CUDA(cudaDeviceGetAttribute(&max_shared_mem_bytes, cudaDevAttrMaxSharedMemoryPerBlock, 0));
    int BcDim = std::min(cdiv(max_shared_mem_bytes, (sizeof(float) * 4 * dim)), N);
    int BrDim = std::min(dim, BcDim);
    size_t shared_mem_size = (2 * (BrDim * dim) + 2 * (BcDim * dim) + (BrDim * BcDim) + 3 * BrDim) * sizeof(float);
    while (shared_mem_size > max_shared_mem_bytes) {
        BcDim -= 1;
        BrDim = std::min(dim, BcDim);
        shared_mem_size = (2 * (BrDim * dim) + 2 * (BcDim * dim) + (BrDim * BcDim) + 3 * BrDim) * sizeof(float);
    }

    std::cout << "Calculated block dimensions based on shared memory constraints:" << std::endl;
    std::cout << "max_shared_mem_bytes: " << max_shared_mem_bytes << std::endl;
    std::cout << "shared_mem_size: " << shared_mem_size << std::endl;
    std::cout << "BcDim: " << BcDim << std::endl;
    std::cout << "BrDim: " << BrDim << std::endl;

    // Allocate device memory
    float *d_Q, *d_K, *d_V, *d_O, *d_L;
    size_t size = Bs * Nh * N * dim * sizeof(float);
    CHECK_CUDA(cudaMalloc(&d_Q, size));
    CHECK_CUDA(cudaMalloc(&d_K, size));
    CHECK_CUDA(cudaMalloc(&d_V, size));
    CHECK_CUDA(cudaMalloc(&d_O, size));
    CHECK_CUDA(cudaMalloc(&d_L, Bs * Nh * N * sizeof(float)));

    std::cout << "Device memory allocation successful." << std::endl;

    // Initialize input data with random values
    curandGenerator_t gen;
    CHECK_CURAND(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL));
    CHECK_CURAND(curandGenerateUniform(gen, d_Q, Bs * Nh * N * dim));
    CHECK_CURAND(curandGenerateUniform(gen, d_K, Bs * Nh * N * dim));
    CHECK_CURAND(curandGenerateUniform(gen, d_V, Bs * Nh * N * dim));
    CHECK_CURAND(curandDestroyGenerator(gen));

    std::cout << "Input data initialized with random values." << std::endl;

    // Run flash attention
    try {
        flash_attention(d_Q, d_K, d_V, d_O, d_L, BrDim, BcDim, Bs, Nh, N, dim);
        std::cout << "Flash attention completed successfully." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error occurred: " << e.what() << std::endl;
    }

    // Clean up
    CHECK_CUDA(cudaFree(d_Q));
    CHECK_CUDA(cudaFree(d_K));
    CHECK_CUDA(cudaFree(d_V));
    CHECK_CUDA(cudaFree(d_O));
    CHECK_CUDA(cudaFree(d_L));

    std::cout << "Device memory deallocation successful." << std::endl;

    return 0;
}