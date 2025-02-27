#include <cuda_runtime.h>
#include <curand.h>
#include <cuda_fp16.h>
#include <iostream>
#include <cassert>
#include <vector>
#include <cstring>
#include <cmath>
#include "../flash_attention_2.cu"

// Helper function to check CUDA errors
#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ << ": " \
                  << cudaGetErrorString(err) << std::endl; \
        exit(1); \
    } \
} while(0)

// Helper function to check cuRAND errors
#define CURAND_CHECK(call) \
do { \
    curandStatus_t status = call; \
    if (status != CURAND_STATUS_SUCCESS) { \
        std::cerr << "cuRAND error in " << __FILE__ << ":" << __LINE__ << ": " \
                  << status << std::endl; \
        exit(1); \
    } \
} while(0)

// Kernel to convert float data to half with scaling
__global__ void convert_to_half_kernel(half* data, float* float_data, float scale, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = __float2half(float_data[idx] * scale);
    }
}

// Helper function to initialize random data
void initialize_random_data(half* data, size_t size, float scale = 1.0f, curandGenerator_t gen = nullptr) {
    bool local_gen = false;
    curandGenerator_t local_generator;
    
    if (gen == nullptr) {
        local_gen = true;
        CURAND_CHECK(curandCreateGenerator(&local_generator, CURAND_RNG_PSEUDO_DEFAULT));
        CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(local_generator, 1234ULL));
        gen = local_generator;
    }
    
    // Generate random floats
    float* float_data;
    CUDA_CHECK(cudaMalloc(&float_data, size * sizeof(float)));
    CURAND_CHECK(curandGenerateUniform(gen, float_data, size));
    
    // Scale and convert to half using a kernel
    dim3 block(256);
    dim3 grid((size + block.x - 1) / block.x);
    
    convert_to_half_kernel<<<grid, block>>>(data, float_data, scale, size);
    
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(float_data));
    
    if (local_gen) {
        CURAND_CHECK(curandDestroyGenerator(local_generator));
    }
}

// Kernel to set specific values in the input tensors
__global__ void set_mixed_scale_values_kernel(half* Q, half* K, half* V, 
                                             int batch_size, int num_heads, 
                                             int seq_len, int head_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * num_heads * seq_len * head_dim) return;
    
    // Calculate position
    int pos = idx % (seq_len * head_dim);
    int seq_idx = pos / head_dim;
    int dim_idx = pos % head_dim;
    
    // Set different scales for different parts of the sequence
    if (seq_idx < seq_len / 4) {
        // First quarter: small values
        Q[idx] = __float2half(0.001f);
        K[idx] = __float2half(0.001f);
        V[idx] = __float2half(1.0f);
    } else if (seq_idx < seq_len / 2) {
        // Second quarter: medium values
        Q[idx] = __float2half(1.0f);
        K[idx] = __float2half(1.0f);
        V[idx] = __float2half(1.0f);
    } else if (seq_idx < 3 * seq_len / 4) {
        // Third quarter: large values
        Q[idx] = __float2half(100.0f);
        K[idx] = __float2half(0.01f);
        V[idx] = __float2half(1.0f);
    } else {
        // Fourth quarter: mixed values
        if (dim_idx % 4 == 0) {
            Q[idx] = __float2half(0.0001f);
            K[idx] = __float2half(1000.0f);
        } else if (dim_idx % 4 == 1) {
            Q[idx] = __float2half(10.0f);
            K[idx] = __float2half(0.1f);
        } else if (dim_idx % 4 == 2) {
            Q[idx] = __float2half(1.0f);
            K[idx] = __float2half(1.0f);
        } else {
            Q[idx] = __float2half(100.0f);
            K[idx] = __float2half(0.01f);
        }
        V[idx] = __float2half(1.0f);
    }
}

// Kernel to set extreme attention patterns
__global__ void set_extreme_attention_patterns_kernel(half* Q, half* K, half* V, 
                                                    int batch_size, int num_heads, 
                                                    int seq_len, int head_dim,
                                                    int pattern_type) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * num_heads * seq_len * head_dim) return;
    
    // Calculate position
    int pos = idx % (seq_len * head_dim);
    int seq_idx = pos / head_dim;
    int dim_idx = pos % head_dim;
    
    // Set different extreme attention patterns
    switch (pattern_type) {
        case 0: // Single token attention (first token attends strongly to itself)
            if (seq_idx == 0) {
                Q[idx] = __float2half(10.0f);
                K[idx] = __float2half(10.0f);
            } else {
                Q[idx] = __float2half(0.1f);
                K[idx] = __float2half(0.1f);
            }
            break;
            
        case 1: // Uniform attention (all tokens attend equally to all others)
            Q[idx] = __float2half(1.0f);
            K[idx] = __float2half(1.0f);
            break;
            
        case 2: // Alternating strong/weak attention
            if (seq_idx % 2 == 0) {
                Q[idx] = __float2half(5.0f);
                K[idx] = __float2half(5.0f);
            } else {
                Q[idx] = __float2half(0.2f);
                K[idx] = __float2half(0.2f);
            }
            break;
            
        case 3: // Exponentially decreasing attention
            {
                float scale = expf(-0.1f * seq_idx);
                Q[idx] = __float2half(scale);
                K[idx] = __float2half(scale);
            }
            break;
    }
    
    // Set V to 1.0 for all patterns
    V[idx] = __float2half(1.0f);
}

// Test 1: Basic NaN/Inf Detection Test
void test_nan_inf_detection() {
    std::cout << "Running Basic NaN/Inf Detection Test..." << std::endl;
    
    // Test with different configurations
    const int batch_sizes[] = {1, 2};
    const int num_heads[] = {1, 4};
    const int seq_lens[] = {32, 128};
    const int head_dims[] = {32, 64};
    const int block_dims[][2] = {{16, 16}, {32, 32}};
    
    curandGenerator_t generator;
    CURAND_CHECK(curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(generator, 1234ULL));
    
    for (int bs : batch_sizes) {
        for (int nh : num_heads) {
            for (int seq_len : seq_lens) {
                for (int head_dim : head_dims) {
                    for (auto& block_dim : block_dims) {
                        int BrDim = block_dim[0];
                        int BcDim = block_dim[1];
                        
                        // Skip if block dimensions are too large for sequence length
                        if (BrDim > seq_len || BcDim > seq_len) continue;
                        
                        std::cout << "  Testing configuration: bs=" << bs 
                                  << ", nh=" << nh 
                                  << ", seq_len=" << seq_len 
                                  << ", head_dim=" << head_dim 
                                  << ", BrDim=" << BrDim 
                                  << ", BcDim=" << BcDim << std::endl;
                        
                        // Calculate sizes
                        size_t tensor_size = bs * nh * seq_len * head_dim;
                        size_t tensor_bytes = tensor_size * sizeof(half);
                        size_t L_size = bs * nh * seq_len;
                        size_t L_bytes = L_size * sizeof(float);
                        
                        // Allocate device memory
                        half *d_Q, *d_K, *d_V, *d_O;
                        float *d_L;
                        
                        CUDA_CHECK(cudaMalloc(&d_Q, tensor_bytes));
                        CUDA_CHECK(cudaMalloc(&d_K, tensor_bytes));
                        CUDA_CHECK(cudaMalloc(&d_V, tensor_bytes));
                        CUDA_CHECK(cudaMalloc(&d_O, tensor_bytes));
                        CUDA_CHECK(cudaMalloc(&d_L, L_bytes));
                        
                        // Initialize with random data (scale 1.0)
                        initialize_random_data(d_Q, tensor_size, 1.0f, generator);
                        initialize_random_data(d_K, tensor_size, 1.0f, generator);
                        initialize_random_data(d_V, tensor_size, 1.0f, generator);
                        
                        // Initialize outputs with zeros
                        CUDA_CHECK(cudaMemset(d_O, 0, tensor_bytes));
                        CUDA_CHECK(cudaMemset(d_L, 0, L_bytes));
                        
                        // Run the kernel
                        flash_attention(d_Q, d_K, d_V, d_O, d_L, BrDim, BcDim, 
                                       bs, nh, seq_len, head_dim);
                        CUDA_CHECK(cudaDeviceSynchronize());
                        
                        // Copy outputs back to host
                        std::vector<half> h_O(tensor_size);
                        std::vector<float> h_L(L_size);
                        
                        CUDA_CHECK(cudaMemcpy(h_O.data(), d_O, tensor_bytes, cudaMemcpyDeviceToHost));
                        CUDA_CHECK(cudaMemcpy(h_L.data(), d_L, L_bytes, cudaMemcpyDeviceToHost));
                        
                        // Check for NaN or Inf in O
                        bool has_nan_inf_O = false;
                        for (size_t i = 0; i < h_O.size(); ++i) {
                            float val = __half2float(h_O[i]);
                            if (std::isnan(val) || std::isinf(val)) {
                                has_nan_inf_O = true;
                                std::cerr << "NaN or Inf detected in O at index " << i 
                                          << ": " << val << std::endl;
                                break;
                            }
                        }
                        
                        // Check for NaN or Inf in L
                        bool has_nan_inf_L = false;
                        for (size_t i = 0; i < h_L.size(); ++i) {
                            if (std::isnan(h_L[i]) || std::isinf(h_L[i])) {
                                has_nan_inf_L = true;
                                std::cerr << "NaN or Inf detected in L at index " << i 
                                          << ": " << h_L[i] << std::endl;
                                break;
                            }
                        }
                        
                        // Check for all zeros in O (which might indicate a problem)
                        bool all_zeros_O = true;
                        for (size_t i = 0; i < h_O.size() && all_zeros_O; ++i) {
                            if (__half2float(h_O[i]) != 0.0f) {
                                all_zeros_O = false;
                            }
                        }
                        
                        if (has_nan_inf_O || has_nan_inf_L || all_zeros_O) {
                            std::cerr << "✗ Test failed for this configuration" << std::endl;
                            if (has_nan_inf_O) std::cerr << "  - NaN or Inf detected in O" << std::endl;
                            if (has_nan_inf_L) std::cerr << "  - NaN or Inf detected in L" << std::endl;
                            if (all_zeros_O) std::cerr << "  - All zeros detected in O" << std::endl;
                            
                            // Clean up before exiting
                            CUDA_CHECK(cudaFree(d_Q));
                            CUDA_CHECK(cudaFree(d_K));
                            CUDA_CHECK(cudaFree(d_V));
                            CUDA_CHECK(cudaFree(d_O));
                            CUDA_CHECK(cudaFree(d_L));
                            CURAND_CHECK(curandDestroyGenerator(generator));
                            exit(1);
                        } else {
                            std::cout << "  ✓ No numerical issues detected for this configuration" << std::endl;
                        }
                        
                        // Clean up
                        CUDA_CHECK(cudaFree(d_Q));
                        CUDA_CHECK(cudaFree(d_K));
                        CUDA_CHECK(cudaFree(d_V));
                        CUDA_CHECK(cudaFree(d_O));
                        CUDA_CHECK(cudaFree(d_L));
                    }
                }
            }
        }
    }
    
    CURAND_CHECK(curandDestroyGenerator(generator));
    std::cout << "✓ Basic NaN/Inf Detection Test passed" << std::endl;
}

// Test 2: Zero Input Test
void test_zero_input() {
    std::cout << "Running Zero Input Test..." << std::endl;
    
    // Test with a subset of configurations
    const int batch_size = 1;
    const int num_heads = 1;
    const int seq_len = 32;
    const int head_dim = 64;
    const int BrDim = 16;
    const int BcDim = 16;
    
    // Calculate sizes
    size_t tensor_size = batch_size * num_heads * seq_len * head_dim;
    size_t tensor_bytes = tensor_size * sizeof(half);
    size_t L_size = batch_size * num_heads * seq_len;
    size_t L_bytes = L_size * sizeof(float);
    
    // Allocate device memory
    half *d_Q, *d_K, *d_V, *d_O;
    float *d_L;
    
    CUDA_CHECK(cudaMalloc(&d_Q, tensor_bytes));
    CUDA_CHECK(cudaMalloc(&d_K, tensor_bytes));
    CUDA_CHECK(cudaMalloc(&d_V, tensor_bytes));
    CUDA_CHECK(cudaMalloc(&d_O, tensor_bytes));
    CUDA_CHECK(cudaMalloc(&d_L, L_bytes));
    
    // Test cases with different scales
    const float scales[] = {0.0f, 1e-10f, 1e-5f};
    const char* scale_names[] = {"Zero", "Near-zero (1e-10)", "Small (1e-5)"};
    
    for (int i = 0; i < 3; ++i) {
        float scale = scales[i];
        std::cout << "  Testing with " << scale_names[i] << " inputs" << std::endl;
        
        // Initialize with scaled random data
        curandGenerator_t generator;
        CURAND_CHECK(curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT));
        CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(generator, 1234ULL));
        
        initialize_random_data(d_Q, tensor_size, scale, generator);
        initialize_random_data(d_K, tensor_size, scale, generator);
        initialize_random_data(d_V, tensor_size, scale, generator);
        
        // Initialize outputs with zeros
        CUDA_CHECK(cudaMemset(d_O, 0, tensor_bytes));
        CUDA_CHECK(cudaMemset(d_L, 0, L_bytes));
        
        // Run the kernel
        flash_attention(d_Q, d_K, d_V, d_O, d_L, BrDim, BcDim, 
                       batch_size, num_heads, seq_len, head_dim);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Copy outputs back to host
        std::vector<half> h_O(tensor_size);
        std::vector<float> h_L(L_size);
        
        CUDA_CHECK(cudaMemcpy(h_O.data(), d_O, tensor_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_L.data(), d_L, L_bytes, cudaMemcpyDeviceToHost));
        
        // Check for NaN or Inf in O
        bool has_nan_inf_O = false;
        for (size_t j = 0; j < h_O.size(); ++j) {
            float val = __half2float(h_O[j]);
            if (std::isnan(val) || std::isinf(val)) {
                has_nan_inf_O = true;
                std::cerr << "NaN or Inf detected in O at index " << j 
                          << ": " << val << std::endl;
                break;
            }
        }
        
        // Check for NaN or Inf in L
        bool has_nan_inf_L = false;
        for (size_t j = 0; j < h_L.size(); ++j) {
            if (std::isnan(h_L[j]) || std::isinf(h_L[j])) {
                has_nan_inf_L = true;
                std::cerr << "NaN or Inf detected in L at index " << j 
                          << ": " << h_L[j] << std::endl;
                break;
            }
        }
        
        if (has_nan_inf_O || has_nan_inf_L) {
            std::cerr << "✗ Test failed for " << scale_names[i] << " inputs" << std::endl;
            if (has_nan_inf_O) std::cerr << "  - NaN or Inf detected in O" << std::endl;
            if (has_nan_inf_L) std::cerr << "  - NaN or Inf detected in L" << std::endl;
            
            // Clean up before exiting
            CUDA_CHECK(cudaFree(d_Q));
            CUDA_CHECK(cudaFree(d_K));
            CUDA_CHECK(cudaFree(d_V));
            CUDA_CHECK(cudaFree(d_O));
            CUDA_CHECK(cudaFree(d_L));
            CURAND_CHECK(curandDestroyGenerator(generator));
            exit(1);
        } else {
            std::cout << "  ✓ No numerical issues detected for " << scale_names[i] << " inputs" << std::endl;
        }
        
        CURAND_CHECK(curandDestroyGenerator(generator));
    }
    
    // Clean up
    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_O));
    CUDA_CHECK(cudaFree(d_L));
    
    std::cout << "✓ Zero Input Test passed" << std::endl;
}

// Test 3: Large Value Test
void test_large_values() {
    std::cout << "Running Large Value Test..." << std::endl;
    
    // Test with a subset of configurations
    const int batch_size = 1;
    const int num_heads = 1;
    const int seq_len = 32;
    const int head_dim = 64;
    const int BrDim = 16;
    const int BcDim = 16;
    
    // Calculate sizes
    size_t tensor_size = batch_size * num_heads * seq_len * head_dim;
    size_t tensor_bytes = tensor_size * sizeof(half);
    size_t L_size = batch_size * num_heads * seq_len;
    size_t L_bytes = L_size * sizeof(float);
    
    // Allocate device memory
    half *d_Q, *d_K, *d_V, *d_O;
    float *d_L;
    
    CUDA_CHECK(cudaMalloc(&d_Q, tensor_bytes));
    CUDA_CHECK(cudaMalloc(&d_K, tensor_bytes));
    CUDA_CHECK(cudaMalloc(&d_V, tensor_bytes));
    CUDA_CHECK(cudaMalloc(&d_O, tensor_bytes));
    CUDA_CHECK(cudaMalloc(&d_L, L_bytes));
    
    // Test cases with different scales
    // Note: FP16 max value is around 65504
    const float scales[] = {10.0f, 100.0f, 1000.0f, 10000.0f};
    const char* scale_names[] = {"10.0", "100.0", "1000.0", "10000.0"};
    
    for (int i = 0; i < 4; ++i) {
        float scale = scales[i];
        std::cout << "  Testing with scale " << scale_names[i] << std::endl;
        
        // Initialize with scaled random data
        curandGenerator_t generator;
        CURAND_CHECK(curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT));
        CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(generator, 1234ULL));
        
        initialize_random_data(d_Q, tensor_size, scale, generator);
        initialize_random_data(d_K, tensor_size, scale, generator);
        initialize_random_data(d_V, tensor_size, scale, generator);
        
        // Initialize outputs with zeros
        CUDA_CHECK(cudaMemset(d_O, 0, tensor_bytes));
        CUDA_CHECK(cudaMemset(d_L, 0, L_bytes));
        
        // Run the kernel
        flash_attention(d_Q, d_K, d_V, d_O, d_L, BrDim, BcDim, 
                       batch_size, num_heads, seq_len, head_dim);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Copy outputs back to host
        std::vector<half> h_O(tensor_size);
        std::vector<float> h_L(L_size);
        
        CUDA_CHECK(cudaMemcpy(h_O.data(), d_O, tensor_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_L.data(), d_L, L_bytes, cudaMemcpyDeviceToHost));
        
        // Check for NaN or Inf in O
        bool has_nan_inf_O = false;
        for (size_t j = 0; j < h_O.size(); ++j) {
            float val = __half2float(h_O[j]);
            if (std::isnan(val) || std::isinf(val)) {
                has_nan_inf_O = true;
                std::cerr << "NaN or Inf detected in O at index " << j 
                          << ": " << val << std::endl;
                break;
            }
        }
        
        // Check for NaN or Inf in L
        bool has_nan_inf_L = false;
        for (size_t j = 0; j < h_L.size(); ++j) {
            if (std::isnan(h_L[j]) || std::isinf(h_L[j])) {
                has_nan_inf_L = true;
                std::cerr << "NaN or Inf detected in L at index " << j 
                          << ": " << h_L[j] << std::endl;
                break;
            }
        }
        
        if (has_nan_inf_O || has_nan_inf_L) {
            std::cerr << "✗ Test failed for scale " << scale_names[i] << std::endl;
            if (has_nan_inf_O) std::cerr << "  - NaN or Inf detected in O" << std::endl;
            if (has_nan_inf_L) std::cerr << "  - NaN or Inf detected in L" << std::endl;
            
            // Clean up before exiting
            CUDA_CHECK(cudaFree(d_Q));
            CUDA_CHECK(cudaFree(d_K));
            CUDA_CHECK(cudaFree(d_V));
            CUDA_CHECK(cudaFree(d_O));
            CUDA_CHECK(cudaFree(d_L));
            CURAND_CHECK(curandDestroyGenerator(generator));
            exit(1);
        } else {
            std::cout << "  ✓ No numerical issues detected for scale " << scale_names[i] << std::endl;
        }
        
        CURAND_CHECK(curandDestroyGenerator(generator));
    }
    
    // Clean up
    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_O));
    CUDA_CHECK(cudaFree(d_L));
    
    std::cout << "✓ Large Value Test passed" << std::endl;
}

// Test 4: Mixed Scale Test
void test_mixed_scales() {
    std::cout << "Running Mixed Scale Test..." << std::endl;
    
    // Test with a subset of configurations
    const int batch_size = 1;
    const int num_heads = 1;
    const int seq_len = 32;
    const int head_dim = 64;
    const int BrDim = 16;
    const int BcDim = 16;
    
    // Calculate sizes
    size_t tensor_size = batch_size * num_heads * seq_len * head_dim;
    size_t tensor_bytes = tensor_size * sizeof(half);
    size_t L_size = batch_size * num_heads * seq_len;
    size_t L_bytes = L_size * sizeof(float);
    
    // Allocate device memory
    half *d_Q, *d_K, *d_V, *d_O;
    float *d_L;
    
    CUDA_CHECK(cudaMalloc(&d_Q, tensor_bytes));
    CUDA_CHECK(cudaMalloc(&d_K, tensor_bytes));
    CUDA_CHECK(cudaMalloc(&d_V, tensor_bytes));
    CUDA_CHECK(cudaMalloc(&d_O, tensor_bytes));
    CUDA_CHECK(cudaMalloc(&d_L, L_bytes));
    
    // Initialize with mixed scale values
    dim3 block(256);
    dim3 grid((tensor_size + block.x - 1) / block.x);
    set_mixed_scale_values_kernel<<<grid, block>>>(d_Q, d_K, d_V, batch_size, num_heads, seq_len, head_dim);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Initialize outputs with zeros
    CUDA_CHECK(cudaMemset(d_O, 0, tensor_bytes));
    CUDA_CHECK(cudaMemset(d_L, 0, L_bytes));
    
    // Run the kernel
    flash_attention(d_Q, d_K, d_V, d_O, d_L, BrDim, BcDim, 
                   batch_size, num_heads, seq_len, head_dim);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy outputs back to host
    std::vector<half> h_O(tensor_size);
    std::vector<float> h_L(L_size);
    
    CUDA_CHECK(cudaMemcpy(h_O.data(), d_O, tensor_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_L.data(), d_L, L_bytes, cudaMemcpyDeviceToHost));
    
    // Check for NaN or Inf in O
    bool has_nan_inf_O = false;
    for (size_t i = 0; i < h_O.size(); ++i) {
        float val = __half2float(h_O[i]);
        if (std::isnan(val) || std::isinf(val)) {
            has_nan_inf_O = true;
            std::cerr << "NaN or Inf detected in O at index " << i 
                      << ": " << val << std::endl;
            break;
        }
    }
    
    // Check for NaN or Inf in L
    bool has_nan_inf_L = false;
    for (size_t i = 0; i < h_L.size(); ++i) {
        if (std::isnan(h_L[i]) || std::isinf(h_L[i])) {
            has_nan_inf_L = true;
            std::cerr << "NaN or Inf detected in L at index " << i 
                      << ": " << h_L[i] << std::endl;
            break;
        }
    }
    
    // Check for all zeros in O (which might indicate a problem)
    bool all_zeros_O = true;
    for (size_t i = 0; i < h_O.size() && all_zeros_O; ++i) {
        if (__half2float(h_O[i]) != 0.0f) {
            all_zeros_O = false;
        }
    }
    
    if (has_nan_inf_O || has_nan_inf_L || all_zeros_O) {
        std::cerr << "✗ Mixed Scale Test failed" << std::endl;
        if (has_nan_inf_O) std::cerr << "  - NaN or Inf detected in O" << std::endl;
        if (has_nan_inf_L) std::cerr << "  - NaN or Inf detected in L" << std::endl;
        if (all_zeros_O) std::cerr << "  - All zeros detected in O" << std::endl;
        
        // Clean up before exiting
        CUDA_CHECK(cudaFree(d_Q));
        CUDA_CHECK(cudaFree(d_K));
        CUDA_CHECK(cudaFree(d_V));
        CUDA_CHECK(cudaFree(d_O));
        CUDA_CHECK(cudaFree(d_L));
        exit(1);
    } else {
        std::cout << "  ✓ No numerical issues detected with mixed scale inputs" << std::endl;
    }
    
    // Clean up
    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_O));
    CUDA_CHECK(cudaFree(d_L));
    
    std::cout << "✓ Mixed Scale Test passed" << std::endl;
}

// Test 5: Block Size Variation Test
void test_block_size_variations() {
    std::cout << "Running Block Size Variation Test..." << std::endl;
    
    // Test with fixed configuration but varying block sizes
    const int batch_size = 1;
    const int num_heads = 1;
    const int seq_len = 128;
    const int head_dim = 64;
    
    // Different block sizes to test
    const int block_dims[][2] = {
        {8, 8},     // Small blocks
        {16, 16},   // Medium blocks
        {32, 32},   // Large blocks
        {64, 64}    // Very large blocks - might exceed shared memory
    };
    
    // Calculate sizes
    size_t tensor_size = batch_size * num_heads * seq_len * head_dim;
    size_t tensor_bytes = tensor_size * sizeof(half);
    size_t L_size = batch_size * num_heads * seq_len;
    size_t L_bytes = L_size * sizeof(float);
    
    // Allocate device memory
    half *d_Q, *d_K, *d_V, *d_O;
    float *d_L;
    
    CUDA_CHECK(cudaMalloc(&d_Q, tensor_bytes));
    CUDA_CHECK(cudaMalloc(&d_K, tensor_bytes));
    CUDA_CHECK(cudaMalloc(&d_V, tensor_bytes));
    CUDA_CHECK(cudaMalloc(&d_O, tensor_bytes));
    CUDA_CHECK(cudaMalloc(&d_L, L_bytes));
    
    // Initialize with random data
    curandGenerator_t generator;
    CURAND_CHECK(curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(generator, 1234ULL));
    
    initialize_random_data(d_Q, tensor_size, 1.0f, generator);
    initialize_random_data(d_K, tensor_size, 1.0f, generator);
    initialize_random_data(d_V, tensor_size, 1.0f, generator);
    
    // Reference output (using 16x16 blocks)
    half *d_O_ref;
    float *d_L_ref;
    CUDA_CHECK(cudaMalloc(&d_O_ref, tensor_bytes));
    CUDA_CHECK(cudaMalloc(&d_L_ref, L_bytes));
    
    CUDA_CHECK(cudaMemset(d_O_ref, 0, tensor_bytes));
    CUDA_CHECK(cudaMemset(d_L_ref, 0, L_bytes));
    
    flash_attention(d_Q, d_K, d_V, d_O_ref, d_L_ref, 16, 16, 
                   batch_size, num_heads, seq_len, head_dim);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy reference outputs to host
    std::vector<half> h_O_ref(tensor_size);
    std::vector<float> h_L_ref(L_size);
    
    CUDA_CHECK(cudaMemcpy(h_O_ref.data(), d_O_ref, tensor_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_L_ref.data(), d_L_ref, L_bytes, cudaMemcpyDeviceToHost));
    
    // Get device properties to check shared memory limits
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    size_t max_shared_mem = prop.sharedMemPerBlock;
    
    // Test with different block sizes
    for (auto& block_dim : block_dims) {
        int BrDim = block_dim[0];
        int BcDim = block_dim[1];
        
        // Skip if block dimensions are too large for sequence length
        if (BrDim > seq_len || BcDim > seq_len) continue;
        
        // Calculate required shared memory
        size_t shared_mem_size = 0;
        shared_mem_size += BrDim * head_dim * sizeof(half);  // Qs
        shared_mem_size += BcDim * head_dim * sizeof(half);  // Ks
        shared_mem_size += BrDim * BcDim * sizeof(float);    // Ss
        shared_mem_size += BcDim * head_dim * sizeof(half);  // Vs
        shared_mem_size += BrDim * head_dim * sizeof(half);  // Os
        shared_mem_size += BrDim * sizeof(float);            // ls
        shared_mem_size += BrDim * sizeof(float);            // ms
        shared_mem_size += BrDim * sizeof(float);            // expMaxDelta
        
        // Ensure alignment
        shared_mem_size = ((shared_mem_size + 15) / 16) * 16;
        
        // Check if shared memory size is within limits
        if (shared_mem_size > max_shared_mem) {
            std::cout << "  Skipping block size: BrDim=" << BrDim << ", BcDim=" << BcDim 
                      << " - Required shared memory (" << shared_mem_size << " bytes) exceeds device limit (" 
                      << max_shared_mem << " bytes)" << std::endl;
            continue;
        }
        
        std::cout << "  Testing block size: BrDim=" << BrDim << ", BcDim=" << BcDim << std::endl;
        
        // Initialize outputs with zeros
        CUDA_CHECK(cudaMemset(d_O, 0, tensor_bytes));
        CUDA_CHECK(cudaMemset(d_L, 0, L_bytes));
        
        // Run the kernel with this block size
        flash_attention(d_Q, d_K, d_V, d_O, d_L, BrDim, BcDim, 
                       batch_size, num_heads, seq_len, head_dim);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Copy outputs back to host
        std::vector<half> h_O(tensor_size);
        std::vector<float> h_L(L_size);
        
        CUDA_CHECK(cudaMemcpy(h_O.data(), d_O, tensor_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_L.data(), d_L, L_bytes, cudaMemcpyDeviceToHost));
        
        // Check for NaN or Inf in O
        bool has_nan_inf_O = false;
        for (size_t i = 0; i < h_O.size(); ++i) {
            float val = __half2float(h_O[i]);
            if (std::isnan(val) || std::isinf(val)) {
                has_nan_inf_O = true;
                std::cerr << "NaN or Inf detected in O at index " << i 
                          << ": " << val << std::endl;
                break;
            }
        }
        
        // Check for NaN or Inf in L
        bool has_nan_inf_L = false;
        for (size_t i = 0; i < h_L.size(); ++i) {
            if (std::isnan(h_L[i]) || std::isinf(h_L[i])) {
                has_nan_inf_L = true;
                std::cerr << "NaN or Inf detected in L at index " << i 
                          << ": " << h_L[i] << std::endl;
                break;
            }
        }
        
        // Check for consistency with reference output
        bool consistent_O = true;
        float max_diff_O = 0.0f;
        for (size_t i = 0; i < h_O.size() && consistent_O; ++i) {
            float val = __half2float(h_O[i]);
            float ref_val = __half2float(h_O_ref[i]);
            float diff = std::abs(val - ref_val);
            max_diff_O = std::max(max_diff_O, diff);
            
            // Allow for small numerical differences
            if (diff > 0.01f) {
                consistent_O = false;
                std::cerr << "Inconsistent O value at index " << i 
                          << ": current=" << val << ", reference=" << ref_val 
                          << ", diff=" << diff << std::endl;
            }
        }
        
        bool consistent_L = true;
        float max_diff_L = 0.0f;
        for (size_t i = 0; i < h_L.size() && consistent_L; ++i) {
            float diff = std::abs(h_L[i] - h_L_ref[i]);
            max_diff_L = std::max(max_diff_L, diff);
            
            // Allow for small numerical differences
            if (diff > 0.01f) {
                consistent_L = false;
                std::cerr << "Inconsistent L value at index " << i 
                          << ": current=" << h_L[i] << ", reference=" << h_L_ref[i] 
                          << ", diff=" << diff << std::endl;
            }
        }
        
        if (has_nan_inf_O || has_nan_inf_L || !consistent_O || !consistent_L) {
            std::cerr << "✗ Test failed for block size: BrDim=" << BrDim << ", BcDim=" << BcDim << std::endl;
            if (has_nan_inf_O) std::cerr << "  - NaN or Inf detected in O" << std::endl;
            if (has_nan_inf_L) std::cerr << "  - NaN or Inf detected in L" << std::endl;
            if (!consistent_O) std::cerr << "  - Inconsistent O values (max diff: " << max_diff_O << ")" << std::endl;
            if (!consistent_L) std::cerr << "  - Inconsistent L values (max diff: " << max_diff_L << ")" << std::endl;
            
            // Clean up before exiting
            CUDA_CHECK(cudaFree(d_Q));
            CUDA_CHECK(cudaFree(d_K));
            CUDA_CHECK(cudaFree(d_V));
            CUDA_CHECK(cudaFree(d_O));
            CUDA_CHECK(cudaFree(d_L));
            CUDA_CHECK(cudaFree(d_O_ref));
            CUDA_CHECK(cudaFree(d_L_ref));
            CURAND_CHECK(curandDestroyGenerator(generator));
            exit(1);
        } else {
            std::cout << "  ✓ No numerical issues detected for this block size (max diff O: " 
                      << max_diff_O << ", max diff L: " << max_diff_L << ")" << std::endl;
        }
    }
    
    // Clean up
    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_O));
    CUDA_CHECK(cudaFree(d_L));
    CUDA_CHECK(cudaFree(d_O_ref));
    CUDA_CHECK(cudaFree(d_L_ref));
    CURAND_CHECK(curandDestroyGenerator(generator));
    
    std::cout << "✓ Block Size Variation Test passed" << std::endl;
}

// Test 6: Extreme Attention Patterns Test
void test_extreme_attention_patterns() {
    std::cout << "Running Extreme Attention Patterns Test..." << std::endl;
    
    // Test with a subset of configurations
    const int batch_size = 1;
    const int num_heads = 1;
    const int seq_len = 64;
    const int head_dim = 64;
    const int BrDim = 16;
    const int BcDim = 16;
    
    // Calculate sizes
    size_t tensor_size = batch_size * num_heads * seq_len * head_dim;
    size_t tensor_bytes = tensor_size * sizeof(half);
    size_t L_size = batch_size * num_heads * seq_len;
    size_t L_bytes = L_size * sizeof(float);
    
    // Allocate device memory
    half *d_Q, *d_K, *d_V, *d_O;
    float *d_L;
    
    CUDA_CHECK(cudaMalloc(&d_Q, tensor_bytes));
    CUDA_CHECK(cudaMalloc(&d_K, tensor_bytes));
    CUDA_CHECK(cudaMalloc(&d_V, tensor_bytes));
    CUDA_CHECK(cudaMalloc(&d_O, tensor_bytes));
    CUDA_CHECK(cudaMalloc(&d_L, L_bytes));
    
    // Test different extreme attention patterns
    const char* pattern_names[] = {
        "Single token attention",
        "Uniform attention",
        "Alternating strong/weak attention",
        "Exponentially decreasing attention"
    };
    
    for (int pattern = 0; pattern < 4; ++pattern) {
        std::cout << "  Testing with " << pattern_names[pattern] << " pattern" << std::endl;
        
        // Set the extreme attention pattern
        dim3 block(256);
        dim3 grid((tensor_size + block.x - 1) / block.x);
        set_extreme_attention_patterns_kernel<<<grid, block>>>(d_Q, d_K, d_V, batch_size, num_heads, seq_len, head_dim, pattern);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Initialize outputs with zeros
        CUDA_CHECK(cudaMemset(d_O, 0, tensor_bytes));
        CUDA_CHECK(cudaMemset(d_L, 0, L_bytes));
        
        // Run the kernel
        flash_attention(d_Q, d_K, d_V, d_O, d_L, BrDim, BcDim, 
                       batch_size, num_heads, seq_len, head_dim);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Copy outputs back to host
        std::vector<half> h_O(tensor_size);
        std::vector<float> h_L(L_size);
        
        CUDA_CHECK(cudaMemcpy(h_O.data(), d_O, tensor_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_L.data(), d_L, L_bytes, cudaMemcpyDeviceToHost));
        
        // Check for NaN or Inf in O
        bool has_nan_inf_O = false;
        for (size_t i = 0; i < h_O.size(); ++i) {
            float val = __half2float(h_O[i]);
            if (std::isnan(val) || std::isinf(val)) {
                has_nan_inf_O = true;
                std::cerr << "NaN or Inf detected in O at index " << i 
                          << ": " << val << std::endl;
                break;
            }
        }
        
        // Check for NaN or Inf in L
        bool has_nan_inf_L = false;
        for (size_t i = 0; i < h_L.size(); ++i) {
            if (std::isnan(h_L[i]) || std::isinf(h_L[i])) {
                has_nan_inf_L = true;
                std::cerr << "NaN or Inf detected in L at index " << i 
                          << ": " << h_L[i] << std::endl;
                break;
            }
        }
        
        // Check for all zeros in O (which might indicate a problem)
        bool all_zeros_O = true;
        for (size_t i = 0; i < h_O.size() && all_zeros_O; ++i) {
            if (__half2float(h_O[i]) != 0.0f) {
                all_zeros_O = false;
            }
        }
        
        if (has_nan_inf_O || has_nan_inf_L || all_zeros_O) {
            std::cerr << "✗ Test failed for " << pattern_names[pattern] << " pattern" << std::endl;
            if (has_nan_inf_O) std::cerr << "  - NaN or Inf detected in O" << std::endl;
            if (has_nan_inf_L) std::cerr << "  - NaN or Inf detected in L" << std::endl;
            if (all_zeros_O) std::cerr << "  - All zeros detected in O" << std::endl;
            
            // Clean up before exiting
            CUDA_CHECK(cudaFree(d_Q));
            CUDA_CHECK(cudaFree(d_K));
            CUDA_CHECK(cudaFree(d_V));
            CUDA_CHECK(cudaFree(d_O));
            CUDA_CHECK(cudaFree(d_L));
            exit(1);
        } else {
            std::cout << "  ✓ No numerical issues detected for " << pattern_names[pattern] << " pattern" << std::endl;
        }
    }
    
    // Clean up
    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_O));
    CUDA_CHECK(cudaFree(d_L));
    
    std::cout << "✓ Extreme Attention Patterns Test passed" << std::endl;
}

int main() {
    std::cout << "Starting FlashAttention 2 Numerical Stability Tests\n" << std::endl;
    
    // Run tests
    test_nan_inf_detection();
    test_zero_input();
    test_large_values();
    test_mixed_scales();
    test_block_size_variations();
    test_extreme_attention_patterns();
    
    std::cout << "\nAll numerical stability tests completed successfully!" << std::endl;
    return 0;
} 