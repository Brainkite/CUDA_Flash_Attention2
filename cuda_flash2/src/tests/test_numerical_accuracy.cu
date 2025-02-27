#include <cuda_runtime.h>
#include <curand.h>
#include <cuda_fp16.h>
#include <iostream>
#include <cassert>
#include <vector>
#include <cstring>
#include <cmath>
#include <iomanip>
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

// Kernel to set specific edge case patterns
__global__ void set_edge_case_pattern_kernel(
    half* Q, half* K, half* V,
    int batch_size, int num_heads, int seq_len, int head_dim,
    int pattern_type
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * num_heads * seq_len * head_dim) return;
    
    // Calculate position
    int b = idx / (num_heads * seq_len * head_dim);
    int h = (idx % (num_heads * seq_len * head_dim)) / (seq_len * head_dim);
    int s = (idx % (seq_len * head_dim)) / head_dim;
    int d = idx % head_dim;
    
    // Base offset for this token
    int offset = b * num_heads * seq_len * head_dim + h * seq_len * head_dim + s * head_dim + d;
    
    switch (pattern_type) {
        case 0: // Extremely skewed attention (first token gets all attention)
            if (s == 0) {
                Q[offset] = __float2half(10.0f);
                K[offset] = __float2half(10.0f);
            } else {
                Q[offset] = __float2half(0.01f);
                K[offset] = __float2half(0.01f);
            }
            V[offset] = __float2half(1.0f);
            break;
            
        case 1: // Very small values
            Q[offset] = __float2half(1e-4f);
            K[offset] = __float2half(1e-4f);
            V[offset] = __float2half(1.0f);
            break;
            
        case 2: // Very large values (but within FP16 range)
            Q[offset] = __float2half(10.0f);
            K[offset] = __float2half(10.0f);
            V[offset] = __float2half(60.0f);  // Will create large outputs
            break;
            
        case 3: // Attention masking pattern (upper triangular)
            if (s <= d % seq_len) {
                Q[offset] = __float2half(1.0f);
                K[offset] = __float2half(1.0f);
            } else {
                Q[offset] = __float2half(-50.0f);  // Effectively masked
                K[offset] = __float2half(-50.0f);
            }
            V[offset] = __float2half(1.0f);
            break;
            
        case 4: // Alternating extreme values
            if ((s + d) % 2 == 0) {
                Q[offset] = __float2half(10.0f);
                K[offset] = __float2half(0.01f);
            } else {
                Q[offset] = __float2half(0.01f);
                K[offset] = __float2half(10.0f);
            }
            V[offset] = __float2half(1.0f);
            break;
    }
}

// Naive reference implementation of attention in CUDA
// This is a straightforward implementation without the optimizations of FlashAttention
__global__ void naive_attention_kernel(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ O,
    float* __restrict__ L,
    int batch_size, int num_heads, int seq_len, int head_dim
) {
    // Each thread handles one position in the sequence for one head in one batch
    int b = blockIdx.z;
    int h = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (b >= batch_size || h >= num_heads || i >= seq_len) return;
    
    // Calculate base indices
    int batch_head_offset = b * num_heads * seq_len * head_dim + h * seq_len * head_dim;
    int seq_offset = i * head_dim;
    int qi_offset = batch_head_offset + seq_offset;
    
    // Compute attention scores for this position
    float scores[4096];  // Assuming max seq_len is 4096, could be made dynamic
    float max_score = -INFINITY;
    
    // Compute QK^T / sqrt(d)
    float scale = 1.0f / sqrtf(head_dim);
    for (int j = 0; j < seq_len; ++j) {
        float score = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            float q_val = __half2float(Q[qi_offset + d]);
            float k_val = __half2float(K[batch_head_offset + j * head_dim + d]);
            score += q_val * k_val;
        }
        scores[j] = score * scale;
        max_score = fmaxf(max_score, scores[j]);
    }
    
    // Compute softmax
    float exp_sum = 0.0f;
    for (int j = 0; j < seq_len; ++j) {
        scores[j] = expf(scores[j] - max_score);
        exp_sum += scores[j];
    }
    
    // Normalize
    for (int j = 0; j < seq_len; ++j) {
        scores[j] /= exp_sum;
    }
    
    // Compute weighted sum of values
    for (int d = 0; d < head_dim; ++d) {
        float sum = 0.0f;
        for (int j = 0; j < seq_len; ++j) {
            sum += scores[j] * __half2float(V[batch_head_offset + j * head_dim + d]);
        }
        O[qi_offset + d] = __float2half(sum);
    }
    
    // Store the log-sum-exp for this position
    L[b * num_heads * seq_len + h * seq_len + i] = max_score + logf(exp_sum);
}

// Host function for naive attention
void naive_attention(
    half* Q, half* K, half* V, half* O, float* L,
    int batch_size, int num_heads, int seq_len, int head_dim
) {
    // Initialize output arrays
    CUDA_CHECK(cudaMemset(O, 0, batch_size * num_heads * seq_len * head_dim * sizeof(half)));
    CUDA_CHECK(cudaMemset(L, 0, batch_size * num_heads * seq_len * sizeof(float)));
    
    // Launch kernel
    dim3 block(256);
    dim3 grid((seq_len + block.x - 1) / block.x, num_heads, batch_size);
    
    naive_attention_kernel<<<grid, block>>>(Q, K, V, O, L, batch_size, num_heads, seq_len, head_dim);
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch error: " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
    
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Function to compare two tensors and report differences
template <typename T>
bool compare_tensors(
    const std::vector<T>& a, 
    const std::vector<T>& b, 
    float tolerance, 
    const char* name,
    bool print_details = true
) {
    if (a.size() != b.size()) {
        std::cerr << "Size mismatch for " << name << ": " << a.size() << " vs " << b.size() << std::endl;
        return false;
    }
    
    float max_diff = 0.0f;
    float avg_diff = 0.0f;
    int num_diffs = 0;
    int max_diff_idx = -1;
    
    for (size_t i = 0; i < a.size(); ++i) {
        float val_a, val_b;
        if constexpr (std::is_same<T, half>::value) {
            val_a = __half2float(a[i]);
            val_b = __half2float(b[i]);
        } else {
            val_a = a[i];
            val_b = b[i];
        }
        
        float diff = std::abs(val_a - val_b);
        avg_diff += diff;
        
        if (diff > max_diff) {
            max_diff = diff;
            max_diff_idx = i;
        }
        
        if (diff > tolerance) {
            num_diffs++;
            if (print_details && num_diffs <= 10) {
                std::cerr << "Difference at index " << i << ": " 
                          << val_a << " vs " << val_b 
                          << " (diff: " << diff << ")" << std::endl;
            }
        }
    }
    
    avg_diff /= a.size();
    
    if (print_details) {
        std::cout << "  " << name << " comparison:" << std::endl;
        std::cout << "    Max difference: " << max_diff << " at index " << max_diff_idx << std::endl;
        std::cout << "    Average difference: " << avg_diff << std::endl;
        std::cout << "    Number of differences > " << tolerance << ": " << num_diffs 
                  << " (" << (100.0f * num_diffs / a.size()) << "%)" << std::endl;
    }
    
    return max_diff <= tolerance;
}

// Test edge cases specifically
void test_edge_cases() {
    std::cout << "Running Edge Case Tests..." << std::endl;
    
    // Fixed configuration for edge cases
    const int batch_size = 1;
    const int num_heads = 1;
    const int seq_len = 64;
    const int head_dim = 64;
    const int BrDim = 16;
    const int BcDim = 16;
    
    // Tolerance for FP16 comparisons
    const float tolerance = 1e-2f;  // Relatively high tolerance for FP16
    
    // Edge case patterns to test
    const char* pattern_names[] = {
        "Extremely skewed attention",
        "Very small values",
        "Very large values",
        "Attention masking pattern",
        "Alternating extreme values"
    };
    
    bool all_passed = true;
    
    for (int pattern = 0; pattern < 5; ++pattern) {
        std::cout << "  Testing edge case: " << pattern_names[pattern] << std::endl;
        
        // Calculate sizes
        size_t tensor_size = batch_size * num_heads * seq_len * head_dim;
        size_t tensor_bytes = tensor_size * sizeof(half);
        size_t L_size = batch_size * num_heads * seq_len;
        size_t L_bytes = L_size * sizeof(float);
        
        // Allocate device memory
        half *d_Q, *d_K, *d_V, *d_O_flash, *d_O_naive;
        float *d_L_flash, *d_L_naive;
        
        CUDA_CHECK(cudaMalloc(&d_Q, tensor_bytes));
        CUDA_CHECK(cudaMalloc(&d_K, tensor_bytes));
        CUDA_CHECK(cudaMalloc(&d_V, tensor_bytes));
        CUDA_CHECK(cudaMalloc(&d_O_flash, tensor_bytes));
        CUDA_CHECK(cudaMalloc(&d_O_naive, tensor_bytes));
        CUDA_CHECK(cudaMalloc(&d_L_flash, L_bytes));
        CUDA_CHECK(cudaMalloc(&d_L_naive, L_bytes));
        
        // Set edge case pattern
        dim3 block(256);
        dim3 grid((tensor_size + block.x - 1) / block.x);
        set_edge_case_pattern_kernel<<<grid, block>>>(d_Q, d_K, d_V, batch_size, num_heads, seq_len, head_dim, pattern);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Run FlashAttention
        flash_attention(d_Q, d_K, d_V, d_O_flash, d_L_flash, BrDim, BcDim, 
                       batch_size, num_heads, seq_len, head_dim);
        
        // Run naive attention
        naive_attention(d_Q, d_K, d_V, d_O_naive, d_L_naive, 
                       batch_size, num_heads, seq_len, head_dim);
        
        // Copy outputs back to host
        std::vector<half> h_O_flash(tensor_size);
        std::vector<half> h_O_naive(tensor_size);
        std::vector<float> h_L_flash(L_size);
        std::vector<float> h_L_naive(L_size);
        
        CUDA_CHECK(cudaMemcpy(h_O_flash.data(), d_O_flash, tensor_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_O_naive.data(), d_O_naive, tensor_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_L_flash.data(), d_L_flash, L_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_L_naive.data(), d_L_naive, L_bytes, cudaMemcpyDeviceToHost));
        
        // Compare outputs
        bool O_match = compare_tensors(h_O_flash, h_O_naive, tolerance, "Output tensor (O)");
        bool L_match = compare_tensors(h_L_flash, h_L_naive, tolerance, "Scaling factors (L)");
        
        if (O_match && L_match) {
            std::cout << "  ✓ Edge case passed: " << pattern_names[pattern] << std::endl;
        } else {
            std::cout << "  ✗ Edge case failed: " << pattern_names[pattern] << std::endl;
            all_passed = false;
        }
        
        // Clean up
        CUDA_CHECK(cudaFree(d_Q));
        CUDA_CHECK(cudaFree(d_K));
        CUDA_CHECK(cudaFree(d_V));
        CUDA_CHECK(cudaFree(d_O_flash));
        CUDA_CHECK(cudaFree(d_O_naive));
        CUDA_CHECK(cudaFree(d_L_flash));
        CUDA_CHECK(cudaFree(d_L_naive));
    }
    
    if (all_passed) {
        std::cout << "✓ All edge case tests passed within tolerance" << std::endl;
    } else {
        std::cout << "✗ Some edge case tests failed" << std::endl;
    }
}

// Main test function
void test_numerical_accuracy() {
    std::cout << "Running Numerical Accuracy Test..." << std::endl;
    
    // Test configurations
    const int batch_sizes[] = {1, 2};
    const int num_heads[] = {1, 4, 8};
    const int seq_lens[] = {32, 128, 256};
    const int head_dims[] = {32, 64, 128};
    
    // Tolerance for FP16 comparisons
    const float tolerance = 1e-2f;  // Relatively high tolerance for FP16
    
    // Create random number generator
    curandGenerator_t generator;
    CURAND_CHECK(curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(generator, 1234ULL));
    
    bool all_passed = true;
    
    for (int bs : batch_sizes) {
        for (int nh : num_heads) {
            for (int seq_len : seq_lens) {
                for (int head_dim : head_dims) {
                    // Skip very large configurations to avoid excessive memory usage
                    if (bs * nh * seq_len * head_dim > 16 * 1024 * 1024) continue;
                    
                    std::cout << "Testing configuration: bs=" << bs 
                              << ", nh=" << nh 
                              << ", seq_len=" << seq_len 
                              << ", head_dim=" << head_dim << std::endl;
                    
                    // Calculate sizes
                    size_t tensor_size = bs * nh * seq_len * head_dim;
                    size_t tensor_bytes = tensor_size * sizeof(half);
                    size_t L_size = bs * nh * seq_len;
                    size_t L_bytes = L_size * sizeof(float);
                    
                    // Allocate device memory
                    half *d_Q, *d_K, *d_V, *d_O_flash, *d_O_naive;
                    float *d_L_flash, *d_L_naive;
                    
                    CUDA_CHECK(cudaMalloc(&d_Q, tensor_bytes));
                    CUDA_CHECK(cudaMalloc(&d_K, tensor_bytes));
                    CUDA_CHECK(cudaMalloc(&d_V, tensor_bytes));
                    CUDA_CHECK(cudaMalloc(&d_O_flash, tensor_bytes));
                    CUDA_CHECK(cudaMalloc(&d_O_naive, tensor_bytes));
                    CUDA_CHECK(cudaMalloc(&d_L_flash, L_bytes));
                    CUDA_CHECK(cudaMalloc(&d_L_naive, L_bytes));
                    
                    // Initialize with random data
                    initialize_random_data(d_Q, tensor_size, 1.0f, generator);
                    initialize_random_data(d_K, tensor_size, 1.0f, generator);
                    initialize_random_data(d_V, tensor_size, 1.0f, generator);
                    
                    // Run FlashAttention
                    int BrDim = 16;
                    int BcDim = 16;
                    
                    flash_attention(d_Q, d_K, d_V, d_O_flash, d_L_flash, BrDim, BcDim, 
                                   bs, nh, seq_len, head_dim);
                    
                    // Run naive attention
                    naive_attention(d_Q, d_K, d_V, d_O_naive, d_L_naive, 
                                   bs, nh, seq_len, head_dim);
                    
                    // Copy outputs back to host
                    std::vector<half> h_O_flash(tensor_size);
                    std::vector<half> h_O_naive(tensor_size);
                    std::vector<float> h_L_flash(L_size);
                    std::vector<float> h_L_naive(L_size);
                    
                    CUDA_CHECK(cudaMemcpy(h_O_flash.data(), d_O_flash, tensor_bytes, cudaMemcpyDeviceToHost));
                    CUDA_CHECK(cudaMemcpy(h_O_naive.data(), d_O_naive, tensor_bytes, cudaMemcpyDeviceToHost));
                    CUDA_CHECK(cudaMemcpy(h_L_flash.data(), d_L_flash, L_bytes, cudaMemcpyDeviceToHost));
                    CUDA_CHECK(cudaMemcpy(h_L_naive.data(), d_L_naive, L_bytes, cudaMemcpyDeviceToHost));
                    
                    // Compare outputs
                    bool O_match = compare_tensors(h_O_flash, h_O_naive, tolerance, "Output tensor (O)");
                    bool L_match = compare_tensors(h_L_flash, h_L_naive, tolerance, "Scaling factors (L)");
                    
                    if (O_match && L_match) {
                        std::cout << "  ✓ Outputs match within tolerance" << std::endl;
                    } else {
                        std::cout << "  ✗ Outputs differ beyond tolerance" << std::endl;
                        all_passed = false;
                    }
                    
                    // Clean up
                    CUDA_CHECK(cudaFree(d_Q));
                    CUDA_CHECK(cudaFree(d_K));
                    CUDA_CHECK(cudaFree(d_V));
                    CUDA_CHECK(cudaFree(d_O_flash));
                    CUDA_CHECK(cudaFree(d_O_naive));
                    CUDA_CHECK(cudaFree(d_L_flash));
                    CUDA_CHECK(cudaFree(d_L_naive));
                }
            }
        }
    }
    
    CURAND_CHECK(curandDestroyGenerator(generator));
    
    if (all_passed) {
        std::cout << "✓ All numerical accuracy tests passed within tolerance" << std::endl;
    } else {
        std::cout << "✗ Some numerical accuracy tests failed" << std::endl;
    }
}

int main() {
    std::cout << "Starting FlashAttention 2 Numerical Accuracy Tests\n" << std::endl;
    
    test_numerical_accuracy();
    test_edge_cases();
    
    std::cout << "\nAll numerical accuracy tests completed!" << std::endl;
    return 0;
} 