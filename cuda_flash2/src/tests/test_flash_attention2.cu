#include <cuda_runtime.h>
#include <curand.h>
#include <cuda_fp16.h>
#include <iostream>
#include <cassert>
#include <vector>
#include <cstring>  // for memcmp
#include "../flash_attention_2.cu"
#include <iomanip>

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

void test_kernel_compilation() {
    std::cout << "Running Kernel Compilation Test..." << std::endl;
    
    // Test parameters (very small input)
    const int batch_size = 1;
    const int num_heads = 1;
    const int seq_len = 32;
    const int head_dim = 32;
    const int BrDim = 16;
    const int BcDim = 16;
    
    // Calculate sizes
    size_t tensor_size = batch_size * num_heads * seq_len * head_dim * sizeof(half);
    size_t L_size = batch_size * num_heads * seq_len * sizeof(float);
    
    // Allocate device memory
    half *d_Q, *d_K, *d_V, *d_O;
    float *d_L;
    
    CUDA_CHECK(cudaMalloc(&d_Q, tensor_size));
    CUDA_CHECK(cudaMalloc(&d_K, tensor_size));
    CUDA_CHECK(cudaMalloc(&d_V, tensor_size));
    CUDA_CHECK(cudaMalloc(&d_O, tensor_size));
    CUDA_CHECK(cudaMalloc(&d_L, L_size));
    
    // Initialize with zeros
    CUDA_CHECK(cudaMemset(d_Q, 0, tensor_size));
    CUDA_CHECK(cudaMemset(d_K, 0, tensor_size));
    CUDA_CHECK(cudaMemset(d_V, 0, tensor_size));
    CUDA_CHECK(cudaMemset(d_O, 0, tensor_size));
    CUDA_CHECK(cudaMemset(d_L, 0, L_size));
    
    // Try to launch the kernel
    try {
        flash_attention(d_Q, d_K, d_V, d_O, d_L, BrDim, BcDim, 
                       batch_size, num_heads, seq_len, head_dim);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::cout << "✓ Kernel compilation test passed: kernel launched successfully" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "✗ Kernel compilation test failed: " << e.what() << std::endl;
        exit(1);
    }
    
    // Clean up
    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_O));
    CUDA_CHECK(cudaFree(d_L));
}

void test_memory_initialization() {
    std::cout << "Running Memory Allocation and Initialization Test..." << std::endl;
    
    // Test parameters
    const int batch_size = 1;
    const int num_heads = 1;
    const int seq_len = 32;
    const int head_dim = 32;
    const int BrDim = 16;
    const int BcDim = 16;
    
    // Calculate sizes
    size_t tensor_size = batch_size * num_heads * seq_len * head_dim * sizeof(half);
    size_t L_size = batch_size * num_heads * seq_len * sizeof(float);
    
    // Allocate device memory
    half *d_Q, *d_K, *d_V, *d_O;
    float *d_L;
    
    CUDA_CHECK(cudaMalloc(&d_Q, tensor_size));
    CUDA_CHECK(cudaMalloc(&d_K, tensor_size));
    CUDA_CHECK(cudaMalloc(&d_V, tensor_size));
    CUDA_CHECK(cudaMalloc(&d_O, tensor_size));
    CUDA_CHECK(cudaMalloc(&d_L, L_size));
    
    // Initialize with a known pattern (all ones for Q, K, V)
    std::vector<half> h_ones(batch_size * num_heads * seq_len * head_dim);
    for (size_t i = 0; i < h_ones.size(); ++i) {
        h_ones[i] = __float2half(1.0f);
    }
    
    CUDA_CHECK(cudaMemcpy(d_Q, h_ones.data(), tensor_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_ones.data(), tensor_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_ones.data(), tensor_size, cudaMemcpyHostToDevice));
    
    // Initialize outputs with zeros
    CUDA_CHECK(cudaMemset(d_O, 0, tensor_size));
    CUDA_CHECK(cudaMemset(d_L, 0, L_size));
    
    // Verify initialization
    std::vector<half> h_Q(batch_size * num_heads * seq_len * head_dim);
    std::vector<half> h_O(batch_size * num_heads * seq_len * head_dim);
    std::vector<float> h_L(batch_size * num_heads * seq_len);
    
    CUDA_CHECK(cudaMemcpy(h_Q.data(), d_Q, tensor_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_O.data(), d_O, tensor_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_L.data(), d_L, L_size, cudaMemcpyDeviceToHost));
    
    // Check Q initialization (should be all ones)
    bool q_init_correct = true;
    for (size_t i = 0; i < h_Q.size() && q_init_correct; ++i) {
        if (__half2float(h_Q[i]) != 1.0f) {
            q_init_correct = false;
            std::cerr << "Q initialization failed at index " << i 
                      << ": expected 1.0, got " << __half2float(h_Q[i]) << std::endl;
        }
    }
    
    // Check O initialization (should be all zeros)
    bool o_init_correct = true;
    for (size_t i = 0; i < h_O.size() && o_init_correct; ++i) {
        if (__half2float(h_O[i]) != 0.0f) {
            o_init_correct = false;
            std::cerr << "O initialization failed at index " << i 
                      << ": expected 0.0, got " << __half2float(h_O[i]) << std::endl;
        }
    }
    
    // Check L initialization (should be all zeros)
    bool l_init_correct = true;
    for (size_t i = 0; i < h_L.size() && l_init_correct; ++i) {
        if (h_L[i] != 0.0f) {
            l_init_correct = false;
            std::cerr << "L initialization failed at index " << i 
                      << ": expected 0.0, got " << h_L[i] << std::endl;
        }
    }
    
    if (q_init_correct && o_init_correct && l_init_correct) {
        std::cout << "✓ Memory initialization test passed: all tensors initialized correctly" << std::endl;
    } else {
        std::cerr << "✗ Memory initialization test failed" << std::endl;
        exit(1);
    }
    
    // Clean up
    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_O));
    CUDA_CHECK(cudaFree(d_L));
}

void test_output_consistency() {
    std::cout << "Running Output Consistency Test..." << std::endl;
    
    // Test parameters
    const int batch_size = 1;
    const int num_heads = 1;
    const int seq_len = 32;
    const int head_dim = 32;
    const int BrDim = 16;
    const int BcDim = 16;
    const int num_runs = 3;  // Number of times to run the kernel
    
    // Calculate sizes
    size_t tensor_size = batch_size * num_heads * seq_len * head_dim * sizeof(half);
    size_t L_size = batch_size * num_heads * seq_len * sizeof(float);
    
    // Allocate device memory
    half *d_Q, *d_K, *d_V, *d_O;
    float *d_L;
    
    CUDA_CHECK(cudaMalloc(&d_Q, tensor_size));
    CUDA_CHECK(cudaMalloc(&d_K, tensor_size));
    CUDA_CHECK(cudaMalloc(&d_V, tensor_size));
    CUDA_CHECK(cudaMalloc(&d_O, tensor_size));
    CUDA_CHECK(cudaMalloc(&d_L, L_size));
    
    // Initialize inputs with a constant pattern
    std::vector<half> h_ones(batch_size * num_heads * seq_len * head_dim);
    for (size_t i = 0; i < h_ones.size(); ++i) {
        h_ones[i] = __float2half(1.0f);
    }
    
    CUDA_CHECK(cudaMemcpy(d_Q, h_ones.data(), tensor_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_ones.data(), tensor_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_ones.data(), tensor_size, cudaMemcpyHostToDevice));
    
    // Vectors to store outputs from different runs
    std::vector<std::vector<half>> O_outputs(num_runs);
    std::vector<std::vector<float>> L_outputs(num_runs);
    
    for (int i = 0; i < num_runs; ++i) {
        // Initialize outputs with zeros
        CUDA_CHECK(cudaMemset(d_O, 0, tensor_size));
        CUDA_CHECK(cudaMemset(d_L, 0, L_size));
        
        // Run the kernel
        flash_attention(d_Q, d_K, d_V, d_O, d_L, BrDim, BcDim, 
                       batch_size, num_heads, seq_len, head_dim);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Copy outputs back to host
        O_outputs[i].resize(batch_size * num_heads * seq_len * head_dim);
        L_outputs[i].resize(batch_size * num_heads * seq_len);
        
        CUDA_CHECK(cudaMemcpy(O_outputs[i].data(), d_O, tensor_size, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(L_outputs[i].data(), d_L, L_size, cudaMemcpyDeviceToHost));
    }
    
    // Compare outputs from different runs
    bool outputs_consistent = true;
    for (int i = 1; i < num_runs && outputs_consistent; ++i) {
        // Compare O outputs
        for (size_t j = 0; j < O_outputs[i].size() && outputs_consistent; ++j) {
            if (__half2float(O_outputs[i][j]) != __half2float(O_outputs[0][j])) {
                outputs_consistent = false;
                std::cerr << "O output inconsistency detected at run " << i << ", index " << j 
                          << ": first run = " << __half2float(O_outputs[0][j])
                          << ", current run = " << __half2float(O_outputs[i][j]) << std::endl;
            }
        }
        
        // Compare L outputs
        for (size_t j = 0; j < L_outputs[i].size() && outputs_consistent; ++j) {
            if (L_outputs[i][j] != L_outputs[0][j]) {
                outputs_consistent = false;
                std::cerr << "L output inconsistency detected at run " << i << ", index " << j 
                          << ": first run = " << L_outputs[0][j]
                          << ", current run = " << L_outputs[i][j] << std::endl;
            }
        }
    }
    
    if (outputs_consistent) {
        std::cout << "✓ Output consistency test passed: all runs produced identical results" << std::endl;
    } else {
        std::cerr << "✗ Output consistency test failed: inconsistent results between runs" << std::endl;
        exit(1);
    }
    
    // Clean up
    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_O));
    CUDA_CHECK(cudaFree(d_L));
}

int main() {
    std::cout << "Starting FlashAttention 2 Tests\n" << std::endl;
    
    // Run tests
    test_kernel_compilation();
    test_memory_initialization();
    test_output_consistency();
    
    std::cout << "\nAll tests completed successfully!" << std::endl;
    return 0;
} 