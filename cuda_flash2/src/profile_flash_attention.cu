#include <cuda_runtime.h>
#include <curand.h>
#include <cuda_fp16.h>
#include <iostream>
#include "flash_attention_2.cu"

// Helper kernel to convert float data to half
__global__ void convert_to_half_kernel(half* data, float* float_data, float scale, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = __float2half(float_data[idx] * scale);
    }
}

int main() {
    // Configuration for profiling
    const int batch_size = 1;
    const int num_heads = 8;
    const int seq_len = 512;
    const int head_dim = 64;
    const int BrDim = 32;
    const int BcDim = 32;
    
    // Calculate sizes
    size_t tensor_size = batch_size * num_heads * seq_len * head_dim;
    size_t tensor_bytes = tensor_size * sizeof(half);
    size_t L_size = batch_size * num_heads * seq_len;
    size_t L_bytes = L_size * sizeof(float);
    
    // Allocate device memory
    half *d_Q, *d_K, *d_V, *d_O;
    float *d_L;
    
    cudaMalloc(&d_Q, tensor_bytes);
    cudaMalloc(&d_K, tensor_bytes);
    cudaMalloc(&d_V, tensor_bytes);
    cudaMalloc(&d_O, tensor_bytes);
    cudaMalloc(&d_L, L_bytes);
    
    // Initialize with random data
    curandGenerator_t generator;
    curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(generator, 1234ULL);
    
    float* float_data;
    cudaMalloc(&float_data, tensor_size * sizeof(float));
    
    // Generate and convert Q
    curandGenerateUniform(generator, float_data, tensor_size);
    dim3 block(256);
    dim3 grid((tensor_size + block.x - 1) / block.x);
    convert_to_half_kernel<<<grid, block>>>(d_Q, float_data, 1.0f, tensor_size);
    
    // Generate and convert K
    curandGenerateUniform(generator, float_data, tensor_size);
    convert_to_half_kernel<<<grid, block>>>(d_K, float_data, 1.0f, tensor_size);
    
    // Generate and convert V
    curandGenerateUniform(generator, float_data, tensor_size);
    convert_to_half_kernel<<<grid, block>>>(d_V, float_data, 1.0f, tensor_size);
    
    cudaDeviceSynchronize();
    
    // Warmup run
    flash_attention(d_Q, d_K, d_V, d_O, d_L, BrDim, BcDim, 
                   batch_size, num_heads, seq_len, head_dim);
    
    // Profile run
    cudaDeviceSynchronize();
    std::cout << "Running FlashAttention 2 kernel for profiling..." << std::endl;
    flash_attention(d_Q, d_K, d_V, d_O, d_L, BrDim, BcDim, 
                   batch_size, num_heads, seq_len, head_dim);
    cudaDeviceSynchronize();
    std::cout << "Profiling run completed." << std::endl;
    
    // Clean up
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O);
    cudaFree(d_L);
    cudaFree(float_data);
    curandDestroyGenerator(generator);
    
    return 0;
} 