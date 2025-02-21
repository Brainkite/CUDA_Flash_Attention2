"""
Numba CUDA Implementation of FlashAttention 2

This module provides a Numba CUDA implementation of the FlashAttention 2 algorithm.
The implementation focuses on memory efficiency and parallel processing using CUDA.

Key Features:
- Block-sparse attention computation
- Efficient shared memory usage
- Warp-level reductions for performance
- Automatic handling of sequence padding
"""

import math
import numpy as np
import numba.cuda as ncuda
from ..utils import cdiv
import torch

@ncuda.jit
def flash_attention(Q, K, V, O, L, BrDim, BcDim, Bs, Nh, N, dim):
    """
    FlashAttention 2 CUDA kernel implementation.
    
    This function implements the core FlashAttention algorithm, processing input matrices
    Q, K, and V in blocks to compute attention scores and output values efficiently.
    The implementation follows these steps:
    1. Load Q block into shared memory
    2. For each K,V block:
       a. Load K,V block into shared memory
       b. Compute attention scores
       c. Update running statistics (mi, Pi)
       d. Update output accumulator (Oi)
    3. Normalize and write final outputs
    
    Args:
        Q (DeviceArray): Input queries [Bs*Nh*N*dim]
        K (DeviceArray): Input keys [Bs*Nh*N*dim]
        V (DeviceArray): Input values [Bs*Nh*N*dim]
        O (DeviceArray): Output tensor [Bs*Nh*N*dim]
        L (DeviceArray): Output scaling factors [Bs*Nh*N]
        BrDim (int): Block size for rows
        BcDim (int): Block size for columns
        Bs (int): Batch size
        Nh (int): Number of attention heads
        N (int): Sequence length
        dim (int): Head dimension
    
    Shared Memory Layout:
        - Qs: Query block [BrDim x dim]
        - Ks: Key block [BcDim x dim]
        - Ss: Score matrix [BrDim x BcDim]
        - Vs: Value block [BcDim x dim]
        - Os: Output accumulator [BrDim x dim]
        - ls: Row scaling factors [BrDim]
        - ms: Row maxima [BrDim]
    
    Performance Notes:
        - Uses warp-level reductions for efficiency
        - Optimizes shared memory access patterns
        - Handles sequence padding automatically
    """
    BrIdx = ncuda.blockIdx.x
    NhIdx = ncuda.blockIdx.y
    BsIdx = ncuda.blockIdx.z
    TcDim = ncuda.blockDim.x
    TrDim = ncuda.blockDim.y
    TcIdx = ncuda.threadIdx.x
    TrIdx = ncuda.threadIdx.y

    # Compute the offset for global input indexing
    global_offset = BsIdx * Nh * N * dim + NhIdx * N * dim

    ### Assign shared memory
    BrSize = BrDim * dim
    BcSize = BcDim * dim
    BattSize = BrDim * BcDim
    shar = ncuda.shared.array(0, dtype=np.float32)
    Qs = shar[ 0                                      : BrSize                                  ]
    Ks = shar[ BrSize                                 : BrSize + BcSize                         ]
    Ss = shar[ BrSize + BcSize                        : BrSize + BcSize + BattSize              ]
    Vs = shar[ BrSize + BcSize + BattSize             : BrSize + 2*BcSize + BattSize            ]
    Os = shar[ BrSize + 2*BcSize + BattSize           : 2*BrSize + 2*BcSize + BattSize          ]
    ls = shar[ 2*BrSize + 2*BcSize + BattSize         : 2*BrSize + 2*BcSize + BattSize + BrDim  ]
    ms = shar[ 2*BrSize + 2*BcSize + BattSize + BrDim : 2*BrSize + 2*BcSize + BattSize + 2*BrDim]
    expMaxDelta = shar[ 2*BrSize + 2*BcSize + BattSize + 2*BrDim : 2*BrSize + 2*BcSize + BattSize + 3*BrDim]

    ### Initialize ms
    for i in range(TrIdx * TcDim + TcIdx, BrDim, TrDim * TcDim):
        if i < BrDim:
            ms[i] = -1e30

    ### Load Q into shared memory
    for i in range(TrIdx, BrDim, TrDim):
        for j in range(TcIdx, dim, TcDim):
            idx = BrIdx * BrDim + i
            if idx < N:
                Qs[i * dim + j] = Q[global_offset + idx * dim + j]

    ### Loop over column blocks of K and V
    num_Bc = (N + BcDim - 1) // BcDim
    for BcIdx in range(0, num_Bc):
        ### Load K and V into shared memory
        for i in range(TrIdx, BcDim, TrDim):
            for j in range(TcIdx, dim, TcDim):
                idx = BcIdx * BcDim + i
                if idx < N:
                    Ks[i * dim + j] = K[global_offset + idx * dim + j]
                    Vs[i * dim + j] = V[global_offset + idx * dim + j]
        ncuda.syncthreads()

        ### Compute attention scores
        sqrt_dim = math.sqrt(dim)
        for i in range(TrIdx, BrDim, TrDim):
            for j in range(TcIdx, BcDim, TcDim):
                if (BrIdx*BrDim+i < N) and (BcIdx*BcDim+j < N):
                    s = 0.0
                    for k in range(dim):
                        s += Qs[i * dim + k] * Ks[j * dim + k]
                    Ss[i * BcDim + j] = s / sqrt_dim
        ncuda.syncthreads()
        
        ### Compute mi and Pi
        for i in range(TrIdx, BrDim, TrDim):
            if BrIdx*BrDim+i < N:
                # Find thread-wise max
                row_max = -1e30
                for j in range(TcIdx, BcDim, TcDim):
                    if BcIdx*BcDim+j < N:
                        row_max = max(row_max, Ss[i * BcDim + j])

                # Find row_max using warp shuffle
                for offset in [16, 8, 4, 2, 1]:
                    row_max = max(row_max, ncuda.shfl_down_sync(0xffffffff, row_max, offset))
                if TcIdx == 0:
                    row_max = max(ms[i], row_max)
                row_max = ncuda.shfl_sync(0xffffffff, row_max, 0)
                
                # Update Ss by computing exp(Ss[i,j] - row_max)
                for j in range(TcIdx, BcDim, TcDim):
                    if BcIdx*BcDim+j < N:
                        Ss[i * BcDim + j] = math.exp(Ss[i * BcDim + j] - row_max)
                
                # Store expMaxDelta and ms values
                if TcIdx == 0:
                    expMaxDelta[i] = math.exp(ms[i] - row_max)
                    ms[i] = row_max
        ncuda.syncthreads()

        ### Compute li and Oi
        for i in range(TrIdx, BrDim, TrDim):
            if BrIdx * BrDim + i < N:
                # Get row sum
                row_sum = 0.0
                for j in range(TcIdx, BcDim, TcDim):
                    if BcIdx * BcDim + j < N:
                        row_sum += Ss[i * BcDim + j]
                for offset in [16, 8, 4, 2, 1]:
                    row_sum += ncuda.shfl_down_sync(0xffffffff, row_sum, offset)
                
                # Update ls
                if TcIdx == 0:
                    ls[i] = max(ls[i] * expMaxDelta[i] + row_sum, 1e-7)
                
                # Update Os
                for j in range(TcIdx, dim, TcDim):
                    pv = 0.0
                    for k in range(BcDim):
                        if BcIdx * BcDim + k < N:
                            pv += Ss[i * BcDim + k] * Vs[k * dim + j]
                    Os[i * dim + j] = Os[i * dim + j] * expMaxDelta[i] + pv
        ncuda.syncthreads()

    ### Update final Oi and li
    for i in range(TrIdx, BrDim, TrDim):
        if BrIdx * BrDim + i < N:
                for j in range(TcIdx, dim, TcDim):
                    Os[i * dim + j] /= ls[i]
                if TcIdx == 0:
                    ls[i] = ms[i] + math.log(ls[i])
    ncuda.syncthreads()

    ### Write output
    for i in range(TrIdx, BrDim, TrDim):
        idx = BrIdx * BrDim + i
        if idx < N:
            for j in range(TcIdx, dim, TcDim):
                O[global_offset + idx * dim + j] = Os[i * dim + j]
            if TcIdx == 0:
                L[BsIdx * Nh * N + NhIdx * N + idx] = ls[i]

def flash_attention2_launcher(Q, K, V):
    """
    Launch the FlashAttention 2 CUDA kernel with optimal parameters.
    
    This function handles:
    1. Computing optimal block sizes based on shared memory constraints
    2. Setting up thread block dimensions
    3. Preparing input/output arrays
    4. Launching the CUDA kernel
    
    Args:
        Q (torch.Tensor): Query tensor [Bs, Nh, N, dim]
        K (torch.Tensor): Key tensor [Bs, Nh, N, dim]
        V (torch.Tensor): Value tensor [Bs, Nh, N, dim]
    
    Returns:
        tuple: (O, L)
            - O (torch.Tensor): Output tensor [Bs, Nh, N, dim]
            - L (torch.Tensor): Scaling factors [Bs, Nh, N]
    """
    # Get input dimensions
    Bs, Nh, N, dim = Q.shape
    input_dtype = Q.dtype
    
    # Convert inputs to float32 for Numba processing if needed
    if input_dtype != torch.float32:
        Q = Q.float()
        K = K.float()
        V = V.float()
    
    # Calculate BrDim and BcDim based on shared memory constraints
    max_shared_mem_bytes = ncuda.get_current_device().MAX_SHARED_MEMORY_PER_BLOCK
    BcDim = min(math.ceil(max_shared_mem_bytes / (4 * 4 * dim)), N)
    BrDim = min(dim, BcDim)
    shared_mem_size = (2 * (BrDim * dim) + 2 * (BcDim * dim) + (BrDim * BcDim) + 3 * BrDim) * 4
    
    # Adjust block sizes if needed to fit in shared memory
    while shared_mem_size > max_shared_mem_bytes:
        BcDim -= 1
        BrDim = min(dim, BcDim)
        shared_mem_size = (2 * (BrDim * dim) + 2 * (BcDim * dim) + (BrDim * BcDim) + 3 * BrDim) * 4
    
    # Set thread block dimensions
    TrDim = 16
    TcDim = 32
    tpb = (TcDim, TrDim)
    blocks = (cdiv(N, BrDim), Nh, Bs)
    
    # Prepare input and output arrays
    Q_flat = Q.flatten().contiguous()
    K_flat = K.flatten().contiguous()
    V_flat = V.flatten().contiguous()
    O_flat = torch.zeros_like(Q_flat)
    L_flat = torch.zeros(Bs * Nh * N, device=Q.device)
    
    # Launch kernel
    flash_attention[blocks, tpb, 0, shared_mem_size](
        ncuda.as_cuda_array(Q_flat),
        ncuda.as_cuda_array(K_flat),
        ncuda.as_cuda_array(V_flat),
        ncuda.as_cuda_array(O_flat),
        ncuda.as_cuda_array(L_flat),
        BrDim, BcDim, Bs, Nh, N, dim
    )
    
    # Reshape output and convert back to input dtype if needed
    O = O_flat.view(Bs, Nh, N, dim)
    L = L_flat.view(Bs, Nh, N)
    
    if input_dtype != torch.float32:
        O = O.to(input_dtype)
        L = L.to(input_dtype)
    
    return O, L 