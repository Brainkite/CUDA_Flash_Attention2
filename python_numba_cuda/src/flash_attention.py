import unittest
import torch
import torch.nn.functional as F
import math
import numpy as np
import numba.cuda as ncuda
from numba.cuda.testing import CUDATestCase
from numba.cuda import as_cuda_array as ca
from python_numba_cuda.src.utils import cdiv
torch.set_printoptions(precision=4)

@ncuda.jit
def flash_attention(Q, K, V, O, L, BrDim, BcDim, N, dim):
    """
    # Prepare shared memory
    # load Q row block into shared memory
    # for each col block:
    #     load K and V col block into shared memory
    #     compute attention scores
    #     compute mi
    #     compute Pi
    #     compute li
    #     compute Oi
    # update final Oi and li
    # write to output
    """
    BrIdx = ncuda.blockIdx.x
    TcDim = ncuda.blockDim.x
    TrDim = ncuda.blockDim.y
    TcIdx = ncuda.threadIdx.x
    TrIdx = ncuda.threadIdx.y

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
                Qs[i * dim + j] = Q[idx * dim + j]

    ### Loop over column blocks of K and V
    num_Bc = (N + BcDim - 1) // BcDim
    for BcIdx in range(0, num_Bc):

        ### Load K and V into shared memory
        for i in range(TrIdx, BcDim, TrDim):
            for j in range(TcIdx, dim, TcDim):
                idx = BcIdx * BcDim + i
                if idx < N:
                    Ks[i * dim + j] = K[idx * dim + j]
                    Vs[i * dim + j] = V[idx * dim + j]
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
                    expMaxDelta[i] = max(math.exp(ms[i] - row_max), 1e-30)
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
                    ls[i] = max(ls[i] * expMaxDelta[i] + row_sum, 1e-30) #update min value if changing precision
                
                # Update Os
                for j in range(TcIdx, dim, TcDim):
                    pv = 0.0
                    for k in range(BcDim):
                        if BcIdx * BcDim + k < N:
                            pv += Ss[i * BcDim + k] * Vs[k * dim + j]
                    Os[i * dim + j] = Os[i * dim + j] / expMaxDelta[i] + pv
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
                O[idx * dim + j] = Os[i * dim + j]
            if TcIdx == 0:
                L[idx] = ls[i]
    
def standard_attention(Q, K, V):
    d = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d ** 0.5)
    attn = F.softmax(scores, dim=-1)
    output = torch.matmul(attn, V)
    return output, scores

class TestFlashAttention(CUDATestCase):
    def test_flash_attention(self):
        # Test parameters
        N = 1024
        dim = 64

        # Calculate BrDim and BcDim based on shared memory constraints
        max_shared_mem_bytes = ncuda.get_current_device().MAX_SHARED_MEMORY_PER_BLOCK
        BcDim = min(math.ceil(max_shared_mem_bytes / (4 * 4 * dim)), N)
        BrDim = min(dim, BcDim)
        shared_mem_size = (2 * (BrDim * dim) + 2 * (BcDim * dim) + (BrDim * BcDim) + 3 * BrDim) * 4
        while shared_mem_size > max_shared_mem_bytes:
            BcDim -= 1
            BrDim = min(dim, BcDim)
            shared_mem_size = (2 * (BrDim * dim) + 2 * (BcDim * dim) + (BrDim * BcDim) + 3 * BrDim) * 4

        # Calculate threads per block and blocks per grid
        TrDim = 16
        TcDim = 32
        tpb = (TcDim, TrDim)
        blocks = cdiv(N,BrDim)
        print('BrDim:', BrDim, 'BcDim:', BcDim, 'N:', N, 'dim:', dim)
        print('blocks:', blocks, 'tpb:', tpb, 'shared_mem_size:', shared_mem_size, "max_shared_mem_bytes:", max_shared_mem_bytes)

        # Prepare input and output arrays
        Q = torch.randn(N * dim, dtype=torch.float32).contiguous().cuda()
        K = torch.randn(N * dim, dtype=torch.float32).contiguous().cuda()
        V = torch.randn(N * dim, dtype=torch.float32).contiguous().cuda()
        output_O = torch.zeros(N * dim, dtype=torch.float32).contiguous().cuda()
        output_L = torch.zeros(N, dtype=torch.float32).contiguous().cuda()


        flash_attention[blocks, tpb, 0, shared_mem_size](ca(Q), ca(K), ca(V), ca(output_O), ca(output_L), BrDim, BcDim, N, dim)

        # Compute expected output using torch implemntation of flash attention
        Q_torch = Q.view(N, dim).clone()
        K_torch = K.view(N, dim).clone()
        V_torch = V.view(N, dim).clone()
        expected_O_torch = torch.nn.functional.scaled_dot_product_attention(Q_torch, K_torch, V_torch)
        expected_O_torch = expected_O_torch.flatten()

        print("expected_torch_O  :", expected_O_torch.shape, expected_O_torch[:5], expected_O_torch[-5:])
        print("output_O    :", output_O.shape, output_O[:5], output_O[-5:])
        torch.testing.assert_close(output_O.cpu(), expected_O_torch.cpu())

if __name__ == '__main__':
    unittest.main()
