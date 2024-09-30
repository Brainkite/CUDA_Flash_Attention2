import unittest
import torch
import math
import numpy as np
import numba.cuda as ncuda
from numba.cuda.testing import CUDATestCase
from numba.cuda import as_cuda_array as ca
from python_numba_cuda.src.utils import cdiv
from python_numba_cuda.src.device_functions.assign_shared_memory import assign_shared_memory
from python_numba_cuda.src.device_functions.initialize_ms import initialize_ms
from python_numba_cuda.src.device_functions.load_Q_into_shared_memory import load_Q_into_shared_memory

@ncuda.jit(device=True, inline=True)
def load_KV_into_shared_memory(K, V, Ks, Vs, BcIdx, BcDim, TrDim, TcDim, TrIdx, TcIdx, N, dim):
    for i in range(TrIdx, BcDim, TrDim):
        for j in range(TcIdx, dim, TcDim):
            idx = BcIdx * BcDim + i
            if idx < N:
                Ks[i * dim + j] = K[idx * dim + j]
                Vs[i * dim + j] = V[idx * dim + j]

@ncuda.jit
def test_load_KV_into_shared_memory_kernel(Q, K, V, output_K, output_V, BrDim, BcDim, N, dim):
    BrIdx = ncuda.blockIdx.x
    TcDim = ncuda.blockDim.x
    TrDim = ncuda.blockDim.y
    TcIdx = ncuda.threadIdx.x
    TrIdx = ncuda.threadIdx.y


    Qs, Ks, Ss, Vs, Os, ls, ms, expMaxDelta = assign_shared_memory(BrDim, BcDim, dim)
    initialize_ms(ms, BrDim, TrDim, TcDim, TrIdx, TcIdx)
    load_Q_into_shared_memory(Q, Qs, BrIdx, BrDim, TrDim, TcDim, TrIdx, TcIdx, N, dim)
    
    # Loop over column blocks of K and V
    for BcIdx in range(0, N, BcDim):
        load_KV_into_shared_memory(K, V, Ks, Vs, BcIdx, BcDim, TrDim, TcDim, TrIdx, TcIdx, N, dim)
        ncuda.syncthreads()
    
        # Copy Ks and Vs to output
        if TcIdx == 0 and TrIdx == 0 and BrIdx == 0 and BcIdx == 0:
            for i in range(BcDim):
                if BcIdx*BcDim + i < N:
                    for j in range(dim):
                        output_K[i * dim + j] = Ks[i * dim + j]
                        output_V[i * dim + j] = Vs[i * dim + j]

class TestLoadKVIntoSharedMemory(CUDATestCase):
    def test_load_KV_into_shared_memory(self):
        # Test parameters
        N = 65
        dim = 32
        max_shared_mem_bytes = ncuda.get_current_device().MAX_SHARED_MEMORY_PER_BLOCK
        BcDim = min(math.ceil(max_shared_mem_bytes / (4 * 4 * dim)), N)
        BrDim = min(dim, BcDim)
        shared_mem_size = (2*(BrDim * dim) + 2*(BcDim * dim) + (BrDim * BcDim) + 3*BrDim) * 4
        while shared_mem_size > max_shared_mem_bytes:
            BcDim = BcDim - 1
            BrDim = min(dim, BcDim)
            shared_mem_size = (2*(BrDim * dim) + 2*(BcDim * dim) + (BrDim * BcDim) + 3*BrDim) * 4
        print('B_r:', BrDim, 'B_c:', BcDim)

        TrDim = 16
        TcDim = 32
        tpb = (TcDim, TrDim)
        blocks = cdiv(N, BcDim)
        print('blocks:', blocks, 'tpb:', tpb, 'shared_mem_size:', shared_mem_size, "max_shared_mem_bytes:", max_shared_mem_bytes)

        # Prepare input and output arrays
        Q = torch.randn(N*dim, dtype=torch.float32).contiguous().cuda()
        K = torch.randn(N*dim, dtype=torch.float32).contiguous().cuda()
        V = torch.randn(N*dim, dtype=torch.float32).contiguous().cuda()
        output_K = torch.zeros(BcDim*dim, dtype=torch.float32).contiguous().cuda()
        output_V = torch.zeros(BcDim*dim, dtype=torch.float32).contiguous().cuda()

        # Launch the test kernel
        test_load_KV_into_shared_memory_kernel[blocks, tpb, 0, shared_mem_size](ca(Q), ca(K), ca(V), ca(output_K), ca(output_V), BrDim, BcDim, N, dim)

        # Compare results
        expected_K = K[:BcDim*dim].clone()
        expected_V = V[:BcDim*dim].clone()
        torch.testing.assert_close(output_K.cpu(), expected_K.cpu())
        torch.testing.assert_close(output_V.cpu(), expected_V.cpu())
        print('output K  :', output_K.cpu()[:10].tolist())
        print('expected K:', expected_K.cpu()[:10].tolist())
        print('output V  :', output_V.cpu()[:10].tolist())
        print('expected V:', expected_V.cpu()[:10].tolist())

if __name__ == '__main__':
    unittest.main()