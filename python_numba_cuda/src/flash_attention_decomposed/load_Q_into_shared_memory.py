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

### DEFINITION ###

@ncuda.jit(device=True, inline=True)
def load_Q_into_shared_memory(Q, Qs, BrIdx, BrDim, TrDim, TcDim, TrIdx, TcIdx, N, dim):
    for i in range(TrIdx, BrDim, TrDim):
        for j in range(TcIdx, dim, TcDim):
            idx = BrIdx * BrDim + i
            if idx < N:
                Qs[i * dim + j] = Q[idx * dim + j]

### TESTS ###

@ncuda.jit
def test_load_Q_into_shared_memory_kernel(Q, output, BrDim, BcDim, N, dim):
    BrIdx = ncuda.blockIdx.x
    TcDim = ncuda.blockDim.x
    TrDim = ncuda.blockDim.y
    TcIdx = ncuda.threadIdx.x
    TrIdx = ncuda.threadIdx.y

    Qs, Ks, Ss, Vs, Os, ls, ms, expMaxDelta = assign_shared_memory(BrDim, BcDim, dim)
    initialize_ms(ms, BrDim, TrDim, TcDim, TrIdx, TcIdx)
    
    # Load Q into shared memory
    load_Q_into_shared_memory(Q, Qs, BrIdx, BrDim, TrDim, TcDim, TrIdx, TcIdx, N, dim)
    ncuda.syncthreads()
    
    # Copy Qs to output
    if TrIdx == 0 and TcIdx == 0 and BrIdx == 0:
        for i in range(BrDim):
            if BrIdx*BrDim + i < N:
                for j in range(dim):
                    output[i * dim + j] = Qs[i * dim + j]

class TestLoadQIntoSharedMemory(CUDATestCase):
    def test_load_Q_into_shared_memory(self):
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
        blocks = cdiv(N, BrDim)
        print('blocks:', blocks, 'tpb:', tpb, 'shared_mem_size:', shared_mem_size, "max_shared_mem_bytes:", max_shared_mem_bytes)

        # Prepare input and output arrays
        Q = torch.randn(N*dim, dtype=torch.float32).contiguous().cuda()
        output = torch.zeros(BrDim*dim, dtype=torch.float32).contiguous().cuda()

        # Launch the test kernel
        test_load_Q_into_shared_memory_kernel[blocks, tpb, 0, shared_mem_size](ca(Q), ca(output), BrDim, BcDim, N, dim)

        # Compare results
        expected = Q[:BrDim*dim].clone()
        torch.testing.assert_close(output.cpu(), expected.cpu())
        print('output   :', output.cpu()[:10].tolist())
        print('expected :', expected.cpu()[:10].tolist())

if __name__ == '__main__':
    unittest.main()