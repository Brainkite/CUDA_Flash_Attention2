import unittest
import torch
import math
import numpy as np
import numba.cuda as ncuda
from numba.cuda.testing import CUDATestCase
from numba.cuda import as_cuda_array as ca
from python_numba_cuda.src.utils import cdiv
from python_numba_cuda.src.device_functions.assign_shared_memory import assign_shared_memory

### DEFINITION ###

@ncuda.jit(device=True, inline=True)
def initialize_ms(ms, BrDim, TrDim, TcDim, TrIdx, TcIdx):
    for i in range(TrIdx * TcDim + TcIdx, BrDim, TrDim * TcDim):
        if i < BrDim:
            ms[i] = -1e30

### TESTS ###

@ncuda.jit
def test_initialize_ms_kernel(output, BrDim, BcDim, N, dim):

    BrIdx = ncuda.blockIdx.x
    TcDim = ncuda.blockDim.x
    TrDim = ncuda.blockDim.y
    TcIdx = ncuda.threadIdx.x
    TrIdx = ncuda.threadIdx.y

    Qs, Ks, Ss, Vs, Os, ls, ms, expMaxDelta = assign_shared_memory(BrDim, BcDim, dim)
    
    # Initialize ms
    initialize_ms(ms, BrDim, TrDim, TcDim, TrIdx, TcIdx)
    ncuda.syncthreads()
    
    # Copy ms to output
    if ncuda.threadIdx.x == 0 and ncuda.threadIdx.y == 0:
        for i in range(BrDim):
            output[i] = ms[i]

class TestInitializeMs(CUDATestCase):
    def test_initialize_ms(self):
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

        # Prepare output array
        output = torch.zeros(BrDim, dtype=torch.float32).contiguous().cuda()

        # Launch the test kernel
        test_initialize_ms_kernel[blocks, tpb, 0, shared_mem_size](ca(output), BrDim, BcDim, N, dim)

        # Compare results
        expected = torch.full((BrDim,), -1e30, dtype=torch.float32)
        torch.testing.assert_close(output.cpu(), expected)
        print('output   :', output.cpu()[:10].tolist())
        print('expected :', expected[:10].tolist())

if __name__ == '__main__':
    unittest.main()