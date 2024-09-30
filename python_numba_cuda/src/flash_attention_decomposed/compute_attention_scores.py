import unittest
import torch
import math
import numpy as np
import numba.cuda as ncuda
from numba.cuda.testing import CUDATestCase
from numba.cuda import as_cuda_array as ca
from python_numba_cuda.src.utils import cdiv
from python_numba_cuda.src.device_functions.assign_shared_memory import assign_shared_memory
from python_numba_cuda.src.device_functions.load_Q_into_shared_memory import load_Q_into_shared_memory
from python_numba_cuda.src.device_functions.load_KV_into_shared_memory import load_KV_into_shared_memory
from python_numba_cuda.src.device_functions.initialize_ms import initialize_ms


@ncuda.jit(device=True, inline=True)
def compute_attention_scores(Qs, Ks, Ss, BrIdx, BcIdx, BrDim, BcDim, TrDim, TcDim, TrIdx, TcIdx, N, dim):
    """
    Compute the attention scores of current row block and col block, store them in Ss in shared memory.
    """
    sqrt_dim = math.sqrt(dim)
    for i in range(TrIdx, BrDim, TrDim):
        for j in range(TcIdx, BcDim, TcDim):
            if (BrIdx*BrDim+i < N) and (BcIdx*BcDim+j < N):
                s = 0.0
                for k in range(dim):
                    s += Qs[i * dim + k] * Ks[j * dim + k]
                Ss[i * BcDim + j] = s / sqrt_dim

@ncuda.jit
def test_compute_attention_scores_kernel(Q, K, V, output, BrDim, BcDim, N, dim):
    BrIdx = ncuda.blockIdx.x
    TcDim = ncuda.blockDim.x
    TrDim = ncuda.blockDim.y
    TcIdx = ncuda.threadIdx.x
    TrIdx = ncuda.threadIdx.y

    Qs, Ks, Ss, Vs, Os, ls, ms, expMaxDelta = assign_shared_memory(BrDim, BcDim, dim)
    initialize_ms(ms, BrDim, TrDim, TcDim, TrIdx, TcIdx)
    load_Q_into_shared_memory(Q, Qs, BrIdx, BrDim, TrDim, TcDim, TrIdx, TcIdx, N, dim)
    for BcIdx in range(0, N, BcDim):
        load_KV_into_shared_memory(K, V, Ks, Vs, BcIdx, BcDim, TrDim, TcDim, TrIdx, TcIdx, N, dim)
        ncuda.syncthreads()

        # Compute attention scores
        compute_attention_scores(Qs, Ks, Ss, BrIdx, BcIdx, BrDim, BcDim, TrDim, TcDim, TrIdx, TcIdx, N, dim)
        ncuda.syncthreads()

        # Copy results to output
        if TrIdx == 0 and TcIdx == 0 and BrIdx == 0 and BcIdx == 0:
            for i in range(BrDim):
                for j in range(BcDim):
                    output[i * BcDim + j] = Ss[i * BcDim + j]


class TestComputeAttentionScores(CUDATestCase):
    def test_compute_attention_scores(self):
        # Test parameters
        N = 65
        dim = 32
        max_shared_mem_bytes = ncuda.get_current_device().MAX_SHARED_MEMORY_PER_BLOCK
        BcDim = min(math.ceil(max_shared_mem_bytes / (4 * 4 * dim)), N)
        BrDim = min(dim, BcDim)
        shared_mem_size = (2 * (BrDim * dim) + 2 * (BcDim * dim) + (BrDim * BcDim) + 3 * BrDim) * 4
        while shared_mem_size > max_shared_mem_bytes:
            BcDim -= 1
            BrDim = min(dim, BcDim)
            shared_mem_size = (2 * (BrDim * dim) + 2 * (BcDim * dim) + (BrDim * BcDim) + 3 * BrDim) * 4

        TrDim = 16
        TcDim = 32
        tpb = (TcDim, TrDim)
        blocks = cdiv(N, BrDim)
        print('blocks:', blocks, 'tpb:', tpb, 'shared_mem_size:', shared_mem_size, "max_shared_mem_bytes:", max_shared_mem_bytes)

        # Prepare input and output arrays
        Q = torch.randn(N * dim, dtype=torch.float32).contiguous().cuda()
        K = torch.randn(N * dim, dtype=torch.float32).contiguous().cuda()
        V = torch.randn(N * dim, dtype=torch.float32).contiguous().cuda()
        output = torch.zeros(BrDim * BcDim, dtype=torch.float32).contiguous().cuda()


        test_compute_attention_scores_kernel[blocks, tpb, 0, shared_mem_size](ca(Q), ca(K), ca(V), ca(output), BrDim, BcDim, N, dim)


        Q_0 = Q.view(N, dim)[:BrDim].clone()
        K_0 = K.view(N, dim)[:BcDim].clone()
        sqrt_dim = math.sqrt(dim)
        expected = torch.matmul(Q_0, K_0.T) / sqrt_dim
        expected = expected.flatten().cpu()

        # Compare results
        print('BrDim:', BrDim, 'BcDim:', BcDim, 'dim:', dim)
        print('expected shape:', expected.shape)
        print('output shape:', output.shape)
        print('Output    :', output.cpu()[:10].tolist())
        print('Expected  :', expected.flatten()[:10].tolist())
        torch.testing.assert_close(output.cpu(), expected)



if __name__ == '__main__':
    unittest.main()
