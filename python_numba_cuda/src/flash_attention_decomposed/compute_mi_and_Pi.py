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
from python_numba_cuda.src.device_functions.compute_attention_scores import compute_attention_scores

torch.set_printoptions(precision=6)

@ncuda.jit(device=True, inline=True)
def compute_mi_and_Pi(Ss, ms, expMaxDelta, BrIdx, BcIdx, BcDim, BrDim, TrIdx, TcIdx, TcDim, TrDim, N):
    """
    Compute the mi (row max) and Pi (softmax attention scores) sored in Ss variable of the i-eth row block.
    Then update the expMaxDelta and ms values.
    """
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


@ncuda.jit
def test_compute_mi_and_Pi(Q, K, V, output_P, output_ms, output_expMaxDelta, BrDim, BcDim, N, dim):
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
        compute_attention_scores(Qs, Ks, Ss, BrIdx, BcIdx, BrDim, BcDim, TrDim, TcDim, TrIdx, TcIdx, N, dim)
        ncuda.syncthreads()
        compute_mi_and_Pi(Ss, ms, expMaxDelta, BrIdx, BcIdx, BcDim, BrDim, TrIdx, TcIdx, TcDim, TrDim, N)
    
    # Copy Ss, ms, and expMaxDelta to output arrays
    if TrIdx == 0 and TcIdx == 0 and BrIdx == 0 and BcIdx == 0:
        for i in range(BrDim):
            for j in range(BcDim):
                output_P[i * BcDim + j] = Ss[i * BcDim + j]
            output_ms[i] = ms[i]
            output_expMaxDelta[i] = expMaxDelta[i]


class TestComputeMAndP(CUDATestCase):
    def test_compute_mi_and_Pi(self):
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
        blocks = cdiv(N,BrDim)
        print('BrDim:', BrDim, 'BcDim:', BcDim, 'N:', N, 'dim:', dim)
        print('blocks:', blocks, 'tpb:', tpb, 'shared_mem_size:', shared_mem_size, "max_shared_mem_bytes:", max_shared_mem_bytes)

        # Prepare input and output arrays
        Q = torch.randn(N * dim, dtype=torch.float32).contiguous().cuda()
        K = torch.randn(N * dim, dtype=torch.float32).contiguous().cuda()
        V = torch.randn(N * dim, dtype=torch.float32).contiguous().cuda()
        m0 = torch.full((BrDim,), -1e30, dtype=torch.float32).contiguous().cuda()
        output_P = torch.zeros(BrDim * BcDim, dtype=torch.float32).contiguous().cuda()
        output_ms = torch.zeros(BrDim, dtype=torch.float32).contiguous().cuda()
        output_expMaxDelta = torch.zeros(BrDim, dtype=torch.float32).contiguous().cuda()


        test_compute_mi_and_Pi[blocks, tpb, 0, shared_mem_size](ca(Q), ca(K), ca(V), ca(output_P), ca(output_ms), ca(output_expMaxDelta), BrDim, BcDim, N, dim)


        Q_0 = Q.view(N, dim)[:BrDim].clone()
        K_0 = K.view(N, dim)[:BcDim].clone()
        sqrt_dim = math.sqrt(dim)
        S_0 = torch.matmul(Q_0, K_0.T) / sqrt_dim
        expected_ms = S_0.max(dim=1).values
        expected_ms = torch.max(m0, expected_ms)
        expected_P = torch.exp(S_0 - expected_ms.unsqueeze(1)).flatten()
        expected_expMaxDelta = torch.exp(m0 - expected_ms)
        expected_expMaxDelta = torch.clamp(expected_expMaxDelta, 1e-30)

        print('output_P:', output_P[:7])
        print('expected_P:', expected_P[:7])
        print()
        print('output_ms:', output_ms[:7])
        print('expected_ms:', expected_ms[:7])
        print()
        print('output_expMaxDelta:', output_expMaxDelta[:7])
        print('expected_expMaxDelta:', expected_expMaxDelta[:7])
        torch.testing.assert_close(output_P.cpu(), expected_P.cpu() )
        torch.testing.assert_close(output_ms.cpu(), expected_ms.cpu())
        torch.testing.assert_close(output_expMaxDelta.cpu(), expected_expMaxDelta.cpu())



if __name__ == '__main__':
    unittest.main()
