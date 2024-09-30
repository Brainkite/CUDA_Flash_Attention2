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
from python_numba_cuda.src.device_functions.compute_mi_and_Pi import compute_mi_and_Pi

torch.set_printoptions(precision=6)

@ncuda.jit(device=True, inline=True)
def compute_li_and_Oi(Ss, Vs, Os, expMaxDelta, ls, BrIdx, BcIdx, BcDim, BrDim, TrIdx, TcIdx, TcDim, TrDim, N, dim):
    """
    Compute the li (log sum exp) and Oi (partial output) of the i-eth row block.
    """
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
                ls[i] = max(ls[i] * expMaxDelta[i] + row_sum, 1e-30)
            
            # Update Os
            for j in range(TcIdx, dim, TcDim):
                pv = 0.0
                for k in range(BcDim):
                    if BcIdx * BcDim + k < N:
                        pv += Ss[i * BcDim + k] * Vs[k * dim + j]
                
                Os[i * dim + j] = Os[i * dim + j] / expMaxDelta[i] + pv


@ncuda.jit
def test_compute_li_and_Oi(Q, K, V, output_ls, output_pO, BrDim, BcDim, N, dim):
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
        ncuda.syncthreads()
        compute_li_and_Oi(Ss, Vs, Os, expMaxDelta, ls, BrIdx, BcIdx, BcDim, BrDim, TrIdx, TcIdx, TcDim, TrDim, N, dim)
    
    if TrIdx == 0 and TcIdx == 0 and BrIdx == 0 and BcIdx == 0:
        for i in range(BrDim):
            output_ls[i] = ls[i]
            for j in range(dim):
                output_pO[i * dim + j] = Os[i * dim + j]
        


class TestComputeLiAndOi(CUDATestCase):
    def test_compute_li_and_Oi(self):
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
        output_ls = torch.zeros(BrDim, dtype=torch.float32).contiguous().cuda()
        output_pO = torch.zeros(BrDim * dim, dtype=torch.float32).contiguous().cuda()


        test_compute_li_and_Oi[blocks, tpb, 0, shared_mem_size](Q, K, V, output_ls, output_pO, BrDim, BcDim, N, dim)


        Q_0 = Q.view(N, dim)[:BrDim].clone()
        K_0 = K.view(N, dim)[:BcDim].clone()
        V_0 = V.view(N, dim)[:BcDim].clone()
        O_0 = torch.zeros((BrDim, dim), dtype=torch.float32).contiguous().cuda()
        m0 = torch.full((BrDim,), -1e30, dtype=torch.float32).contiguous().cuda()
        l0 = torch.zeros(BrDim, dtype=torch.float32).contiguous().cuda()

        S_0 = torch.matmul(Q_0, K_0.T) / math.sqrt(dim) 
        ms = S_0.max(dim=1).values 
        ms = torch.max(m0, ms)
        print("ms:", ms[:5])
        P_0 = torch.exp(S_0 - ms.unsqueeze(1))
        delta = m0 - ms
        print("maxDelta:", delta[:5])
        expMaxDelta = torch.exp(delta)
        expMaxDelta = torch.full_like(expMaxDelta, 1e-30)
        print("expMaxDelta:", expMaxDelta[:5])
        expected_ls = torch.clip(l0 * expMaxDelta + torch.sum(P_0, dim=1), min=1e-30)
        print("expected_ls:", expected_ls[:5])
        pv_0 = torch.matmul(P_0, V_0)
        print("pv_0:", pv_0.flatten()[:5])
        O_0 = O_0 / expMaxDelta.unsqueeze(1) + pv_0
        print("O_0:", O_0.flatten()[:5])
        expected_pO = O_0.flatten()

        print('output_ls:', output_ls[:7])
        print('expected_ls:', expected_ls[:7])
        print()
        print('output_pO:', output_pO[:7])
        print('expected_pO:', expected_pO[:7])
        torch.testing.assert_close(output_ls.cpu(), expected_ls.cpu())
        torch.testing.assert_close(output_pO.cpu(), expected_pO.cpu())



if __name__ == '__main__':
    unittest.main()
