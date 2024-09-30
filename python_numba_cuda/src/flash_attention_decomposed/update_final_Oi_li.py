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
from python_numba_cuda.src.device_functions.compute_li_and_Oi import compute_li_and_Oi
torch.set_printoptions(precision=6)

@ncuda.jit(device=True, inline=True)
def update_final_Oi_li(Os, ls, ms, BrIdx, BrDim, TrDim, TcDim, TrIdx, TcIdx, N, dim):
    """
    Update final Oi (output block of the i-eth row block) using li (log sum exp of the i-eth row block).
    Update li (log sum exp of the i-eth row block) using ms (row_max of the i-eth row block).
    """
    for i in range(TrIdx, BrDim, TrDim):
        if BrIdx * BrDim + i < N:
                for j in range(TcIdx, dim, TcDim):
                    Os[i * dim + j] /= ls[i]
                if TcIdx == 0:
                    ls[i] = ms[i] + math.log(ls[i])

@ncuda.jit
def test_update_final_Oi_li(Q, K, V, output_Oi, output_li, BrDim, BcDim, N, dim):
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
    ncuda.syncthreads()

    update_final_Oi_li(Os, ls, ms, BrIdx, BrDim, TrDim, TcDim, TrIdx, TcIdx, N, dim)
    
    if TrIdx == 0 and TcIdx == 0 and BrIdx == 0 and BcIdx == 0:
        for i in range(BrDim):
            for j in range(dim):
                output_Oi[i * dim + j] = Os[i * dim + j]
            output_li[i] = ls[i]


class TestComputeLAndpO(CUDATestCase):
    def test_update_final_Oi_li(self):
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
        output_Oi = torch.zeros(BrDim * dim, dtype=torch.float32).contiguous().cuda()
        output_li = torch.zeros(BrDim, dtype=torch.float32).contiguous().cuda()


        test_update_final_Oi_li[blocks, tpb, 0, shared_mem_size](Q, K, V, output_Oi, output_li, BrDim, BcDim, N, dim)


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
        ls = torch.clamp(l0 * expMaxDelta + torch.sum(P_0, dim=1), min=1e-30)
        print("ls:", ls[:5]) 
        pv_0 = torch.matmul(P_0, V_0)
        print("pv_0:", pv_0.flatten()[:5])
        O_0 = O_0 / expMaxDelta.unsqueeze(1) + pv_0
        print("O_0:", O_0.flatten()[:5])
        O_0 /= ls.unsqueeze(1)
        expected_ls = ms + torch.log(ls)
        expected_Oi = O_0.flatten()

        print("expected_Oi:", expected_Oi[:5])
        print("output_Oi:", output_Oi[:5])
        print("expected_ls:", expected_ls[:5])
        print("output_li:", output_li[:5])
        torch.testing.assert_close(output_Oi.cpu(), expected_Oi.cpu())
        torch.testing.assert_close(output_li.cpu(), expected_ls.cpu())

if __name__ == '__main__':
    unittest.main()
