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
from python_numba_cuda.src.device_functions.update_final_Oi_li import update_final_Oi_li
torch.set_printoptions(precision=4)

@ncuda.jit(device=True, inline=True)
def write_output(O, L, Os, ls, BrIdx, BrDim, TrDim, TcDim, TrIdx, TcIdx, N, dim):
    for i in range(TrIdx, BrDim, TrDim):
        idx = BrIdx * BrDim + i
        if idx < N:
            for j in range(TcIdx, dim, TcDim):
                O[idx * dim + j] = Os[i * dim + j]
            if TcIdx == 0:
                L[idx] = ls[i]

@ncuda.jit
def test_update_final_Oi_li(Q, K, V, output_O, output_L, BrDim, BcDim, N, dim):
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
    ncuda.syncthreads()

    write_output(output_O, output_L, Os, ls, BrIdx, BrDim, TrDim, TcDim, TrIdx, TcIdx, N, dim)
    


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
        output_O = torch.zeros(N * dim, dtype=torch.float32).contiguous().cuda()
        output_L = torch.zeros(N, dtype=torch.float32).contiguous().cuda()


        test_update_final_Oi_li[blocks, tpb, 0, shared_mem_size](Q, K, V, output_O, output_L, BrDim, BcDim, N, dim)


        Q_0 = Q.view(N, dim)[:BrDim].clone()
        K_0 = K.view(N, dim)[:BcDim].clone()
        V_0 = V.view(N, dim)[:BcDim].clone()
        O_0 = torch.zeros((BrDim, dim), dtype=torch.float32).contiguous().cuda()
        m0 = torch.full((BrDim,), -1e30, dtype=torch.float32).contiguous().cuda()
        l0 = torch.zeros(BrDim, dtype=torch.float32).contiguous().cuda()

        S_0 = torch.matmul(Q_0, K_0.T) / math.sqrt(dim) 
        ms = S_0.max(dim=1).values 
        ms = torch.max(m0, ms)
        print("ms         :", ms[:5])
        P_0 = torch.exp(S_0 - ms.unsqueeze(1))
        delta = m0 - ms
        print("maxDelta   :", delta[:5])
        expMaxDelta = torch.exp(delta)
        expMaxDelta = torch.full_like(expMaxDelta, 1e-30)
        print("expMaxDelta:", expMaxDelta[:5])
        ls = torch.clamp(l0 * expMaxDelta + torch.sum(P_0, dim=1), min=1e-30)
        print("ls         :", ls[:5]) 
        pv_0 = torch.matmul(P_0, V_0)
        print("pv_0       :", pv_0.flatten()[:5])
        O_0 = O_0 / expMaxDelta.unsqueeze(1) + pv_0
        print("O_0        :", O_0.flatten()[:5])
        O_0 /= ls.unsqueeze(1)
        print("final_O_0  :", O_0.flatten()[:5])
        ls = ms + torch.log(ls)
        print("final_ls   :", ls[:5])
        print()
        O = torch.zeros(N, dim, dtype=torch.float32).contiguous().cuda()
        O[:BrDim] = O_0
        expected_O = O.flatten()
        expected_L = torch.zeros(N, dtype=torch.float32).contiguous().cuda()
        expected_L[:BrDim] = ls

        Q_torch = Q.view(N, dim).clone()
        K_torch = K.view(N, dim).clone()
        V_torch = V.view(N, dim).clone()
        expected_O_torch = torch.nn.functional.scaled_dot_product_attention(Q_torch, K_torch, V_torch)
        expected_O_torch = expected_O_torch.flatten()



        print("expected_O  :", expected_O[:5], expected_O[-5:])
        print("output_O    :", output_O[:5], output_O[-5:])
        print()
        print("expected_L  :", expected_L[:5], expected_L[-5:])
        print("output_L    :", output_L[:5], output_L[-5:])
        print()
        print("expected_torch_O  :", expected_O_torch[:5], expected_O_torch[-5:])
        print("output_O    :", output_O[:5], output_O[-5:])
        torch.testing.assert_close(output_O[:BrDim*dim].cpu(), expected_O[:BrDim*dim].cpu())
        torch.testing.assert_close(output_L[:BrDim].cpu(), expected_L[:BrDim].cpu())
        torch.testing.assert_close(output_O.cpu(), expected_O_torch.cpu())

if __name__ == '__main__':
    unittest.main()
