import unittest
import torch
import math
import numpy as np
import numba.cuda as ncuda
from numba.cuda.testing import CUDATestCase
from numba.cuda import as_cuda_array as ca
from python_numba_cuda.src.utils import cdiv

### DEFINITION ###

@ncuda.jit(device=True, inline=True)
def assign_shared_memory(BrDim, BcDim, dim):
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
    return Qs, Ks, Ss, Vs, Os, ls, ms, expMaxDelta

### TEST ###

@ncuda.jit
def test_assign_shared_memory_kernel(output, BrDim, BcDim, dim):

    BrIdx = ncuda.blockIdx.x
    TcDim = ncuda.blockDim.x
    TrDim = ncuda.blockDim.y
    TcIdx = ncuda.threadIdx.x
    TrIdx = ncuda.threadIdx.y

    Qs, Ks, Ss, Vs, Os, ls, ms, expMaxDelta = assign_shared_memory(BrDim, BcDim, dim)
    
    # Store the sizes of each array in the output
    if ncuda.threadIdx.x == 0 and ncuda.threadIdx.y == 0:
        output[0] = len(Qs)
        output[1] = len(Ks)
        output[2] = len(Ss)
        output[3] = len(Vs)
        output[4] = len(Os)
        output[5] = len(ls)
        output[6] = len(ms)
        output[7] = len(expMaxDelta)

class TestAssignSharedMemory(CUDATestCase):
    def test_assign_shared_memory(self):
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
        print('BrDim:',BrDim,'BcDim:',BcDim)

        TrDim = 16
        TcDim = 32
        tpb = (TcDim, TrDim)
        blocks = cdiv(N,BrDim)
        print('blocks:', blocks, 'tpb:', tpb, 'shared_mem_size:', shared_mem_size, "max_shared_mem_bytes:", max_shared_mem_bytes)

        # Calculate expected sizes
        BrSize = BrDim * dim
        BcSize = BcDim * dim
        BattSize = BrDim * BcDim

        expected_sizes = torch.tensor([
            BrSize,          # Qs
            BcSize,          # Ks
            BattSize,        # Ss
            BcSize,          # Vs
            BrSize,          # Os
            BrDim,           # ls
            BrDim            # ms
        ], dtype=torch.int32)

        # Prepare output array
        output = torch.zeros(7, dtype=torch.int32).contiguous().cuda()

        # Launch the test kernel
        test_assign_shared_memory_kernel[blocks, tpb, 0, shared_mem_size](ca(output), BrDim, BcDim, dim)

        # Compare results
        torch.testing.assert_close(output.cpu(), expected_sizes)
        print('output   :',output.cpu().tolist())
        print('expected:',expected_sizes.tolist())

if __name__ == '__main__':
    unittest.main()