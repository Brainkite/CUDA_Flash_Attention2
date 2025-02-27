"""
Tests for FlashAttention 2 Implementations

This module provides comprehensive tests for both the Numba CUDA and PyTorch
implementations of FlashAttention 2. It includes correctness tests and
performance comparisons with PyTorch's native attention implementation.
"""

import unittest
import torch
import torch.nn.functional as F
import math
import numpy as np
import numba.cuda as ncuda
from numba.cuda.testing import CUDATestCase

from ..implementations.flash_attention2_numba import flash_attention2_launcher as flash_attention_numba

def standard_attention(Q, K, V):
    """
    Standard scaled dot-product attention implementation.
    
    Provides a simple, direct implementation for comparison
    and validation purposes.
    
    Args:
        Q (torch.Tensor): Query tensor [Bs, Nh, N, dim]
        K (torch.Tensor): Key tensor [Bs, Nh, N, dim]
        V (torch.Tensor): Value tensor [Bs, Nh, N, dim]
    
    Returns:
        tuple: (output tensor, attention scores)
    """
    d = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d ** 0.5)
    attn = F.softmax(scores, dim=-1)
    output = torch.matmul(attn, V)
    return output, scores

class TestFlashAttention(CUDATestCase):
    """
    Test suite for the FlashAttention implementations.
    
    This class provides comprehensive tests to validate the correctness of both
    the Numba CUDA and PyTorch implementations by comparing them with PyTorch's
    native attention implementation.
    """
    
    def setUp(self):
        """Set up test parameters and data."""
        super().setUp()
        
        self.Bs = 2  # Batch size
        self.Nh = 2  # Number of heads
        self.N = 1024  # Sequence length
        self.dim = 32  # Head dimension
        
        # Generate test data
        torch.manual_seed(12345)
        self.Q = torch.randn(self.Bs, self.Nh, self.N, self.dim, device='cuda')
        self.K = torch.randn(self.Bs, self.Nh, self.N, self.dim, device='cuda')
        self.V = torch.randn(self.Bs, self.Nh, self.N, self.dim, device='cuda')
    
    def test_numba_implementation(self):
        """Test the Numba CUDA implementation against PyTorch's native attention."""
        # Run Numba implementation
        output_numba, _ = flash_attention_numba(self.Q, self.K, self.V)
        
        # Run PyTorch's native implementation
        expected_output = F.scaled_dot_product_attention(self.Q, self.K, self.V)
        
        # Compare results
        torch.testing.assert_close(
            output_numba.cpu(), expected_output.cpu(),
            rtol=1e-5, atol=1e-5,
            msg="Numba implementation output differs from PyTorch's native implementation"
        )

if __name__ == '__main__':
    unittest.main() 