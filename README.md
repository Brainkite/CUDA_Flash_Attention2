# FlashAttention 2 Implementation in Numba and CUDA

This repository is an **educational exercise** to implement the FlashAttention 2 algorithm from the research paper description. The project demonstrates advanced CUDA programming and parallel processing techniques by implementing the algorithm in two ways:

- **Numba-based Implementation:**  
  Uses Python with Numba's CUDA JIT facilities; the implementation is modularized into several components such as shared-memory assignment, loading Q/K/V, computing attention scores, reductions (log-sum-exp), and final output updates.
  
- **CUDA C++ Implementation:**  
  Uses raw CUDA to implement the FlashAttention 2 forward pass with emphasis on performance optimizations and debugging through profiling.

## Overview

FlashAttention 2 efficiently computes scaled dot-product attention by splitting input matrices into blocks and leveraging on-chip SRAM, custom reductions, and careful kernel synchronization. This exercise follows the algorithm described below:

```latex
\begin{algorithm}[H]
  \caption{\small\label{alg:fwd_full}\sysname Forward Pass}
  \begin{algorithmic}[1]
    \REQUIRE Matrices $\vQ, \vK, \vV \in \mathbb{R}^{N \times d}$ in HBM, on-chip SRAM of
    size $M$, softmax scaling constant $\tau \in \mathbb{R}$, masking function
    $\textsc{mask}$, dropout probability $p_\mathrm{drop}$.
    \STATE Initialize the pseudo-random number generator state ${\cal R}$ and save to HBM.
    \STATE Set block sizes $B_c = \left\lceil \frac{M}{4d} \right\rceil, B_r = \min \left( \left\lceil \frac{M}{4d} \right\rceil , d \right)$.
    \STATE Initialize $\vO = (0)_{N \times d} \in \mathbb{R}^{N \times d}, \ell = (0)_N \in \mathbb{R}^{N}, m = (-\infty)_N \in \mathbb{R}^{N}$ in HBM.
    \STATE Divide $\vQ$ into $T_r = \left\lceil\frac{N}{B_r} \right\rceil$ blocks $\vQ_1, \dots, \vQ_{T_r}$ of size $B_r \times d$ each,
    and divide $\vK, \vV$ in to $T_c = \left\lceil \frac{N}{B_c} \right\rceil$ blocks $\vK_1, \dots, \vK_{T_c}$ and
    $\vV_1, \dots, \vV_{T_c}$, of size $B_c \times d$ each.
    \STATE Divide $\vO$ into $T_r$ blocks $\vO_i, \dots, \vO_{T_r}$ of size
    $B_r \times d$ each, divide $\ell$ into $T_r$ blocks $\ell_i, \dots, \ell_{T_r}$ of size
    $B_r$ each, divide $m$ into $T_r$ blocks $m_1, \dots, m_{T_r}$ of size $B_r$ each.
    \FOR{$1 \le j \le T_c$}
      \STATE Load $\vK_j, \vV_j$ from HBM to on-chip SRAM.
      \FOR{$1 \le i \le T_r$}
        \STATE Load $\vQ_i, \vO_i, \ell_i, m_i$ from HBM to on-chip SRAM.
        \STATE On chip, compute $\vS_{ij} = \tau \vQ_i \vK_j^T \in \mathbb{R}^{B_r \times B_c}$.
        \STATE On chip, compute $\vS_{ij}^{\mathrm{masked}} = \textsc{mask}(\vS_{ij})$.
        \STATE On chip, compute $\tilde{m}_{ij} = \mathrm{rowmax}(\vS_{ij}^{\mathrm{masked}}) \in \mathbb{R}^{B_r}$, $\tilde{\vP}_{ij} = \exp(\vS_{ij}^{\mathrm{masked}} - \tilde{m}_{ij}) \in \mathbb{R}^{B_r \times B_c}$ (pointwise),
        $\tilde{\ell}_{ij} = \mathrm{row sum}(\tilde{\vP}_{ij}) \in \mathbb{R}^{B_r}$.
        \STATE On chip, compute $m_i^{\mathrm{new}} = \max(m_i, \tilde{m}_{ij}) \in \mathbb{R}^{B_r}$, $\ell_i^{\mathrm{new}} = e^{m_i - m_i^{\mathrm{new}}} \ell_i + e^{\tilde{m}_{ij} - m_i^{\mathrm{new}}} \tilde{\ell}_{ij} \in \mathbb{R}^{B_r}$.
        \STATE On chip, compute $\tilde{\vP}_{ij}^{\mathrm{dropped}} = \mathrm{dropout}(\tilde{\vP}_{ij}, p_\mathrm{drop})$.
        \STATE Write $\vO_i \leftarrow \diag(\ell_i^{\mathrm{new}})^{-1}(\diag(\ell_i) e^{m_i - m_i^{\mathrm{new}}} \vO_i + e^{\tilde{m}_{ij} - m_i^{\mathrm{new}}}\tilde{\vP}_{ij}^{\mathrm{dropped}} \vV_j)$
        to HBM.
        \STATE Write $\ell_i \leftarrow \ell_i^{\mathrm{new}}$, $m_i \leftarrow m_i^{\mathrm{new}}$ to HBM.
      \ENDFOR
    \ENDFOR
    \STATE Return $\vO, \ell, m, {\cal R}$.
  \end{algorithmic}
\end{algorithm}
```

## Repository Structure

```
.
├── cuda_flash2
│   ├── flash_attention_2.cu      # Main CUDA implementation
│   ├── cuda_check.cu            # CUDA environment validation
│   └── profile_output.ncu-rep   # Performance profiling results
├── numba_flash2
│   ├── __init__.py
│   └── src
│       ├── flash_attention2.py           # Main Numba implementation
│       ├── flash_attention2_pytorch.py   # PyTorch reference implementation
│       ├── utils.py                      # Utility functions
│       └── flash_attention_decomposed/   # Modular components
└── examples
    └── UsageExample.ipynb       # Comprehensive usage examples and benchmarks
```

## Documentation

The codebase is extensively documented with:

- **Inline Documentation:** Detailed function and class docstrings explaining:
  - Purpose and algorithm details
  - Input/output specifications
  - Performance considerations
  - Memory layout and management
  - Implementation notes

- **Usage Examples:** The `examples/UsageExample.ipynb` notebook provides:
  - Step-by-step usage instructions
  - Performance benchmarking
  - Comparison with PyTorch's native attention
  - Parameter tuning guidelines

## Installation and Requirements

- **Hardware:** CUDA-enabled GPU  
- **Software:**  
  - Python (>= 3.7)  
  - [Numba](https://numba.pydata.org/)  
  - [PyTorch](https://pytorch.org/)  
  - CUDA Toolkit (compatible version with your GPU)

### Dependencies

Install the required packages:

```bash
pip install numba torch
```

## Running the Code

### Numba Version

To run the tests for the Numba implementation:

```bash
python -m unittest discover -s numba_flash2/src -p '*_test.py'
```

Or run the main flash attention script:

```bash
python numba_flash2/src/flash_attention2.py
```

### CUDA Version

Compile the CUDA code with `nvcc`:

```bash
nvcc -arch=sm_70 cuda_flash2/flash_attention_2.cu -o flash_attention_2
```

Then execute:

```bash
./flash_attention_2
```

## Performance Tuning

The implementation allows for performance tuning through several parameters:

1. **Block Sizes:**
   - `BrDim`: Row block size
   - `BcDim`: Column block size
   These can be adjusted based on your GPU's shared memory capacity and compute capabilities.

2. **Thread Block Configuration:**
   - Configurable thread block dimensions for optimal occupancy
   - Default values are optimized for common GPU architectures

3. **Memory Management:**
   - Careful shared memory allocation
   - Efficient data loading patterns
   - Optimized memory access patterns

See the usage examples for detailed performance tuning guidelines.

## Benchmarks and Comparison

The file `flash_attention2_pytorch.py` provides utilities for benchmarking and comparing with:
- PyTorch's native scaled dot-product attention
- Standard attention implementation
- Memory usage analysis
- Performance profiling

## Future Work and Contributions

- **Enhance Documentation:** Better inline comments and additional documentation in the docs folder.  
- **Expand Tests:** Organize and extend tests in a dedicated `tests/` folder.  
- **CI Integration:** Add GitHub Actions workflows to run tests automatically.  
- **Performance Tuning:** Further optimize kernel configurations and memory usage.  
- **Additional Examples:** Provide notebooks and tutorials for end-to-end demonstrations.

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

This project was inspired by the FlashAttention 2 paper and serves as an exercise in advanced CUDA programming and parallel processing.
