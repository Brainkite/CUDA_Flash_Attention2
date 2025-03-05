# ⚡FlashAttention2⚡ Implementation in Numba and CUDA

This repository attempts to implements the FlashAttention 2 algorithm as described in the research paper. The project provides two implementations:

- **Numba-based Implementation:** A Python version using Numba's CUDA JIT to gain insight into the algorithm before the full CUDA implementation.
- **CUDA C++ Implementation:** Work in progress CUDA implementation. Currently numericaly stable and accurate and reproduces the tiled computation of attention as described in the Flash Attention 2 Paper.

## Overview

🔍 FlashAttention 2 efficiently computes scaled dot-product attention by splitting input matrices into blocks and leveraging on-chip SRAM, custom reductions, and careful kernel synchronization. This project is intended to:

- 🔬 Experiment with advanced CUDA programming techniques
- ⚡ Provide a reference Numba implementation for rapid prototyping and testing
- 📊 Offer profiling and benchmarking tools to evaluate performance

## Installation and Requirements

### Hardware

- CUDA-enabled GPU

### Software Requirements

#### NVIDIA CUDA Toolkit and Libraries

Ensure you have the CUDA Toolkit installed (tested with CUDA 12.x) along with compatible NVIDIA drivers.

#### Python Environment

Use Python 3.7 or later
- Numba
- NumPy
- PyTorch

Simply install these with:

```bash
pip install numba numpy torch
```


## Running the Code

### Numba Implementation

To run the Numba-based version:

Run the main Numba script:

```bash
python numba_flash2/src/implementations/flash_attention2_numba.py
```

Or run the tests for the Numba implementation:

```bash
python -m unittest discover -s numba_flash2/src/tests -p '*_test.py'
```

### CUDA Implementation

#### Building the CUDA Code

Navigate to the repository root and compile using nvcc. For example:

```bash
nvcc -arch=sm_70 cuda_flash2/src/flash_attention_2.cu -o flash_attention_2
```

#### Running the CUDA Code

Execute the compiled binary:

```bash
./flash_attention_2
```

#### Running CUDA Tests

CUDA tests are located in the `cuda_flash2/src/tests` directory. Compile and run them similarly. For example, to compile a test:

```bash
nvcc -arch=sm_70 cuda_flash2/src/tests/test_flash_attention2.cu -o test_flash_attention2
```

Then execute:

```bash
./test_flash_attention2
```

### Profiling

For performance analysis, additional profiling tools and scripts are provided. Check the `cuda_flash2/profile_flash_attention` directory and refer to `cuda_flash2/profiling_report.md` for details on interpreting the profiling results.
