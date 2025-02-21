# FlashAttention-2:   Faster Attention with Better Parallelism and Work Partitioning

Tri Dao
Department of Computer Science, Princeton University
Department of Computer Science, Stanford University

###### Abstract

Scaling Transformers to longer sequence lengths has been a major problem in the
last several years, promising to improve performance in language modeling and
high-resolution image understanding, as well as to unlock new applications in
code, audio, and video generation.
The attention layer is the main bottleneck in scaling to longer sequences, as
its runtime and memory increase quadratically in the sequence length.
FlashAttention\[ [5](https://ar5iv.labs.arxiv.org/html/2307.08691#bib.bib5 "")\] exploits the asymmetric GPU memory
hierarchy to bring significant memory saving (linear instead of quadratic) and
runtime speedup (2-4×\\times compared to optimized baselines), with no approximation.
However, FlashAttention is still not nearly as fast as optimized matrix-multiply
(GEMM) operations, reaching only 25-40% of the theoretical maximum FLOPs/s.
We observe that the inefficiency is due to suboptimal work partitioning between
different thread blocks and warps on the GPU, causing either low-occupancy or
unnecessary shared memory reads/writes.
We propose FlashAttention-2, with better work partitioning to address these issues.
In particular, we (1) tweak the algorithm to reduce the number of non-matmul
FLOPs (2) parallelize the attention computation, even for a single head, across
different thread blocks to increase occupancy, and (3) within each thread block,
distribute the work between warps to reduce communication through shared memory.
These yield around 2×\\times speedup compared to FlashAttention, reaching 50-73% of the
theoretical maximum FLOPs/s on A100 and getting close to the efficiency of GEMM
operations.
We empirically validate that when used end-to-end to train GPT-style models,
FlashAttention-2 reaches training speed of up to 225 TFLOPs/s per A100 GPU (72% model
FLOPs utilization).111FlashAttention-2 is available at [https://github.com/Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention "")

## 1 Introduction

Scaling up the context length of Transformers \[ [18](https://ar5iv.labs.arxiv.org/html/2307.08691#bib.bib18 "")\] is a
challenge, since the attention layer at their heart has runtime and
memory requirements quadratic in the input sequence length.
Ideally, we would like to go beyond the standard 2k sequence length limit to
train models to understand books, high resolution images, and long-form videos.
Just within the last year, there have been several language models with much
longer context than before: GPT-4 \[ [12](https://ar5iv.labs.arxiv.org/html/2307.08691#bib.bib12 "")\] with context
length 32k, MosaicML’s MPT with context length 65k, and Anthropic’s
Claude with context length 100k.
Emerging use cases such as long document querying and story writing have
demonstrated a need for models with such long context.

To reduce the computational requirement of attention on such long context, there
have been numerous methods proposed to approximate
attention \[ [9](https://ar5iv.labs.arxiv.org/html/2307.08691#bib.bib9 ""), [14](https://ar5iv.labs.arxiv.org/html/2307.08691#bib.bib14 ""), [19](https://ar5iv.labs.arxiv.org/html/2307.08691#bib.bib19 ""), [8](https://ar5iv.labs.arxiv.org/html/2307.08691#bib.bib8 ""), [4](https://ar5iv.labs.arxiv.org/html/2307.08691#bib.bib4 ""), [2](https://ar5iv.labs.arxiv.org/html/2307.08691#bib.bib2 ""), [20](https://ar5iv.labs.arxiv.org/html/2307.08691#bib.bib20 ""), [3](https://ar5iv.labs.arxiv.org/html/2307.08691#bib.bib3 "")\].
Though these methods have seen some use cases, as far as we know, most
large-scale training runs still use standard attention.
Motivated by this, Dao et al. \[ [5](https://ar5iv.labs.arxiv.org/html/2307.08691#bib.bib5 "")\] proposed to reorder the
attention computation and leverages classical techniques (tiling, recomputation)
to significantly speed it up and reduce memory usage from quadratic to linear in
sequence length.
This yields 2-4×\\times wall-clock time speedup over optimized baselines, up to
10-20×\\times memory saving, with no approximation, and as a result FlashAttention has
seen wide adoption in large-scale training and inference of Transformers.

However, context length increases even more, FlashAttention is still not nearly as
efficient as other primitives such as matrix-multiply (GEMM).
In particular, while FlashAttention is already 2-4×\\times faster than a standard
attention implementation, the forward pass only reaches 30-50% of the
theoretical maximum FLOPs/s of the device ( [Fig.5](https://ar5iv.labs.arxiv.org/html/2307.08691#S4.F5 "In 4.1 Benchmarking Attention ‣ 4 Empirical Validation ‣ FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning")), while
the backward pass is even more challenging, reaching only 25-35% of maximum
throughput on A100 GPU ( [Fig.6](https://ar5iv.labs.arxiv.org/html/2307.08691#S4.F6 "In 4.1 Benchmarking Attention ‣ 4 Empirical Validation ‣ FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning")).
In contrast, optimized GEMM can reach up to 80-90% of the theoretical maximum
device throughput.
Through careful profiling, we observe that FlashAttention still has suboptimal work
partitioning between different thread blocks and warps on the GPU, causing
either low-occupancy or unnecessary shared memory reads/writes.

Building on FlashAttention, we propose FlashAttention-2 with better parallelism and work
partitioning to address these challenges.

1. 1.


In [Section3.1](https://ar5iv.labs.arxiv.org/html/2307.08691#S3.SS1 "3.1 Algorithm ‣ 3 FlashAttention-2: Algorithm, Parallelism, and Work Partitioning ‣ FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning"), we tweak the algorithms to reduce the number of non-matmul FLOPs while not
changing the output.
While the non-matmul FLOPs only account for a small fraction of the total FLOPs,
they take longer to perform as GPUs have specialized units for matrix multiply,
and as a result the matmul throughput can be up to 16×\\times higher than non-matmul
throughput.
It is thus important to reduce non-matmul FLOPs and spend as much time as
possible doing matmul FLOPs.

2. 2.


We propose to parallelize both the forward pass and backward pass along
the sequence length dimension, in addition to the batch and number of heads
dimension. This increases occupancy (utilization of GPU resources) in the case
where the sequences are long (and hence batch size is often small).

3. 3.


Even within one block of attention computation, we partition the work
between different warps of a thread block to reduce communication and shared
memory reads/writes.


In [Section4](https://ar5iv.labs.arxiv.org/html/2307.08691#S4 "4 Empirical Validation ‣ FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning"), we empirically validate that FlashAttention-2 yields significant speedup compared to
even FlashAttention. Benchmarks on different settings (with or without causal mask,
different head dimensions) show that FlashAttention-2 achieves around 2×\\times speedup over
FlashAttention, reaching up to 73% of the theoretical max throughput in the
forward pass, and up to 63% of the theoretical max throughput in the backward pass.
When used end-to-end to train GPT-style models, we reach training speed of up to
225 TFLOPs/s per A100 GPU.

## 2 Background

We provide some background on the performance characteristics and execution
model of GPUs.
We also describe the standard implementation of attention, as well as
FlashAttention.

### 2.1 Hardware characteristics

GPU performance characteristics.
The GPU consists of compute elements (e.g., floating point arithmetic units) and
a memory hierarchy.
Most modern GPUs contain specialized units to accelerate matrix multiply in
low-precision (e.g., Tensor Cores on Nvidia GPUs for FP16/BF16 matrix multiply).
The memory hierarchy comprise of high bandwidth memory (HBM), and on-chip SRAM
(aka shared memory).
As an example, the A100 GPU has 40-80GB of high bandwidth memory (HBM) with
bandwidth 1.5-2.0TB/s and 192KB of on-chip SRAM per each of 108 streaming
multiprocessors with bandwidth estimated around 19TB/s \[ [7](https://ar5iv.labs.arxiv.org/html/2307.08691#bib.bib7 ""), [6](https://ar5iv.labs.arxiv.org/html/2307.08691#bib.bib6 "")\].
As the L2 cache is not directly controllable by the programmer, we focus on the
HBM and SRAM for the purpose of this discussion.

Execution Model.
GPUs have a massive number of threads to execute an operation
(called a kernel).
Threads are organized into thread blocks, which are scheduled to run on
streaming multiprocessors (SMs).
Within each thread blocks, threads are grouped into warps (a group of 32
threads).
Threads within a warp can communicate by fast shuffle instructions or cooperate
to perform matrix multiply.
Warps within a thread block can communicate by reading from / writing to shared memory.
Each kernel loads inputs from HBM to registers and SRAM, computes, then writes outputs to HBM.

### 2.2 Standard Attention Implementation

Given input sequences 𝐐,𝐊,𝐕∈ℝN×d𝐐𝐊𝐕superscriptℝ𝑁𝑑\\mathbf{Q},\\mathbf{K},\\mathbf{V}\\in\\mathbb{R}^{N\\times d} where N𝑁N is the sequence length and
d𝑑d is the head dimension, we want to compute the attention output 𝐎∈ℝN×d𝐎superscriptℝ𝑁𝑑\\mathbf{O}\\in\\mathbb{R}^{N\\times d}:

|     |     |     |
| --- | --- | --- |
|  | 𝐒=𝐐𝐊⊤∈ℝN×N,𝐏=softmax​(𝐒)∈ℝN×N,𝐎=𝐏𝐕∈ℝN×d,formulae-sequence𝐒superscript𝐐𝐊topsuperscriptℝ𝑁𝑁𝐏softmax𝐒superscriptℝ𝑁𝑁𝐎𝐏𝐕superscriptℝ𝑁𝑑\\mathbf{S}=\\mathbf{Q}\\mathbf{K}^{\\top}\\in\\mathbb{R}^{N\\times N},\\quad\\mathbf{P}=\\mathrm{softmax}(\\mathbf{S})\\in\\mathbb{R}^{N\\times N},\\quad\\mathbf{O}=\\mathbf{P}\\mathbf{V}\\in\\mathbb{R}^{N\\times d}, |  |

where softmaxsoftmax\\mathrm{softmax} is applied row-wise.222For clarity of exposition, we
omit the scaling of 𝐐𝐊⊤superscript𝐐𝐊top\\mathbf{Q}\\mathbf{K}^{\\top} (typically by 1/d1d1/\\mathrm{d}), and optionally
elementwise masking on 𝐒𝐒\\mathbf{S} and/or dropout applied to 𝐏𝐏\\mathbf{P} For multi-head
attention (MHA), this same computation is performed in parallel across many
heads, and parallel over the batch dimension (number of input sequences in a
batch).

The backward pass of attention proceeds as follows.
Let 𝐝𝐎∈ℝN×d𝐝𝐎superscriptℝ𝑁𝑑\\mathbf{dO}\\in\\mathbb{R}^{N\\times d} be the gradient of 𝐎𝐎\\mathbf{O} with respect to some loss
function. Then by the chain rule (aka backpropagation):

|     |     |     |     |
| --- | --- | --- | --- |
|  | 𝐝𝐕𝐝𝐕\\displaystyle\\mathbf{dV} | =𝐏⊤​𝐝𝐎∈ℝN×dabsentsuperscript𝐏top𝐝𝐎superscriptℝ𝑁𝑑\\displaystyle=\\mathbf{P}^{\\top}\\mathbf{dO}\\in\\mathbb{R}^{N\\times d} |  |
|  | 𝐝𝐏𝐝𝐏\\displaystyle\\mathbf{dP} | =𝐝𝐎𝐕⊤∈ℝN×Nabsentsuperscript𝐝𝐎𝐕topsuperscriptℝ𝑁𝑁\\displaystyle=\\mathbf{dO}\\mathbf{V}^{\\top}\\in\\mathbb{R}^{N\\times N} |  |
|  | 𝐝𝐒𝐝𝐒\\displaystyle\\mathbf{dS} | =dsoftmax​(𝐝𝐏)∈ℝN×Nabsentdsoftmax𝐝𝐏superscriptℝ𝑁𝑁\\displaystyle=\\mathrm{dsoftmax}(\\mathbf{dP})\\in\\mathbb{R}^{N\\times N} |  |
|  | 𝐝𝐐𝐝𝐐\\displaystyle\\mathbf{dQ} | =𝐝𝐒𝐊∈ℝN×dabsent𝐝𝐒𝐊superscriptℝ𝑁𝑑\\displaystyle=\\mathbf{dS}\\mathbf{K}\\in\\mathbb{R}^{N\\times d} |  |
|  | 𝐝𝐊𝐝𝐊\\displaystyle\\mathbf{dK} | =𝐐𝐝𝐒⊤∈ℝN×d,absentsuperscript𝐐𝐝𝐒topsuperscriptℝ𝑁𝑑\\displaystyle=\\mathbf{Q}\\mathbf{dS}^{\\top}\\in\\mathbb{R}^{N\\times d}, |  |

where dsoftmaxdsoftmax\\mathrm{dsoftmax} is the gradient (backward pass) of softmax applied row-wise.
One can work out that if p=softmax​(s)𝑝softmax𝑠p=\\mathrm{softmax}(s) for some vector s𝑠s and p𝑝p, then
with output gradient d​p𝑑𝑝dp, the input gradient d​s=(diag​(p)−p​p⊤)​d​p𝑑𝑠diag𝑝𝑝superscript𝑝top𝑑𝑝ds=(\\mathrm{diag}(p)-pp^{\\top})dp.

Standard attention implementations materialize the matrices 𝐒𝐒\\mathbf{S} and 𝐏𝐏\\mathbf{P} to
HBM, which takes O​(N2)𝑂superscript𝑁2O(N^{2}) memory.
Often N≫dmuch-greater-than𝑁𝑑N\\gg d (typically N𝑁N is on the order of 1k–8k and d𝑑d is around 64–128).
The standard attention implementation (1) calls the matrix multiply (GEMM)
subroutine to multiply 𝐒=𝐐𝐊⊤𝐒superscript𝐐𝐊top\\mathbf{S}=\\mathbf{Q}\\mathbf{K}^{\\top}, writes the result to HBM, then (2)
loads §§\\S from HBM to compute softmax and write the result 𝐏𝐏\\mathbf{P} to HBM, and
finally (3) calls GEMM to get 𝐎=𝐏𝐕𝐎𝐏𝐕\\mathbf{O}=\\mathbf{P}\\mathbf{V}.
As most of the operations are bounded by memory bandwidth, the large number of
memory accesses translates to slow wall-clock time.
Moreover, the required memory is O​(N2)𝑂superscript𝑁2O(N^{2}) due to having to materialize 𝐒𝐒\\mathbf{S} and
𝐏𝐏\\mathbf{P}.
Moreover, one has to save 𝐏∈ℝN×N𝐏superscriptℝ𝑁𝑁\\mathbf{P}\\in\\mathbb{R}^{N\\times N} for the backward pass to compute the
gradients.

### 2.3 FlashAttention

To speed up attention on hardware accelerators such as GPU,
\[ [5](https://ar5iv.labs.arxiv.org/html/2307.08691#bib.bib5 "")\] proposes an algorithm to reduce the memory
reads/writes while maintaining the same output (without approximation).

#### 2.3.1 Forward pass

FlashAttention applies the classical technique of tiling to reduce memory IOs, by
(1) loading blocks of inputs from HBM to SRAM, (2) computing attention with
respect to that block, and then (3) updating the output without writing the
large intermediate matrices 𝐒𝐒\\mathbf{S} and 𝐏𝐏\\mathbf{P} to HBM.
As the softmax couples entire rows or blocks of row, online
softmax \[ [11](https://ar5iv.labs.arxiv.org/html/2307.08691#bib.bib11 ""), [13](https://ar5iv.labs.arxiv.org/html/2307.08691#bib.bib13 "")\] can split the attention
computation into blocks, and rescale the output of each block to finally get the
right result (with no approximation).
By significantly reducing the amount of memory reads/writes, FlashAttention yields
2-4×\\times wall-clock speedup over optimized baseline attention implementations.

We describe the online softmax technique \[ [11](https://ar5iv.labs.arxiv.org/html/2307.08691#bib.bib11 "")\] and how it is
used in attention \[ [13](https://ar5iv.labs.arxiv.org/html/2307.08691#bib.bib13 "")\].
For simplicity, consider just one row block of the attention matrix 𝐒𝐒\\mathbf{S}, of the form
\[𝐒(1)𝐒(2)\]matrixsuperscript𝐒1superscript𝐒2\\begin{bmatrix}\\mathbf{S}^{(1)}&\\mathbf{S}^{(2)}\\end{bmatrix} for some matrices
𝐒(1),𝐒(2)∈ℝBr×Bcsuperscript𝐒1superscript𝐒2superscriptℝsubscript𝐵𝑟subscript𝐵𝑐\\mathbf{S}^{(1)},\\mathbf{S}^{(2)}\\in\\mathbb{R}^{B\_{r}\\times B\_{c}}, where Brsubscript𝐵𝑟B\_{r} and Bcsubscript𝐵𝑐B\_{c} are the row and
column block sizes.
We want to compute softmax of this row block and multiply with the value,
of the form \[𝐕(1)𝐕(2)\]matrixsuperscript𝐕1superscript𝐕2\\begin{bmatrix}\\mathbf{V}^{(1)}\\\
\\mathbf{V}^{(2)}\\end{bmatrix} for some
matrices 𝐕(1),𝐕(2)∈ℝBc×dsuperscript𝐕1superscript𝐕2superscriptℝsubscript𝐵𝑐𝑑\\mathbf{V}^{(1)},\\mathbf{V}^{(2)}\\in\\mathbb{R}^{B\_{c}\\times d}.
Standard softmax would compute:

|     |     |     |     |
| --- | --- | --- | --- |
|  | m𝑚\\displaystyle m | =max⁡(rowmax​(𝐒(1)),rowmax​(𝐒(2)))∈ℝBrabsentrowmaxsuperscript𝐒1rowmaxsuperscript𝐒2superscriptℝsubscript𝐵𝑟\\displaystyle=\\max(\\mathrm{rowmax}(\\mathbf{S}^{(1)}),\\mathrm{rowmax}(\\mathbf{S}^{(2)}))\\in\\mathbb{R}^{B\_{r}} |  |
|  | ℓℓ\\displaystyle\\ell | =rowsum​(e𝐒(1)−m)+rowsum​(e𝐒(2)−m)∈ℝBrabsentrowsumsuperscript𝑒superscript𝐒1𝑚rowsumsuperscript𝑒superscript𝐒2𝑚superscriptℝsubscript𝐵𝑟\\displaystyle=\\mathrm{rowsum}(e^{\\mathbf{S}^{(1)}-m})+\\mathrm{rowsum}(e^{\\mathbf{S}^{(2)}-m})\\in\\mathbb{R}^{B\_{r}} |  |
|  | 𝐏𝐏\\displaystyle\\mathbf{P} | =\[𝐏(1)𝐏(2)\]=diag​(ℓ)−1​\[e𝐒(1)−me𝐒(2)−m\]∈ℝBr×2​Bcabsentmatrixsuperscript𝐏1superscript𝐏2diagsuperscriptℓ1matrixsuperscript𝑒superscript𝐒1𝑚superscript𝑒superscript𝐒2𝑚superscriptℝsubscript𝐵𝑟2subscript𝐵𝑐\\displaystyle=\\begin{bmatrix}\\mathbf{P}^{(1)}&\\mathbf{P}^{(2)}\\end{bmatrix}=\\mathrm{diag}(\\ell)^{-1}\\begin{bmatrix}e^{\\mathbf{S}^{(1)}-m}&e^{\\mathbf{S}^{(2)}-m}\\end{bmatrix}\\in\\mathbb{R}^{B\_{r}\\times 2B\_{c}} |  |
|  | 𝐎𝐎\\displaystyle\\mathbf{O} | =\[𝐏(1)𝐏(2)\]​\[𝐕(1)𝐕(2)\]=diag​(ℓ)−1​e𝐒(1)−m​𝐕(1)+e𝐒(2)−m​𝐕(2)∈ℝBr×d.absentmatrixsuperscript𝐏1superscript𝐏2matrixsuperscript𝐕1superscript𝐕2diagsuperscriptℓ1superscript𝑒superscript𝐒1𝑚superscript𝐕1superscript𝑒superscript𝐒2𝑚superscript𝐕2superscriptℝsubscript𝐵𝑟𝑑\\displaystyle=\\begin{bmatrix}\\mathbf{P}^{(1)}&\\mathbf{P}^{(2)}\\end{bmatrix}\\begin{bmatrix}\\mathbf{V}^{(1)}\\\<br>\\mathbf{V}^{(2)}\\end{bmatrix}=\\mathrm{diag}(\\ell)^{-1}e^{\\mathbf{S}^{(1)}-m}\\mathbf{V}^{(1)}+e^{\\mathbf{S}^{(2)}-m}\\mathbf{V}^{(2)}\\in\\mathbb{R}^{B\_{r}\\times d}. |  |

Online softmax instead computes “local” softmax with respect to each block and
rescale to get the right output at the end:

|     |     |     |     |
| --- | --- | --- | --- |
|  | m(1)superscript𝑚1\\displaystyle m^{(1)} | =rowmax​(𝐒(1))∈ℝBrabsentrowmaxsuperscript𝐒1superscriptℝsubscript𝐵𝑟\\displaystyle=\\mathrm{rowmax}(\\mathbf{S}^{(1)})\\in\\mathbb{R}^{B\_{r}} |  |
|  | ℓ(1)superscriptℓ1\\displaystyle\\ell^{(1)} | =rowsum​(e𝐒(1)−m(1))∈ℝBrabsentrowsumsuperscript𝑒superscript𝐒1superscript𝑚1superscriptℝsubscript𝐵𝑟\\displaystyle=\\mathrm{rowsum}(e^{\\mathbf{S}^{(1)}-m^{(1)}})\\in\\mathbb{R}^{B\_{r}} |  |
|  | 𝐏~(1)superscript~𝐏1\\displaystyle\\tilde{\\mathbf{P}}^{(1)} | =diag​(ℓ(1))−1​e𝐒(1)−m(1)∈ℝBr×Bcabsentdiagsuperscriptsuperscriptℓ11superscript𝑒superscript𝐒1superscript𝑚1superscriptℝsubscript𝐵𝑟subscript𝐵𝑐\\displaystyle=\\mathrm{diag}(\\ell^{(1)})^{-1}e^{\\mathbf{S}^{(1)}-m^{(1)}}\\in\\mathbb{R}^{B\_{r}\\times B\_{c}} |  |
|  | 𝐎(1)superscript𝐎1\\displaystyle\\mathbf{O}^{(1)} | =𝐏~(1)​𝐕(1)=diag​(ℓ(1))−1​e𝐒(1)−m(1)​𝐕(1)∈ℝBr×dabsentsuperscript~𝐏1superscript𝐕1diagsuperscriptsuperscriptℓ11superscript𝑒superscript𝐒1superscript𝑚1superscript𝐕1superscriptℝsubscript𝐵𝑟𝑑\\displaystyle=\\tilde{\\mathbf{P}}^{(1)}\\mathbf{V}^{(1)}=\\mathrm{diag}(\\ell^{(1)})^{-1}e^{\\mathbf{S}^{(1)}-m^{(1)}}\\mathbf{V}^{(1)}\\in\\mathbb{R}^{B\_{r}\\times d} |  |
|  | m(2)superscript𝑚2\\displaystyle m^{(2)} | =max⁡(m(1),rowmax​(𝐒(2)))=mabsentsuperscript𝑚1rowmaxsuperscript𝐒2𝑚\\displaystyle=\\max(m^{(1)},\\mathrm{rowmax}(\\mathbf{S}^{(2)}))=m |  |
|  | ℓ(2)superscriptℓ2\\displaystyle\\ell^{(2)} | =em(1)−m(2)​ℓ(1)+rowsum​(e𝐒(2)−m(2))=rowsum​(e𝐒(1)−m)+rowsum​(e𝐒(2)−m)=ℓabsentsuperscript𝑒superscript𝑚1superscript𝑚2superscriptℓ1rowsumsuperscript𝑒superscript𝐒2superscript𝑚2rowsumsuperscript𝑒superscript𝐒1𝑚rowsumsuperscript𝑒superscript𝐒2𝑚ℓ\\displaystyle=e^{m^{(1)}-m^{(2)}}\\ell^{(1)}+\\mathrm{rowsum}(e^{\\mathbf{S}^{(2)}-m^{(2)}})=\\mathrm{rowsum}(e^{\\mathbf{S}^{(1)}-m})+\\mathrm{rowsum}(e^{\\mathbf{S}^{(2)}-m})=\\ell |  |
|  | 𝐏~(2)superscript~𝐏2\\displaystyle\\tilde{\\mathbf{P}}^{(2)} | =diag​(ℓ(2))−1​e𝐒(2)−m(2)absentdiagsuperscriptsuperscriptℓ21superscript𝑒superscript𝐒2superscript𝑚2\\displaystyle=\\mathrm{diag}(\\ell^{(2)})^{-1}e^{\\mathbf{S}^{(2)}-m^{(2)}} |  |
|  | 𝐎(2)superscript𝐎2\\displaystyle\\mathbf{O}^{(2)} | =diag​(ℓ(1)/ℓ(2))−1​𝐎(1)+𝐏~(2)​𝐕(2)=diag​(ℓ(2))−1​es(1)−m​𝐕(1)+diag​(ℓ(2))−1​es(2)−m​𝐕(2)=𝐎.absentdiagsuperscriptsuperscriptℓ1superscriptℓ21superscript𝐎1superscript~𝐏2superscript𝐕2diagsuperscriptsuperscriptℓ21superscript𝑒superscript𝑠1𝑚superscript𝐕1diagsuperscriptsuperscriptℓ21superscript𝑒superscript𝑠2𝑚superscript𝐕2𝐎\\displaystyle=\\mathrm{diag}(\\ell^{(1)}/\\ell^{(2)})^{-1}\\mathbf{O}^{(1)}+\\tilde{\\mathbf{P}}^{(2)}\\mathbf{V}^{(2)}=\\mathrm{diag}(\\ell^{(2)})^{-1}e^{s^{(1)}-m}\\mathbf{V}^{(1)}+\\mathrm{diag}(\\ell^{(2)})^{-1}e^{s^{(2)}-m}\\mathbf{V}^{(2)}=\\mathbf{O}. |  |

We show how FlashAttention uses online softmax to enable tiling
( [Fig.1](https://ar5iv.labs.arxiv.org/html/2307.08691#S2.F1 "In 2.3.1 Forward pass ‣ 2.3 FlashAttention ‣ 2 Background ‣ FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning")) to reduce memory reads/writes.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2307.08691/assets/figs/flash_attention_diagram.png)Figure 1: Diagram of how FlashAttention forward
pass is performed, when the key 𝐊𝐊\\mathbf{K} is partitioned into two blocks and the
value 𝐕𝐕\\mathbf{V} is also partitioned into two blocks.
By computing attention with respect to each block and rescaling the output,
we get the right answer at the end, while avoiding expensive memory
reads/writes of the intermediate matrices 𝐒𝐒\\mathbf{S} and 𝐏𝐏\\mathbf{P}.
We simplify the diagram, omitting the step in softmax that subtracts each
element by the row-wise max.

#### 2.3.2 Backward pass

In the backward pass, by re-computing the values of the attention matrices 𝐒𝐒\\mathbf{S}
and 𝐏𝐏\\mathbf{P} once blocks of inputs 𝐐,𝐊,𝐕𝐐𝐊𝐕\\mathbf{Q},\\mathbf{K},\\mathbf{V} are already loaded to SRAM,
FlashAttention avoids having to store large intermediate values.
By not having to save the large matrices 𝐒𝐒\\mathbf{S} and 𝐏𝐏\\mathbf{P} of size N×N𝑁𝑁N\\times N,
FlashAttention yields 10-20×\\times memory saving depending on sequence length (memory
required in linear in sequence length N𝑁N instead of quadratic).
The backward pass also achieves 2-4×\\times wall-clock speedup due to reduce memory
reads/writes.

The backward pass applies tiling to the equations in [Section2.2](https://ar5iv.labs.arxiv.org/html/2307.08691#S2.SS2 "2.2 Standard Attention Implementation ‣ 2 Background ‣ FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning").
Though the backward pass is simpler than the forward pass conceptually (there is
no softmax rescaling), the implementation is significantly more involved.
This is because there are more values to be kept in SRAM to perform 5 matrix
multiples in the backward pass, compared to just 2 matrix multiples in the
forward pass.

## 3 FlashAttention-2: Algorithm, Parallelism, and Work Partitioning

We describe the FlashAttention-2 algorithm, which includes several tweaks to FlashAttention to reduce the number of non-matmul FLOPs.
We then describe how to parallelize the computation on different thread blocks
to make full use the GPU resources.
Finally we describe we partition the work between different warps within one
thread block to reduce the amount of shared memory access.
These improvements lead to 2-3×\\times speedup as validated in [Section4](https://ar5iv.labs.arxiv.org/html/2307.08691#S4 "4 Empirical Validation ‣ FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning").

### 3.1 Algorithm

We tweak the algorithm from FlashAttention to reduce the number of non-matmul
FLOPs.
This is because modern GPUs have specialized compute units (e.g., Tensor Cores
on Nvidia GPUs) that makes matmul much faster.
As an example, the A100 GPU has a max theoretical throughput of 312 TFLOPs/s of
FP16/BF16 matmul, but only 19.5 TFLOPs/s of non-matmul FP32.
Another way to think about this is that each non-matmul FLOP is 16×\\times more
expensive than a matmul FLOP.
To maintain high throughput (e.g., more than 50% of the maximum theoretical
TFLOPs/s), we want to spend as much time on matmul FLOPs as possible.

#### 3.1.1 Forward pass

We revisit the online softmax trick as shown in [Section2.3](https://ar5iv.labs.arxiv.org/html/2307.08691#S2.SS3 "2.3 FlashAttention ‣ 2 Background ‣ FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning") and make
two minor tweaks to reduce non-matmul FLOPs:

1. 1.


We do not have to rescale both terms of the output update by diag​(ℓ(2))−1diagsuperscriptsuperscriptℓ21\\mathrm{diag}(\\ell^{(2)})^{-1}:



|     |     |     |
| --- | --- | --- |
|  | 𝐎(2)=diag​(ℓ(1)/ℓ(2))−1​𝐎(1)+diag​(ℓ(2))−1​e𝐒(2)−m(2)​𝐕(2).superscript𝐎2diagsuperscriptsuperscriptℓ1superscriptℓ21superscript𝐎1diagsuperscriptsuperscriptℓ21superscript𝑒superscript𝐒2superscript𝑚2superscript𝐕2\\mathbf{O}^{(2)}=\\mathrm{diag}(\\ell^{(1)}/\\ell^{(2)})^{-1}\\mathbf{O}^{(1)}+\\mathrm{diag}(\\ell^{(2)})^{-1}e^{\\mathbf{S}^{(2)}-m^{(2)}}\\mathbf{V}^{(2)}. |  |



We can instead maintain an “un-scaled” version of 𝐎(2)superscript𝐎2\\mathbf{O}^{(2)} and keep
around the statistics ℓ(2)superscriptℓ2\\ell^{(2)}:



|     |     |     |
| --- | --- | --- |
|  | 𝐎~(2)=diag​(ℓ(1))−1​𝐎(1)+e𝐒(2)−m(2)​𝐕(2).superscript~𝐎2diagsuperscriptsuperscriptℓ11superscript𝐎1superscript𝑒superscript𝐒2superscript𝑚2superscript𝐕2\\tilde{\\mathbf{O}}^{(2)}=\\mathrm{diag}(\\ell^{(1)})^{-1}\\mathbf{O}^{(1)}+e^{\\mathbf{S}^{(2)}-m^{(2)}}\\mathbf{V}^{(2)}. |  |



Only at the every end of the loop do we scale the final
𝐎~(last)superscript~𝐎last\\tilde{\\mathbf{O}}^{(\\mathrm{last})} by diag​(ℓ(last))−1diagsuperscriptsuperscriptℓlast1\\mathrm{diag}(\\ell^{(\\mathrm{last})})^{-1} to get
the right output.

2. 2.


We do not have to save both the max m(j)superscript𝑚𝑗m^{(j)} and the sum of
exponentials ℓ(j)superscriptℓ𝑗\\ell^{(j)} for the backward pass. We only need to store the
logsumexp L(j)=m(j)+log⁡(ℓ(j))superscript𝐿𝑗superscript𝑚𝑗superscriptℓ𝑗L^{(j)}=m^{(j)}+\\log(\\ell^{(j)}).


In the simple case of 2 blocks in [Section2.3](https://ar5iv.labs.arxiv.org/html/2307.08691#S2.SS3 "2.3 FlashAttention ‣ 2 Background ‣ FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning"), the online softmax
trick now becomes:

|     |     |     |     |
| --- | --- | --- | --- |
|  | m(1)superscript𝑚1\\displaystyle m^{(1)} | =rowmax​(𝐒(1))∈ℝBrabsentrowmaxsuperscript𝐒1superscriptℝsubscript𝐵𝑟\\displaystyle=\\mathrm{rowmax}(\\mathbf{S}^{(1)})\\in\\mathbb{R}^{B\_{r}} |  |
|  | ℓ(1)superscriptℓ1\\displaystyle\\ell^{(1)} | =rowsum​(e𝐒(1)−m(1))∈ℝBrabsentrowsumsuperscript𝑒superscript𝐒1superscript𝑚1superscriptℝsubscript𝐵𝑟\\displaystyle=\\mathrm{rowsum}(e^{\\mathbf{S}^{(1)}-m^{(1)}})\\in\\mathbb{R}^{B\_{r}} |  |
|  | 𝐎(1)~~superscript𝐎1\\displaystyle\\tilde{\\mathbf{O}^{(1)}} | =e𝐒(1)−m(1)​𝐕(1)∈ℝBr×dabsentsuperscript𝑒superscript𝐒1superscript𝑚1superscript𝐕1superscriptℝsubscript𝐵𝑟𝑑\\displaystyle=e^{\\mathbf{S}^{(1)}-m^{(1)}}\\mathbf{V}^{(1)}\\in\\mathbb{R}^{B\_{r}\\times d} |  |
|  | m(2)superscript𝑚2\\displaystyle m^{(2)} | =max⁡(m(1),rowmax​(𝐒(2)))=mabsentsuperscript𝑚1rowmaxsuperscript𝐒2𝑚\\displaystyle=\\max(m^{(1)},\\mathrm{rowmax}(\\mathbf{S}^{(2)}))=m |  |
|  | ℓ(2)superscriptℓ2\\displaystyle\\ell^{(2)} | =em(1)−m(2)​ℓ(1)+rowsum​(e𝐒(2)−m(2))=rowsum​(e𝐒(1)−m)+rowsum​(e𝐒(2)−m)=ℓabsentsuperscript𝑒superscript𝑚1superscript𝑚2superscriptℓ1rowsumsuperscript𝑒superscript𝐒2superscript𝑚2rowsumsuperscript𝑒superscript𝐒1𝑚rowsumsuperscript𝑒superscript𝐒2𝑚ℓ\\displaystyle=e^{m^{(1)}-m^{(2)}}\\ell^{(1)}+\\mathrm{rowsum}(e^{\\mathbf{S}^{(2)}-m^{(2)}})=\\mathrm{rowsum}(e^{\\mathbf{S}^{(1)}-m})+\\mathrm{rowsum}(e^{\\mathbf{S}^{(2)}-m})=\\ell |  |
|  | 𝐏~(2)superscript~𝐏2\\displaystyle\\tilde{\\mathbf{P}}^{(2)} | =diag​(ℓ(2))−1​e𝐒(2)−m(2)absentdiagsuperscriptsuperscriptℓ21superscript𝑒superscript𝐒2superscript𝑚2\\displaystyle=\\mathrm{diag}(\\ell^{(2)})^{-1}e^{\\mathbf{S}^{(2)}-m^{(2)}} |  |
|  | 𝐎~(2)superscript~𝐎2\\displaystyle\\tilde{\\mathbf{O}}^{(2)} | =diag​(em(1)−m(2))−1​𝐎~(1)+e𝐒(2)−m(2)​𝐕(2)=es(1)−m​𝐕(1)+es(2)−m​𝐕(2)absentdiagsuperscriptsuperscript𝑒superscript𝑚1superscript𝑚21superscript~𝐎1superscript𝑒superscript𝐒2superscript𝑚2superscript𝐕2superscript𝑒superscript𝑠1𝑚superscript𝐕1superscript𝑒superscript𝑠2𝑚superscript𝐕2\\displaystyle=\\mathrm{diag}(e^{m^{(1)}-m^{(2)}})^{-1}\\tilde{\\mathbf{O}}^{(1)}+e^{\\mathbf{S}^{(2)}-m^{(2)}}\\mathbf{V}^{(2)}=e^{s^{(1)}-m}\\mathbf{V}^{(1)}+e^{s^{(2)}-m}\\mathbf{V}^{(2)} |  |
|  | 𝐎(2)superscript𝐎2\\displaystyle\\mathbf{O}^{(2)} | =diag​(ℓ(2))−1​𝐎~(2)=𝐎.absentdiagsuperscriptsuperscriptℓ21superscript~𝐎2𝐎\\displaystyle=\\mathrm{diag}(\\ell^{(2)})^{-1}\\tilde{\\mathbf{O}}^{(2)}=\\mathbf{O}. |  |

We describe the full FlashAttention-2 forward pass in [Algorithm1](https://ar5iv.labs.arxiv.org/html/2307.08691#alg1 "In 3.1.1 Forward pass ‣ 3.1 Algorithm ‣ 3 FlashAttention-2: Algorithm, Parallelism, and Work Partitioning ‣ FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning").

Algorithm 1FlashAttention-2 forward pass

0:  Matrices 𝐐,𝐊,𝐕∈ℝN×d𝐐𝐊𝐕superscriptℝ𝑁𝑑\\mathbf{Q},\\mathbf{K},\\mathbf{V}\\in\\mathbb{R}^{N\\times d} in HBM, block sizes Bcsubscript𝐵𝑐B\_{c}, Brsubscript𝐵𝑟B\_{r}.

1:   Divide 𝐐𝐐\\mathbf{Q} into Tr=⌈NBr⌉subscript𝑇𝑟𝑁subscript𝐵𝑟T\_{r}=\\left\\lceil\\frac{N}{B\_{r}}\\right\\rceil blocks 𝐐1,…,𝐐Trsubscript𝐐1…subscript𝐐subscript𝑇𝑟\\mathbf{Q}\_{1},\\dots,\\mathbf{Q}\_{T\_{r}} of size Br×dsubscript𝐵𝑟𝑑B\_{r}\\times d each,
and divide 𝐊,𝐕𝐊𝐕\\mathbf{K},\\mathbf{V} in to Tc=⌈NBc⌉subscript𝑇𝑐𝑁subscript𝐵𝑐T\_{c}=\\left\\lceil\\frac{N}{B\_{c}}\\right\\rceil blocks 𝐊1,…,𝐊Tcsubscript𝐊1…subscript𝐊subscript𝑇𝑐\\mathbf{K}\_{1},\\dots,\\mathbf{K}\_{T\_{c}} and
𝐕1,…,𝐕Tcsubscript𝐕1…subscript𝐕subscript𝑇𝑐\\mathbf{V}\_{1},\\dots,\\mathbf{V}\_{T\_{c}}, of size Bc×dsubscript𝐵𝑐𝑑B\_{c}\\times d each.

2:  Divide the output 𝐎∈ℝN×d𝐎superscriptℝ𝑁𝑑\\mathbf{O}\\in\\mathbb{R}^{N\\times d} into Trsubscript𝑇𝑟T\_{r} blocks 𝐎i,…,𝐎Trsubscript𝐎𝑖…subscript𝐎subscript𝑇𝑟\\mathbf{O}\_{i},\\dots,\\mathbf{O}\_{T\_{r}} of size
Br×dsubscript𝐵𝑟𝑑B\_{r}\\times d each, and divide the logsumexp L𝐿L into Trsubscript𝑇𝑟T\_{r} blocks Li,…,LTrsubscript𝐿𝑖…subscript𝐿subscript𝑇𝑟L\_{i},\\dots,L\_{T\_{r}} of size
Brsubscript𝐵𝑟B\_{r} each.

3:for1≤i≤Tr1𝑖subscript𝑇𝑟1\\leq i\\leq T\_{r}do

4:      Load 𝐐isubscript𝐐𝑖\\mathbf{Q}\_{i} from HBM to on-chip SRAM.

5:      On chip, initialize 𝐎i(0)=(0)Br×d∈ℝBr×d,ℓi(0)=(0)Br∈ℝBr,mi(0)=(−∞)Br∈ℝBrformulae-sequencesuperscriptsubscript𝐎𝑖0subscript0subscript𝐵𝑟𝑑superscriptℝsubscript𝐵𝑟𝑑superscriptsubscriptℓ𝑖0subscript0subscript𝐵𝑟superscriptℝsubscript𝐵𝑟superscriptsubscript𝑚𝑖0subscriptsubscript𝐵𝑟superscriptℝsubscript𝐵𝑟\\mathbf{O}\_{i}^{(0)}=(0)\_{B\_{r}\\times d}\\in\\mathbb{R}^{B\_{r}\\times d},\\ell\_{i}^{(0)}=(0)\_{B\_{r}}\\in\\mathbb{R}^{B\_{r}},m\_{i}^{(0)}=(-\\infty)\_{B\_{r}}\\in\\mathbb{R}^{B\_{r}}.

6:for1≤j≤Tc1𝑗subscript𝑇𝑐1\\leq j\\leq T\_{c}do

7:         Load 𝐊j,𝐕jsubscript𝐊𝑗subscript𝐕𝑗\\mathbf{K}\_{j},\\mathbf{V}\_{j} from HBM to on-chip SRAM.

8:         On chip, compute 𝐒i(j)=𝐐i​𝐊jT∈ℝBr×Bcsuperscriptsubscript𝐒𝑖𝑗subscript𝐐𝑖superscriptsubscript𝐊𝑗𝑇superscriptℝsubscript𝐵𝑟subscript𝐵𝑐\\mathbf{S}\_{i}^{(j)}=\\mathbf{Q}\_{i}\\mathbf{K}\_{j}^{T}\\in\\mathbb{R}^{B\_{r}\\times B\_{c}}.

9:         On chip, compute
mi(j)=max​(mi(j−1),rowmax​(𝐒i(j)))∈ℝBrsuperscriptsubscript𝑚𝑖𝑗maxsuperscriptsubscript𝑚𝑖𝑗1rowmaxsuperscriptsubscript𝐒𝑖𝑗superscriptℝsubscript𝐵𝑟m\_{i}^{(j)}=\\mathrm{max}(m\_{i}^{(j-1)},\\mathrm{rowmax}(\\mathbf{S}\_{i}^{(j)}))\\in\\mathbb{R}^{B\_{r}}, 𝐏~i(j)=exp⁡(𝐒i(j)−mi(j))∈ℝBr×Bcsuperscriptsubscript~𝐏𝑖𝑗superscriptsubscript𝐒𝑖𝑗superscriptsubscript𝑚𝑖𝑗superscriptℝsubscript𝐵𝑟subscript𝐵𝑐\\tilde{\\mathbf{P}}\_{i}^{(j)}=\\exp(\\mathbf{S}\_{i}^{(j)}-m\_{i}^{(j)})\\in\\mathbb{R}^{B\_{r}\\times B\_{c}} (pointwise),
ℓi(j)=emij−1−mi(j)​ℓi(j−1)+rowsum​(𝐏~i(j))∈ℝBrsuperscriptsubscriptℓ𝑖𝑗superscript𝑒superscriptsubscript𝑚𝑖𝑗1superscriptsubscript𝑚𝑖𝑗superscriptsubscriptℓ𝑖𝑗1rowsumsuperscriptsubscript~𝐏𝑖𝑗superscriptℝsubscript𝐵𝑟\\ell\_{i}^{(j)}=e^{m\_{i}^{j-1}-m\_{i}^{(j)}}\\ell\_{i}^{(j-1)}+\\mathrm{rowsum}(\\tilde{\\mathbf{P}}\_{i}^{(j)})\\in\\mathbb{R}^{B\_{r}}.

10:         On chip, compute
𝐎i(j)=diag​(emi(j−1)−mi(j))−1​𝐎i(j−1)+𝐏~i(j)​𝐕jsuperscriptsubscript𝐎𝑖𝑗diagsuperscriptsuperscript𝑒superscriptsubscript𝑚𝑖𝑗1superscriptsubscript𝑚𝑖𝑗1superscriptsubscript𝐎𝑖𝑗1superscriptsubscript~𝐏𝑖𝑗subscript𝐕𝑗\\mathbf{O}\_{i}^{(j)}=\\mathrm{diag}(e^{m\_{i}^{(j-1)}-m\_{i}^{(j)}})^{-1}\\mathbf{O}\_{i}^{(j-1)}+\\tilde{\\mathbf{P}}\_{i}^{(j)}\\mathbf{V}\_{j}.

11:endfor

12:     On chip, compute 𝐎i=diag​(ℓi(Tc))−1​𝐎i(Tc)subscript𝐎𝑖diagsuperscriptsuperscriptsubscriptℓ𝑖subscript𝑇𝑐1superscriptsubscript𝐎𝑖subscript𝑇𝑐\\mathbf{O}\_{i}=\\mathrm{diag}(\\ell\_{i}^{(T\_{c})})^{-1}\\mathbf{O}\_{i}^{(T\_{c})}.

13:     On chip, compute Li=mi(Tc)+log⁡(ℓi(Tc))subscript𝐿𝑖superscriptsubscript𝑚𝑖subscript𝑇𝑐superscriptsubscriptℓ𝑖subscript𝑇𝑐L\_{i}=m\_{i}^{(T\_{c})}+\\log(\\ell\_{i}^{(T\_{c})}).

14:     Write 𝐎isubscript𝐎𝑖\\mathbf{O}\_{i} to HBM as the i𝑖i-th block of 𝐎𝐎\\mathbf{O}.

15:     Write Lisubscript𝐿𝑖L\_{i} to HBM as the i𝑖i-th block of L𝐿L.

16:endfor

17:  Return the output 𝐎𝐎\\mathbf{O} and the logsumexp L𝐿L.

##### Causal masking.

One common use case of attention is in auto-regressive language modeling, where
we need to apply a causal mask to the attention matrix 𝐒𝐒\\mathbf{S} (i.e., any entry
𝐒i​jsubscript𝐒𝑖𝑗\\mathbf{S}\_{ij} with j>i𝑗𝑖j>i is set to −∞-\\infty).

1. 1.


As FlashAttention and FlashAttention-2 already operate by blocks, for any blocks
where all the column indices are more than the row indices (approximately half
of the blocks for large sequence length), we can skip the computation of that
block.
This leads to around 1.7-1.8×\\times speedup compared to attention without the
causal mask.

2. 2.


We do not need to apply the causal mask for blocks whose row indices are
guaranteed to be strictly less than the column indices. This means that for
each row, we only need apply causal mask to 1 block (assuming square block).


##### Correctness, runtime, and memory requirement.

As with FlashAttention, [Algorithm1](https://ar5iv.labs.arxiv.org/html/2307.08691#alg1 "In 3.1.1 Forward pass ‣ 3.1 Algorithm ‣ 3 FlashAttention-2: Algorithm, Parallelism, and Work Partitioning ‣ FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning") returns the correct output
𝐎=softmax​(𝐐𝐊⊤)​𝐕𝐎softmaxsuperscript𝐐𝐊top𝐕\\mathbf{O}=\\mathrm{softmax}(\\mathbf{Q}\\mathbf{K}^{\\top})\\mathbf{V} (with no approximation), using O​(N2​d)𝑂superscript𝑁2𝑑O(N^{2}d) FLOPs and
requires O​(N)𝑂𝑁O(N) additional memory beyond inputs and output (to store the
logsumexp L𝐿L).
The proof is almost the same as the proof of
Dao et al. \[ [5](https://ar5iv.labs.arxiv.org/html/2307.08691#bib.bib5 ""), Theorem 1\], so we omit it here.

#### 3.1.2 Backward pass

The backward pass of FlashAttention-2 is almost the same as that of FlashAttention. We make
a minor tweak to only use the row-wise logsumexp L𝐿L instead of both the
row-wise max and row-wise sum of exponentials in the softmax.
We include the backward pass description in [Algorithm2](https://ar5iv.labs.arxiv.org/html/2307.08691#alg2 "In 3.1.2 Backward pass ‣ 3.1 Algorithm ‣ 3 FlashAttention-2: Algorithm, Parallelism, and Work Partitioning ‣ FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning") for completeness.

Algorithm 2FlashAttention-2 Backward Pass

0:  Matrices 𝐐,𝐊,𝐕,𝐎,𝐝𝐎∈ℝN×d𝐐𝐊𝐕𝐎𝐝𝐎superscriptℝ𝑁𝑑\\mathbf{Q},\\mathbf{K},\\mathbf{V},\\mathbf{O},\\mathbf{dO}\\in\\mathbb{R}^{N\\times d} in HBM,
vector L∈ℝN𝐿superscriptℝ𝑁L\\in\\mathbb{R}^{N} in HBM, block sizes Bcsubscript𝐵𝑐B\_{c}, Brsubscript𝐵𝑟B\_{r}.

1:  Divide 𝐐𝐐\\mathbf{Q} into Tr=⌈NBr⌉subscript𝑇𝑟𝑁subscript𝐵𝑟T\_{r}=\\left\\lceil\\frac{N}{B\_{r}}\\right\\rceil blocks 𝐐1,…,𝐐Trsubscript𝐐1…subscript𝐐subscript𝑇𝑟\\mathbf{Q}\_{1},\\dots,\\mathbf{Q}\_{T\_{r}} of size Br×dsubscript𝐵𝑟𝑑B\_{r}\\times d each,
and divide 𝐊,𝐕𝐊𝐕\\mathbf{K},\\mathbf{V} in to Tc=⌈NBc⌉subscript𝑇𝑐𝑁subscript𝐵𝑐T\_{c}=\\left\\lceil\\frac{N}{B\_{c}}\\right\\rceil blocks 𝐊1,…,𝐊Tcsubscript𝐊1…subscript𝐊subscript𝑇𝑐\\mathbf{K}\_{1},\\dots,\\mathbf{K}\_{T\_{c}} and
𝐕1,…,𝐕Tcsubscript𝐕1…subscript𝐕subscript𝑇𝑐\\mathbf{V}\_{1},\\dots,\\mathbf{V}\_{T\_{c}}, of size Bc×dsubscript𝐵𝑐𝑑B\_{c}\\times d each.

2:  Divide 𝐎𝐎\\mathbf{O} into Trsubscript𝑇𝑟T\_{r} blocks 𝐎i,…,𝐎Trsubscript𝐎𝑖…subscript𝐎subscript𝑇𝑟\\mathbf{O}\_{i},\\dots,\\mathbf{O}\_{T\_{r}} of size
Br×dsubscript𝐵𝑟𝑑B\_{r}\\times d each, divide 𝐝𝐎𝐝𝐎\\mathbf{dO} into Trsubscript𝑇𝑟T\_{r} blocks 𝐝𝐎i,…,𝐝𝐎Trsubscript𝐝𝐎𝑖…subscript𝐝𝐎subscript𝑇𝑟\\mathbf{dO}\_{i},\\dots,\\mathbf{dO}\_{T\_{r}}
of size Br×dsubscript𝐵𝑟𝑑B\_{r}\\times d each, and divide L𝐿L into Trsubscript𝑇𝑟T\_{r} blocks Li,…,LTrsubscript𝐿𝑖…subscript𝐿subscript𝑇𝑟L\_{i},\\dots,L\_{T\_{r}} of size
Brsubscript𝐵𝑟B\_{r} each.

3:  Initialize 𝐝𝐐=(0)N×d𝐝𝐐subscript0𝑁𝑑\\mathbf{dQ}=(0)\_{N\\times d} in HBM and divide it into Trsubscript𝑇𝑟T\_{r} blocks 𝐝𝐐1,…,𝐝𝐐Trsubscript𝐝𝐐1…subscript𝐝𝐐subscript𝑇𝑟\\mathbf{dQ}\_{1},\\dots,\\mathbf{dQ}\_{T\_{r}} of size Br×dsubscript𝐵𝑟𝑑B\_{r}\\times d each.
Divide 𝐝𝐊,𝐝𝐕∈ℝN×d𝐝𝐊𝐝𝐕superscriptℝ𝑁𝑑\\mathbf{dK},\\mathbf{dV}\\in\\mathbb{R}^{N\\times d} in to Tcsubscript𝑇𝑐T\_{c} blocks 𝐝𝐊1,…,𝐝𝐊Tcsubscript𝐝𝐊1…subscript𝐝𝐊subscript𝑇𝑐\\mathbf{dK}\_{1},\\dots,\\mathbf{dK}\_{T\_{c}} and
𝐝𝐕1,…,𝐝𝐕Tcsubscript𝐝𝐕1…subscript𝐝𝐕subscript𝑇𝑐\\mathbf{dV}\_{1},\\dots,\\mathbf{dV}\_{T\_{c}}, of size Bc×dsubscript𝐵𝑐𝑑B\_{c}\\times d each.

4:  Compute D=rowsum​(𝐝𝐎∘𝐎)∈ℝd𝐷rowsum𝐝𝐎𝐎superscriptℝ𝑑D=\\mathrm{rowsum}(\\mathbf{dO}\\circ\\mathbf{O})\\in\\mathbb{R}^{d} (pointwise multiply), write
D𝐷D to HBM and divide it into Trsubscript𝑇𝑟T\_{r} blocks D1,…,DTrsubscript𝐷1…subscript𝐷subscript𝑇𝑟D\_{1},\\dots,D\_{T\_{r}} of size
Brsubscript𝐵𝑟B\_{r} each.

5:for1≤j≤Tc1𝑗subscript𝑇𝑐1\\leq j\\leq T\_{c}do

6:     Load 𝐊j,𝐕jsubscript𝐊𝑗subscript𝐕𝑗\\mathbf{K}\_{j},\\mathbf{V}\_{j} from HBM to on-chip SRAM.

7:     Initialize 𝐝𝐊j=(0)Bc×d,𝐝𝐕j=(0)Bc×dformulae-sequencesubscript𝐝𝐊𝑗subscript0subscript𝐵𝑐𝑑subscript𝐝𝐕𝑗subscript0subscript𝐵𝑐𝑑\\mathbf{dK}\_{j}=(0)\_{B\_{c}\\times d},\\mathbf{dV}\_{j}=(0)\_{B\_{c}\\times d} on SRAM.

8:for1≤i≤Tr1𝑖subscript𝑇𝑟1\\leq i\\leq T\_{r}do

9:        Load 𝐐i,𝐎i,𝐝𝐎i,𝐝𝐐i,Li,Disubscript𝐐𝑖subscript𝐎𝑖subscript𝐝𝐎𝑖subscript𝐝𝐐𝑖subscript𝐿𝑖subscript𝐷𝑖\\mathbf{Q}\_{i},\\mathbf{O}\_{i},\\mathbf{dO}\_{i},\\mathbf{dQ}\_{i},L\_{i},D\_{i} from HBM to on-chip SRAM.

10:        On chip, compute 𝐒i(j)=𝐐i​𝐊jT∈ℝBr×Bcsuperscriptsubscript𝐒𝑖𝑗subscript𝐐𝑖superscriptsubscript𝐊𝑗𝑇superscriptℝsubscript𝐵𝑟subscript𝐵𝑐\\mathbf{S}\_{i}^{(j)}=\\mathbf{Q}\_{i}\\mathbf{K}\_{j}^{T}\\in\\mathbb{R}^{B\_{r}\\times B\_{c}}.

11:        On chip, compute 𝐏i(j)=exp⁡(𝐒i​j−Li)∈ℝBr×Bcsuperscriptsubscript𝐏𝑖𝑗subscript𝐒𝑖𝑗subscript𝐿𝑖superscriptℝsubscript𝐵𝑟subscript𝐵𝑐\\mathbf{P}\_{i}^{(j)}=\\exp(\\mathbf{S}\_{ij}-L\_{i})\\in\\mathbb{R}^{B\_{r}\\times B\_{c}}.

12:        On chip, compute
𝐝𝐕j←𝐝𝐕j+(𝐏i(j))⊤​𝐝𝐎i∈ℝBc×d←subscript𝐝𝐕𝑗subscript𝐝𝐕𝑗superscriptsuperscriptsubscript𝐏𝑖𝑗topsubscript𝐝𝐎𝑖superscriptℝsubscript𝐵𝑐𝑑\\mathbf{dV}\_{j}\\leftarrow\\mathbf{dV}\_{j}+(\\mathbf{P}\_{i}^{(j)})^{\\top}\\mathbf{dO}\_{i}\\in\\mathbb{R}^{B\_{c}\\times d}.

13:        On chip, compute
𝐝𝐏i(j)=𝐝𝐎i​𝐕j⊤∈ℝBr×Bcsuperscriptsubscript𝐝𝐏𝑖𝑗subscript𝐝𝐎𝑖superscriptsubscript𝐕𝑗topsuperscriptℝsubscript𝐵𝑟subscript𝐵𝑐\\mathbf{dP}\_{i}^{(j)}=\\mathbf{dO}\_{i}\\mathbf{V}\_{j}^{\\top}\\in\\mathbb{R}^{B\_{r}\\times B\_{c}}.

14:        On chip, compute 𝐝𝐒i(j)=𝐏i(j)∘(𝐝𝐏i(j)−Di)∈ℝBr×Bcsuperscriptsubscript𝐝𝐒𝑖𝑗superscriptsubscript𝐏𝑖𝑗superscriptsubscript𝐝𝐏𝑖𝑗subscript𝐷𝑖superscriptℝsubscript𝐵𝑟subscript𝐵𝑐\\mathbf{dS}\_{i}^{(j)}=\\mathbf{P}\_{i}^{(j)}\\circ(\\mathbf{dP}\_{i}^{(j)}-D\_{i})\\in\\mathbb{R}^{B\_{r}\\times B\_{c}}.

15:        Load 𝐝𝐐isubscript𝐝𝐐𝑖\\mathbf{dQ}\_{i} from HBM to SRAM, then on chip, update
𝐝𝐐i←𝐝𝐐i+𝐝𝐒i(j)​𝐊j∈ℝBr×d←subscript𝐝𝐐𝑖subscript𝐝𝐐𝑖superscriptsubscript𝐝𝐒𝑖𝑗subscript𝐊𝑗superscriptℝsubscript𝐵𝑟𝑑\\mathbf{dQ}\_{i}\\leftarrow\\mathbf{dQ}\_{i}+\\mathbf{dS}\_{i}^{(j)}\\mathbf{K}\_{j}\\in\\mathbb{R}^{B\_{r}\\times d}, and write
back to HBM.

16:        On chip, compute 𝐝𝐊j←𝐝𝐊j+𝐝𝐒i(j)⊤​𝐐i∈ℝBc×d←subscript𝐝𝐊𝑗subscript𝐝𝐊𝑗superscriptsuperscriptsubscript𝐝𝐒𝑖𝑗topsubscript𝐐𝑖superscriptℝsubscript𝐵𝑐𝑑\\mathbf{dK}\_{j}\\leftarrow\\mathbf{dK}\_{j}+{\\mathbf{dS}\_{i}^{(j)}}^{\\top}\\mathbf{Q}\_{i}\\in\\mathbb{R}^{B\_{c}\\times d}.

17:endfor

18:     Write 𝐝𝐊j,𝐝𝐕jsubscript𝐝𝐊𝑗subscript𝐝𝐕𝑗\\mathbf{dK}\_{j},\\mathbf{dV}\_{j} to HBM.

19:endfor

20:  Return 𝐝𝐐,𝐝𝐊,𝐝𝐕𝐝𝐐𝐝𝐊𝐝𝐕\\mathbf{dQ},\\mathbf{dK},\\mathbf{dV}.

##### Multi-query attention and grouped-query attention.

Multi-query attention (MQA) \[ [15](https://ar5iv.labs.arxiv.org/html/2307.08691#bib.bib15 "")\] and grouped-query attention
(GQA) \[ [1](https://ar5iv.labs.arxiv.org/html/2307.08691#bib.bib1 "")\] are variants of attention where multiple heads of
query attend to the same head of key and value, in order to reduce the size of
KV cache during inference.
Instead of having to duplicate the key and value heads for the computation, we
implicitly manipulate the indices into the head to perform the same computation.
In the backward pass, we need to sum the gradients 𝐝𝐊𝐝𝐊\\mathbf{dK} and 𝐝𝐕𝐝𝐕\\mathbf{dV} across
different heads that were implicitly duplicated.

### 3.2 Parallelism

The first version of FlashAttention parallelizes over batch size and number of
heads.
We use 1 thread block to process one attention head, and there are overall
batch size⋅number of heads⋅batch sizenumber of heads\\text{batch size}\\cdot\\text{number of heads} thread blocks.
Each thread block is scheduled to run on a streaming multiprocessor (SM), and
there are 108 of these SMs on an A100 GPU for example.
This scheduling is efficient when this number is large (say ≥80absent80\\geq 80), since we
can effectively use almost all of the compute resources on the GPU.

In the case of long sequences (which usually means small batch sizes or small
number of heads), to make better use of the multiprocessors on the GPU, we now
additionally parallelize over the sequence length dimension.
This results in significant speedup for this regime.

##### Forward pass.

We see that the outer loop (over sequence length) is embarrassingly parallel,
and we schedule them on different thread blocks that do not need to communicate
with each other.
We also parallelize over the batch dimension and number of heads dimension, as
done in FlashAttention.
The increased parallelism over sequence length helps improve occupancy (fraction
of GPU resources being used) when the batch size and number of heads are small,
leading to speedup in this case.

These ideas of swapping the order of the loop (outer loop over row blocks and
inner loop over column blocks, instead of the other way round in the original
FlashAttention paper), as well as parallelizing over the sequence length dimension
were first suggested and implemented by Phil Tillet in the
Triton \[ [17](https://ar5iv.labs.arxiv.org/html/2307.08691#bib.bib17 "")\]
implementation.333 [https://github.com/openai/triton/blob/main/python/tutorials/06-fused-attention.py](https://github.com/openai/triton/blob/main/python/tutorials/06-fused-attention.py "")

##### Backward pass.

Notice that the only shared computation between different column blocks is in
update 𝐝𝐐𝐝𝐐\\mathbf{dQ} in [Algorithm2](https://ar5iv.labs.arxiv.org/html/2307.08691#alg2 "In 3.1.2 Backward pass ‣ 3.1 Algorithm ‣ 3 FlashAttention-2: Algorithm, Parallelism, and Work Partitioning ‣ FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning"), where we need to load 𝐝𝐐isubscript𝐝𝐐𝑖\\mathbf{dQ}\_{i} from HBM
to SRAM, then on chip, update
𝐝𝐐i←𝐝𝐐i+𝐝𝐒i(j)​𝐊j←subscript𝐝𝐐𝑖subscript𝐝𝐐𝑖superscriptsubscript𝐝𝐒𝑖𝑗subscript𝐊𝑗\\mathbf{dQ}\_{i}\\leftarrow\\mathbf{dQ}\_{i}+\\mathbf{dS}\_{i}^{(j)}\\mathbf{K}\_{j}, and write back to HBM.
We thus parallelize over the sequence length dimension as well, and schedule 1
thread block for each column block of the backward pass.
We use atomic adds to communicate between different thread blocks to update 𝐝𝐐𝐝𝐐\\mathbf{dQ}.

We describe the parallelization scheme in [Fig.2](https://ar5iv.labs.arxiv.org/html/2307.08691#S3.F2 "In Backward pass. ‣ 3.2 Parallelism ‣ 3 FlashAttention-2: Algorithm, Parallelism, and Work Partitioning ‣ FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning").

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2307.08691/assets/figs/flashattention_fwd_bwd_parallel.png)Figure 2: In the forward pass (left), we parallelize the
workers (thread blocks) where each worker takes care of a block of rows of
the attention matrix.
In the backward pass (right), each worker takes care of a block of columns
of the attention matrix.

### 3.3 Work Partitioning Between Warps

As [Section3.2](https://ar5iv.labs.arxiv.org/html/2307.08691#S3.SS2 "3.2 Parallelism ‣ 3 FlashAttention-2: Algorithm, Parallelism, and Work Partitioning ‣ FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning") describe how we schedule thread blocks, even within
each thread block, we also have to decide how to partition the work between
different warps.
We typically use 4 or 8 warps per thread block, and the partitioning is described
in [Fig.3](https://ar5iv.labs.arxiv.org/html/2307.08691#S3.F3 "In Forward pass. ‣ 3.3 Work Partitioning Between Warps ‣ 3 FlashAttention-2: Algorithm, Parallelism, and Work Partitioning ‣ FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning").

##### Forward pass.

For each block, FlashAttention splits 𝐊𝐊\\mathbf{K} and 𝐕𝐕\\mathbf{V} across 4 warps while keeping
𝐐𝐐\\mathbf{Q} accessible by all warps.
Each warp multiplies to get a slice of 𝐐𝐊⊤superscript𝐐𝐊top\\mathbf{Q}\\mathbf{K}^{\\top}, then they need to multiply
with a slice of 𝐕𝐕\\mathbf{V} and communicate to add up the result.
This is referred to as the “split-K” scheme.
However, this is inefficient since all warps need to write their intermediate
results out to shared memory, synchronize, then add up the intermediate results.
These shared memory reads/writes slow down the forward pass in FlashAttention.

In FlashAttention-2, we instead split 𝐐𝐐\\mathbf{Q} across 4 warps while keeping 𝐊𝐊\\mathbf{K} and 𝐕𝐕\\mathbf{V}
accessible by all warps.
After each warp performs matrix multiply to get a slice of 𝐐𝐊⊤superscript𝐐𝐊top\\mathbf{Q}\\mathbf{K}^{\\top}, they just need to
multiply with their shared slice of 𝐕𝐕\\mathbf{V} to get their corresponding
slice of the output.
There is no need for communication between warps.
The reduction in shared memory reads/writes yields speedup ( [Section4](https://ar5iv.labs.arxiv.org/html/2307.08691#S4 "4 Empirical Validation ‣ FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning")).

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2307.08691/assets/figs/flash_partitioning.png)(a)FlashAttention

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2307.08691/assets/figs/flash2_partitioning.png)(b)FlashAttention-2

Figure 3: Work partitioning between different warps in the forward pass

##### Backward pass.

Similarly for the backward pass, we choose to partition the warps to avoid the
“split-K” scheme.
However, it still requires some synchronization due to the more complicated
dependency between all the different inputs and gradients
𝐐,𝐊,𝐕,𝐎,𝐝𝐎,𝐝𝐐,𝐝𝐊,𝐝𝐕𝐐𝐊𝐕𝐎𝐝𝐎𝐝𝐐𝐝𝐊𝐝𝐕\\mathbf{Q},\\mathbf{K},\\mathbf{V},\\mathbf{O},\\mathbf{dO},\\mathbf{dQ},\\mathbf{dK},\\mathbf{dV}.
Nevertheless, avoiding “split-K” reduces shared memory reads/writes and again
yields speedup ( [Section4](https://ar5iv.labs.arxiv.org/html/2307.08691#S4 "4 Empirical Validation ‣ FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning")).

##### Tuning block sizes

Increasing block sizes generally reduces shared memory loads/stores, but
increases the number of registers required and the total amount of shared
memory.
Past a certain block size, register spilling causes significant slowdown, or the
amount of shared memory required is larger than what the GPU has available, and
the kernel cannot run at all.
Typically we choose blocks of size {64,128}×{64,128}6412864128\\{64,128\\}\\times\\{64,128\\}, depending on the
head dimension d𝑑d and the device shared memory size.

We manually tune for each head dimensions since there are essentially only 4
choices for block sizes, but this could benefit from auto-tuning to avoid this
manual labor.
We leave this to future work.

## 4 Empirical Validation

We evaluate the impact of using FlashAttention-2 to train Transformer models.

- •


Benchmarking attention.
We measure the runtime of FlashAttention-2 across different sequence lengths and
compare it to a standard implementation in PyTorch, FlashAttention, and
FlashAttention in Triton.
We confirm that FlashAttention-2 is 1.7-3.0×\\times faster than FlashAttention, 1.3-2.5×\\times
faster than FlashAttention in Triton, and 3-10×\\times faster than a standard
attention implementation.
FlashAttention-2 reaches up to 230 TFLOPs/s, 73% of the theoretical maximum TFLOPs/s
on A100 GPUs.

- •


End-to-end training speed
When used end-to-end to train GPT-style models of size 1.3B and 2.7B on
sequence lengths either 2k or 8k, FlashAttention-2 yields up to 1.3×\\times speedup compared to
FlashAttention and 2.8×\\times speedup compared to a baseline without FlashAttention.
FlashAttention-2 reaches up to 225 TFLOPs/s (72% model FLOPs utilization) per A100 GPU.


### 4.1 Benchmarking Attention

We measure the runtime of different attention methods on an A100 80GB SXM4 GPU
for different settings (without / with causal mask, head dimension 64 or 128).
We report the results
in [Fig.4](https://ar5iv.labs.arxiv.org/html/2307.08691#S4.F4 "In 4.1 Benchmarking Attention ‣ 4 Empirical Validation ‣ FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning"), [Fig.5](https://ar5iv.labs.arxiv.org/html/2307.08691#S4.F5 "In 4.1 Benchmarking Attention ‣ 4 Empirical Validation ‣ FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning")
and [Fig.6](https://ar5iv.labs.arxiv.org/html/2307.08691#S4.F6 "In 4.1 Benchmarking Attention ‣ 4 Empirical Validation ‣ FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning"), showing that FlashAttention-2 is around 2×\\times faster
than FlashAttention and FlashAttention in xformers (the “cutlass”
implementation).
FlashAttention-2 is around 1.3-1.5×\\times faster than FlashAttention in Triton in the forward
pass and around 2×\\times faster in the backward pass.
Compared to a standard attention implementation in PyTorch, FlashAttention-2 can be up
to 10×\\times faster.

Benchmark setting: we vary the sequence length from 512, 1k, …, 16k, and set
batch size so that the total number of tokens is 16k.
We set hidden dimension to 2048, and head dimension to be either 64 or 128
(i.e., 32 heads or 16 heads).
To calculate the FLOPs of the forward pass, we use:

|     |     |     |
| --- | --- | --- |
|  | 4⋅seqlen2⋅head dimension⋅number of heads.⋅4superscriptseqlen2head dimensionnumber of heads4\\cdot\\text{seqlen}^{2}\\cdot\\text{head dimension}\\cdot\\text{number of heads}. |  |

With causal mask, we divide this number by 2 to account for the fact that
approximately only half of the entries are calculated.
To get the FLOPs of the backward pass, we multiply the forward pass FLOPs by 2.5
(since there are 2 matmuls in the forward pass and 5 matmuls in the backward
pass, due to recomputation).

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2307.08691/assets/x1.png)(a)Without causal mask, head dimension 64

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2307.08691/assets/x2.png)(b)Without causal mask, head dimension 128

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2307.08691/assets/x3.png)(c)With causal mask, head dimension 64

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2307.08691/assets/x4.png)(d)With causal mask, head dimension 128

Figure 4: Attention forward + backward speed on A100 GPU

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2307.08691/assets/x5.png)(a)Without causal mask, head dimension 64

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2307.08691/assets/x6.png)(b)Without causal mask, head dimension 128

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2307.08691/assets/x7.png)(c)With causal mask, head dimension 64

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2307.08691/assets/x8.png)(d)With causal mask, head dimension 128

Figure 5: Attention forward speed on A100 GPU

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2307.08691/assets/x9.png)(a)Without causal mask, head dimension 64

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2307.08691/assets/x10.png)(b)Without causal mask, head dimension 128

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2307.08691/assets/x11.png)(c)With causal mask, head dimension 64

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2307.08691/assets/x12.png)(d)With causal mask, head dimension 128

Figure 6: Attention backward speed on A100 GPU

Just running the same implementation on H100 GPUs (using no special instructions
to make use of new features such as TMA and 4th-gen Tensor Cores), we obtain up
to 335 TFLOPs/s ( [Fig.7](https://ar5iv.labs.arxiv.org/html/2307.08691#S4.F7 "In 4.1 Benchmarking Attention ‣ 4 Empirical Validation ‣ FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning")).
We expect that by using new instructions, we can obtain another 1.5x-2x speedup
on H100 GPUs. We leave that to future work.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2307.08691/assets/x13.png)(a)Without causal mask, head dimension 64

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2307.08691/assets/x14.png)(b)Without causal mask, head dimension 128

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2307.08691/assets/x15.png)(c)With causal mask, head dimension 64

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2307.08691/assets/x16.png)(d)With causal mask, head dimension 128

Figure 7: Attention forward + backward speed on H100 GPU

### 4.2 End-to-end Performance

We measure the training throughput of GPT-style models with either 1.3B or 2.7B
parameters, on 8×\\timesA100 80GB SXM.
As shown in [Table1](https://ar5iv.labs.arxiv.org/html/2307.08691#S4.T1 "In 4.2 End-to-end Performance ‣ 4 Empirical Validation ‣ FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning"), FlashAttention-2 yields 2.8×\\times speedup compared to
a baseline without FlashAttention and 1.3×\\times speedup compared to FlashAttention-2, reaching up to 225 TFLOPs/s
per A100 GPU.

Note that we calculate the FLOPs by the formula, following Megatron-LM \[ [16](https://ar5iv.labs.arxiv.org/html/2307.08691#bib.bib16 "")\] (and many
other papers and libraries):

|     |     |     |
| --- | --- | --- |
|  | 6⋅seqlen⋅number of params+12⋅number of<br>layers⋅hidden dim⋅seqlen2.⋅6seqlennumber of params⋅12number of<br>layershidden dimsuperscriptseqlen26\\cdot\\text{seqlen}\\cdot\\text{number of params}+12\\cdot\\text{number of<br>layers}\\cdot\\text{hidden dim}\\cdot\\text{seqlen}^{2}. |  |

The first term accounts for the FLOPs due to weight–input multiplication, and
the second term accounts for the FLOPs due to attention.
However, one can argue that the second term should be halved, as with causal
mask we only need to compute approximately half the number of elements in
attention.
We choose to follow the formula from the literature (without dividing the
attention FLOPs by 2) for consistency.

Table 1: Training speed (TFLOPs/s/GPU) of GPT-style
models on 8×\\timesA100 GPUs.
FlashAttention-2 reaches up to 225 TFLOPs/s (72% model FLOPs utilization).
We compare against a baseline running without FlashAttention.

| Model | Without FlashAttention | FlashAttention | FlashAttention-2 |
| --- | --- | --- | --- |
| GPT3-1.3B 2k context | 142 TFLOPs/s | 189 TFLOPs/s | 196 TFLOPs/s |
| GPT3-1.3B 8k context | 72 TFLOPS/s | 170 TFLOPs/s | 220 TFLOPs/s |
| GPT3-2.7B 2k context | 149 TFLOPs/s | 189 TFLOPs/s | 205 TFLOPs/s |
| GPT3-2.7B 8k context | 80 TFLOPs/s | 175 TFLOPs/s | 225 TFLOPs/s |

## 5 Discussion and Future Directions

FlashAttention-2 is 2×\\times faster than FlashAttention, which means that we can train models
with 16k longer context for the same price as previously training a 8k context
model.
We are excited about how this can be used to understand long books and reports,
high resolution images, audio and video.
FlashAttention-2 will also speed up training, finetuning, and inference of
existing models.

In the near future, we plan to collaborate with researchers and engineers to
make FlashAttention widely applicable in different kinds of devices (e.g., H100
GPUs, AMD GPUs), as well as new data types such as FP8.
As an immediate next step, we plan to optimize FlashAttention-2 for H100 GPUs to
use new hardware features (TMA, 4th-gen Tensor Cores, fp8).
Combining the low-level optimizations in FlashAttention-2 with high-level
algorithmic changes (e.g., local, dilated, block-sparse attention) could allow
us to train AI models with much longer context.
We are also excited to work with compiler researchers to make these optimization
techniques easily programmable.

#### Acknowledgments

We thank Phil Tillet and Daniel Haziza, who have implemented versions of
FlashAttention in Triton \[ [17](https://ar5iv.labs.arxiv.org/html/2307.08691#bib.bib17 "")\] and the xformers
library \[ [10](https://ar5iv.labs.arxiv.org/html/2307.08691#bib.bib10 "")\].
FlashAttention-2 was motivated by exchange of ideas between different ways that
attention could be implemented.
We are grateful to the Nvidia CUTLASS team (especially Vijay Thakkar, Cris Cecka, Haicheng
Wu, and Andrew Kerr) for their CUTLASS library, in particular the CUTLASS 3.x
release, which provides clean abstractions and powerful building blocks for the
implementation of FlashAttention-2.
We thank Driss Guessous for integrating FlashAttention to PyTorch.
FlashAttention-2 has benefited from helpful discussions with Phil Wang, Markus Rabe,
James Bradbury, Young-Jun Ko, Julien Launay, Daniel Hesslow, Michaël
Benesty, Horace He, Ashish Vaswani, and Erich Elsen.
Thanks for Stanford CRFM and Stanford NLP for the compute support.
We thank Dan Fu and Christopher Ré for their collaboration, constructive
feedback, and constant encouragement on this line of work of designing
hardware-efficient algorithms.
We thank Albert Gu and Beidi Chen for their helpful suggestions on early drafts
of this technical report.

## References

- Ainslie et al. \[2023\]
Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico
Lebrón, and Sumit Sanghai.

Gqa: Training generalized multi-query transformer models from
multi-head checkpoints.

_arXiv preprint arXiv:2305.13245_, 2023.

- Beltagy et al. \[2020\]
Iz Beltagy, Matthew E Peters, and Arman Cohan.

Longformer: The long-document transformer.

_arXiv preprint arXiv:2004.05150_, 2020.

- Chen et al. \[2021\]
Beidi Chen, Tri Dao, Eric Winsor, Zhao Song, Atri Rudra, and Christopher Ré.

Scatterbrain: Unifying sparse and low-rank attention.

In _Advances in Neural Information Processing Systems_
_(NeurIPS)_, 2021.

- Choromanski et al. \[2020\]
Krzysztof Marcin Choromanski, Valerii Likhosherstov, David Dohan, Xingyou Song,
Andreea Gane, Tamas Sarlos, Peter Hawkins, Jared Quincy Davis, Afroz
Mohiuddin, Lukasz Kaiser, et al.

Rethinking attention with performers.

In _International Conference on Learning Representations_
_(ICLR)_, 2020.

- Dao et al. \[2022\]
Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, and Christopher Ré.

FlashAttention: Fast and memory-efficient exact attention with
IO-awareness.

In _Advances in Neural Information Processing Systems_, 2022.

- Jia and Van Sandt \[2021\]
Zhe Jia and Peter Van Sandt.

Dissecting the Ampere GPU architecture via microbenchmarking.

GPU Technology Conference, 2021.

- Jia et al. \[2018\]
Zhe Jia, Marco Maggioni, Benjamin Staiger, and Daniele P Scarpazza.

Dissecting the nvidia Volta GPU architecture via
microbenchmarking.

_arXiv preprint arXiv:1804.06826_, 2018.

- Katharopoulos et al. \[2020\]
Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, and François
Fleuret.

Transformers are RNNs: Fast autoregressive transformers with linear
attention.

In _International Conference on Machine Learning_, pages
5156–5165. PMLR, 2020.

- Kitaev et al. \[2020\]
Nikita Kitaev, Łukasz Kaiser, and Anselm Levskaya.

Reformer: The efficient transformer.

In _The International Conference on Machine Learning (ICML)_,
2020.

- Lefaudeux et al. \[2022\]
Benjamin Lefaudeux, Francisco Massa, Diana Liskovich, Wenhan Xiong, Vittorio
Caggiano, Sean Naren, Min Xu, Jieru Hu, Marta Tintore, Susan Zhang, Patrick
Labatut, and Daniel Haziza.

xformers: A modular and hackable transformer modelling library.

[https://github.com/facebookresearch/xformers](https://github.com/facebookresearch/xformers ""), 2022.

- Milakov and Gimelshein \[2018\]
Maxim Milakov and Natalia Gimelshein.

Online normalizer calculation for softmax.

_arXiv preprint arXiv:1805.02867_, 2018.

- OpenAI \[2023\]
OpenAI.

Gpt-4 technical report.

_ArXiv_, abs/2303.08774, 2023.

- Rabe and Staats \[2021\]
Markus N Rabe and Charles Staats.

Self-attention does not need O​(n2)𝑂superscript𝑛2{O}(n^{2}) memory.

_arXiv preprint arXiv:2112.05682_, 2021.

- Roy et al. \[2021\]
Aurko Roy, Mohammad Saffar, Ashish Vaswani, and David Grangier.

Efficient content-based sparse attention with routing transformers.

_Transactions of the Association for Computational Linguistics_,
9:53–68, 2021.

- Shazeer \[2019\]
Noam Shazeer.

Fast transformer decoding: One write-head is all you need.

_arXiv preprint arXiv:1911.02150_, 2019.

- Shoeybi et al. \[2019\]
Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper,
and Bryan Catanzaro.

Megatron-LM: Training multi-billion parameter language models using
model parallelism.

_arXiv preprint arXiv:1909.08053_, 2019.

- Tillet et al. \[2019\]
Philippe Tillet, Hsiang-Tsung Kung, and David Cox.

Triton: an intermediate language and compiler for tiled neural
network computations.

In _Proceedings of the 3rd ACM SIGPLAN International Workshop on_
_Machine Learning and Programming Languages_, pages 10–19, 2019.

- Vaswani et al. \[2017\]
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,
Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin.

Attention is all you need.

_Advances in neural information processing systems_, 30, 2017.

- Wang et al. \[2020\]
Sinong Wang, Belinda Z Li, Madian Khabsa, Han Fang, and Hao Ma.

Linformer: Self-attention with linear complexity.

_arXiv preprint arXiv:2006.04768_, 2020.

- Zaheer et al. \[2020\]
Manzil Zaheer, Guru Guruganesh, Kumar Avinava Dubey, Joshua Ainslie, Chris
Alberti, Santiago Ontanon, Philip Pham, Anirudh Ravula, Qifan Wang, Li Yang,
et al.

Big bird: Transformers for longer sequences.

_Advances in Neural Information Processing Systems_, 33, 2020.


[◄](https://ar5iv.labs.arxiv.org/html/2307.08690) [![ar5iv homepage](https://ar5iv.labs.arxiv.org/assets/ar5iv.png)](https://ar5iv.labs.arxiv.org/) [Feeling\\
\\
lucky?](https://ar5iv.labs.arxiv.org/feeling_lucky) [Conversion\\
\\
report](https://ar5iv.labs.arxiv.org/log/2307.08691) [Report\\
\\
an issue](https://github.com/dginev/ar5iv/issues/new?template=improve-article--arxiv-id-.md&title=Improve+article+2307.08691) [View original\\
\\
on arXiv](https://arxiv.org/abs/2307.08691) [►](https://ar5iv.labs.arxiv.org/html/2307.08692)