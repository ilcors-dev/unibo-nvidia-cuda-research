# 22_CUDA

> https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html

## Why GPUs?
- GPUs are designed to have a high throughput for parallel workloads.
- While the CPU is designed to excel at executing a sequence of operations, called a thread, as fast as possible and can execute a few tens of these threads in parallel, the GPU is designed to excel at executing thousands of them in parallel (amortizing the slower single-thread performance to achieve greater throughput).
- The GPU is specialized for highly parallel computations and therefore designed such that more transistors are devoted to data processing rather than data caching and flow control

<img src="../images/Screenshot 2024-04-29 at 19.31.21.png" alt="CPU vs GPU" />

- More FP computations -> higher throughput for parallel workloads
- GPU can hide memory access latencies with computation, instead of relying on large data caches and complex flow control to avoid long memory access latencies, both of which are expensive in terms of transistors.
- The GPU and CPU are designed to work together to provide the best performance for a wide range of applications.
- Applications are a mix of serial and parallel workloads

## CUDA (Compute Unified Device Architecture)
- General purpose parallel computing platform and programming model that leverages the parallel compute engine in NVIDIA GPUs to solve complex computational problems.
- C/C++ extension

### CUDA Programming Model
- Since is an extension of C/C++, it is easy to learn
- At its core it exposes the following to programmers:
    - Hierarchical thread hierarchy
    - Shared memory
    - Barrier synchronization
- The abstractions provide fine-grained data parallelism and thread parallelism, nested within coarse-grained data parallelism and task parallelism.
- The programming model guides the programmer to partition the problem into coarse sub-problems that can be solved independently in parallel by blocks of threads, and each sub-problem into finer pieces that can solved cooperatively by the threads in the block.

<img src="../images/Screenshot 2024-04-29 at 19.39.28.png" alt="CUDA Programming Model" />

#### Kernels
- Kernels are simply functions that are executed on the GPU.
- When called are executed N times in parallel by N different CUDA threads, as opposed to only once like a regular C/C++ function.
- Each thread that exectues the kernel is given a unique threadID that is accessible within the kernel code.

```cpp
// Kernel definition
__global__ void VecAdd(float* A, float* B, float* C)
{
    int i = threadIdx.x;
    C[i] = A[i] + B[i];
}

int main()
{
    ...
    // Kernel invocation with N threads
    VecAdd<<<1, N>>>(A, B, C);
    ...
}
```
> As an illustration, the following sample code, using the built-in variable threadIdx, adds two vectors A and B of size N and stores the result into vector C:
>
> A kernel is defined using the __global__ declaration specifier and the number of CUDA threads that execute that kernel for a given kernel call is specified using a new <<<...>>>execution configuration syntax

#### Thread Hierarchy

```cpp
__global__ void MatAdd(float A[N][N], float B[N][N],
                       float C[N][N])
{
    int i = threadIdx.x;
    int j = threadIdx.y;
    C[i][j] = A[i][j] + B[i][j];
}

int main()
{
    ...
    // Kernel invocation with one block of N * N * 1 threads
    int numBlocks = 1;
    dim3 threadsPerBlock(N, N);
    MatAdd<<<numBlocks, threadsPerBlock>>>(A, B, C);
    ...
}
```

- The index of a thread and its thread ID relate to each other in a straightforward way:
    - one-dimensional blocks are laid out in a single line
    - two-dimensional blocks are laid out in a grid (Dx, Dy), where the thread ID of a thread of index `(x, y)` is `x + y * Dx`
    - three-dimensional blocks are laid out in a 3D grid (Dx, Dy, Dz), where the thread ID of a thread of index `(x, y, z)` is `x + y * Dx + z * Dx * Dy`
- There's a limit to the number of threads per block, all threads of a block are executed on the same SM, and all threads of a block can communicate with each other through shared memory.
- A thread block may contain up to 1024 threads
- A kernel can be executed by multiple equally-sized thread blocks

<img src="../images/grid-of-thread-blocks.png" alt="Thread Hierarchy" />

- Thread blocks are required to execute independently: It must be possible to execute them in any order, in parallel or in series.

- Threads within a block can cooperate through shared memory, and threads within a grid can cooperate through global memory.
- The synchronization barrier `__syncthreads()` can be used to synchronize threads within the same block.
- When using the function, all threads within the block must reach the same point in the code before any is allowed to proceed.

#### Memory Hierarchy
- Each thread has a private local memory
- Each thread block has shared memory visible to all threads of the block and with the same lifetime as the block
- All threads have access to the same global memory

<img src="../images/memory-hierarchy (1).png" alt="Memory Hierarchy" />

#### Heterogeneous Programming
- Modern applications are a mix of serial and parallel workloads
- The CPU is good at serial workloads
- The GPU is good at parallel workloads
- Each retain their own separate memory spaces in DRAM, referred to as the host memory and device memory, respectively

<img src="../images/heterogeneous-programming.png" alt="Heterogeneous Programming" />

> Serial code executes on the host (CPU) and parallel code executes on the device (GPU)

##### Unified memory
- Introduced in CUDA 6
- Allows the CPU and GPU to access the same virtual memory space
- Simplifies memory management and reduces the need to explicitly copy data between the CPU and GPU
- Basically a single pointer value enables all processors in the system (all CPUs, all GPUs) to access this memory with all of their native memory access instructions

- Pros:
    - Simplifies memory management
    - **Productivity**: Easier to write correct code. GPU programs may access Unified Memory from GPU and CPU threads concurrently without needing to create separate allocations (`cudaMalloc()`) and copy memory manually back and forth (`cudaMemcpy*()`).
    - **Performance**: 
        - Data access speed may be maximized by migrating data towards processors that access it most frequently
        - Total system memory may be reduced by avoiding duplicating memory on both CPUs and GPUs
    - **Functionality**: enables GPU programs to work on data that exceeds the GPU memory capacity
- Data movement still takes place

### Asynchronous SIMT Programming Model
- The SIMT (Single Instruction, Multiple Thread) programming model is a key feature of CUDA.
- A thread is the lowest level of abstraction for ding a computation or a memory operation.
- The asynchronous programming model defines the behavior of asynchronous operations in CUDA threads

#### Asynchronous Operations
- An asynchronous operation is defined as an operation that is initiated by a CUDA thread and is executed asynchronously as-if by another thread.
- The CUDA thread that initiated the operation may continue to execute other operations while the asynchronous operation is being executed.
- An asynchronous operation uses a synchronization object to synchronize the completion of the operation

### CUDA Programming Interface
- The CUDA programming interface consists of a set of host functions and device functions.
- The compilation is do with the `nvcc` compiler driver
- The CUDA runtime API provides a simple interface for C and C++ functions that can be called from the host that manage the allocation, deallocation, and transfer of data between the host and the device.
- The runtime is built on top of a lower-level C API

#### Compilation with NVCC
- Kernels can be directly written in PTX (Parallel Thread Execution) assembly language, as a programmer would write directly assembly code for a CPU
- It is instead more common to write kernels in C/C++ and let the compiler generate the PTX code
- After the PTX code is generated, it is compiled to machine code by the PTX compiler named SASS which is architecture-specific

#### Workflow
- Source files compiled with nvcc can include both host and device code
- nvcc separates the host code from the device code and compiles each separately
    - Host code is compiled to object code by the host compiler
    - Device code is compiled to PTX code by the device compiler

#### CUDA runtime
- The runtime is implemented in a shared library that is linked with the application at runtime
- All entry points are prefixed with `cuda`


