# 21_GPU_PROGRAMMING

> https://www.youtube.com/watch?v=xz9DO-4Pkko

- [21\_GPU\_PROGRAMMING](#21_gpu_programming)
  - [Topics](#topics)
  - [Tensor cores](#tensor-cores)
  - [Terminology](#terminology)
  - [Why CUDA (GPUs) is better than SIMD](#why-cuda-gpus-is-better-than-simd)
    - [Drawbacks](#drawbacks)
  - [CPU vs GPU](#cpu-vs-gpu)
    - [CPU](#cpu)
    - [GPU](#gpu)
  - [GPU computing](#gpu-computing)
    - [Traditional program structure](#traditional-program-structure)
      - [Terminology](#terminology-1)
      - [Recall: SPMD](#recall-spmd)
    - [CUDA / OPENCL Programming Model](#cuda--opencl-programming-model)
      - [Traditional program structure in CUDA](#traditional-program-structure-in-cuda)
      - [Indexing and memory access](#indexing-and-memory-access)
        - [How is it actually stored in memory?](#how-is-it-actually-stored-in-memory)
      - [1D Grid (One way to do it)](#1d-grid-one-way-to-do-it)
      - [2D Grid (Another way to do it)](#2d-grid-another-way-to-do-it)
    - [Review of GPU Architecture](#review-of-gpu-architecture)
    - [Performance considerations](#performance-considerations)
      - [Memory access - Latency hiding](#memory-access---latency-hiding)
        - [Example](#example)
    - [Memory access - Coalescing](#memory-access---coalescing)
      - [Example - Uncoalesced memory accesses](#example---uncoalesced-memory-accesses)
      - [Example - Coalesced memory accesses](#example---coalesced-memory-accesses)
      - [AoS (Array of Structures) vs SoA (Structure of Arrays)](#aos-array-of-structures-vs-soa-structure-of-arrays)
    - [Memory access - Data reuse](#memory-access---data-reuse)
      - [Example](#example-1)
        - [Optimization techniques for data reuse](#optimization-techniques-for-data-reuse)
    - [Shared memory](#shared-memory)
      - [Reducing shared memory bank conflicts](#reducing-shared-memory-bank-conflicts)
        - [Drawbacks of shared memory](#drawbacks-of-shared-memory)
    - [SIMD utilization](#simd-utilization)
      - [Example - Intra warp divergence](#example---intra-warp-divergence)
      - [Example - Divergence-free execution](#example---divergence-free-execution)
      - [Vector reduction](#vector-reduction)
        - [Naive mapping](#naive-mapping)
        - [Divergence-free mapping](#divergence-free-mapping)
    - [Atomic operations](#atomic-operations)
      - [Histogram calculation](#histogram-calculation)
        - [Optimizing the histogram calculation](#optimizing-the-histogram-calculation)
    - [Data transfers between CPU and GPU](#data-transfers-between-cpu-and-gpu)
      - [Types of Data Transfers](#types-of-data-transfers)
      - [Asynchronous data transfers](#asynchronous-data-transfers)
        - [Example - Video processing](#example---video-processing)
      - [Unified Memory](#unified-memory)
  - [Summary](#summary)
  - [Collaborative computing](#collaborative-computing)
    - [Unified Memory](#unified-memory-1)
      - [Asynchronous kernel launches](#asynchronous-kernel-launches)
      - [Fine-grained heterogeneity](#fine-grained-heterogeneity)
    - [Collaborative patterns](#collaborative-patterns)


## Topics
- GPU as an accelerator
    - Program structure
        - Bulk synchronous programming model
    - Memory hierarchy and memory management
    - Performance considerations
        - Memory access
        - SIMD utilization
        - Atomic operations
        - Data transfers
- Collaborative computing

## Tensor cores
- It's a specialized hardware unit that can perform large numbers of multiplications and additions in parallel (especially for matrix multiplication).
- They are optimized to perform operations on small matrices very quickly.
- Crucial in modern deep learning & neural network training.
- They are not general-purpose, so they can't be used for arbitrary computations.
- Found within NVIDIA's newer GPUs, tensor cores operate alongside traditional CUDA cores
- To utilize tensor cores, software must be specifically written to leverage this hardware, such as CUDA and cuDNN libraries that support operations on these cores.
- They support mixed-precision computing, which allows for the use of both single and half-precision floating points.

## Terminology

<img src="../images/Screenshot 2024-04-26 at 16.39.34.png" alt="Terminology">

## Why CUDA (GPUs) is better than SIMD
- As discussed in the [SIMD section](./19_SIMD.md), the main pain point is that SIMD is:
    - Not flexible (VLEN required)
    - Not easy to program
- CUDA allows for a dynamic allocation of resources depending on the workload, enhancing flexibility in programming diverse applications.
- GPUs instead are easier to program due to the SPMD programming model.
    - GPUs have democratized High Performance Computing (HPC).
    - Greater FLOPS/$, massively parallel chip on a commodity PC (no need of a super computer, you have a power house in your PC).
- Many workloads exhibit inherent parallelism
    - Matrices
    - Image processing
    - Deep neural networks
- NVIDIA's consistent development and support for CUDA has fostered a robust ecosystem of tools, libraries, and community resources that assist developers in optimizing their applications for GPU execution.
- More scalable than SIMD

### Drawbacks
- New programming model: new and existing developers must learn to think in parallel terms and adapt traditional serial code to take full advantage of GPU capabilities.
- Algorithms need to be re-implemented and rethought
- Still has some bottlenecks
    - CPU-GPU communication (PCIe bus, NVLink)
    - DRAM memory bandwidth (GDDR5, GDDR6, HBM2)
        - Even though modern gpu's memory bandwidth is very high (300-900 GB/s), it's still a bottleneck and far from the theoretical peak of the GPU.
        - The # of cores & how powerful these cores are is higher than the memory bandwidth -> GPUs are more powerful but this ratio is worse than 30 years ago.
        - Data layout: inefficient data layouts can lead to poor memory access patterns that exacerbate bandwidth limitations.
- Performance optimizations are often hardware-specific, meaning that code optimized for one generation of GPUs may not perform as well on another, requiring ongoing maintenance and updates.

## CPU vs GPU
Different philosophies:
- CPU: a few **out-of-order** cores
- GPU: Many **in-order** FGMT (Fine Grained MultiThreading)

### CPU
- CPUs are designed to minimize latency for a small set of instructions at a time. This design is optimized for general-purpose computing where tasks often have complex dependency chains.
- Out-of-order execution allows CPU cores to execute instructions as soon as the operands are available, rather than adhering to the original program order. This improves the efficiency of instruction pipelines by filling in execution stalls caused by instruction dependencies.
    - Requires complex hardware to manage instruction reordering and ensure correct program execution.
    - Due to the complexity of out-of-order execution, CPUs have fewer cores (powerful cores) and are optimized for single-threaded performance rather than parallelism.

### GPU
- Focus on throughput rather than latency. They are optimized for workloads that can be divided into many smaller operations that can be processed simultaneously
- GPU cores generally execute instructions in the order they are received, simplifying the hardware design and allowing for many more cores to be packed onto a single chip.
- To handle the in-order execution without suffering from latency issues, GPUs use fine-grained multithreading. This approach allows each core to switch between multiple threads in a single cycle, thus hiding execution latency caused by long-running operations or memory accesses.
- GPUs contain hundreds to thousands of simpler cores. Smaller cores are more power-efficient and can be packed more densely onto a chip, allowing for massive parallelism.

<img src="../images/Screenshot 2024-04-26 at 16.49.11.png" alt="CPU vs GPU">

## GPU computing
- Computation is offloaded to the GPU
- 3 steps
    1. CPU-GPU data transfer
        - This step involves latency and bandwidth considerations as data moves over PCIe or NVLink connections. The amount of data and the speed of the bus can significantly impact the overall performance of GPU-accelerated applications.
    2. GPU kernel execution
        - The performance of the GPU kernel can vary widely based on how well the kernel is optimized for the GPU’s architecture.
    4. GPU-CPU data transfer

<img src="../images/Screenshot 2024-04-26 at 16.51.43.png" alt="GPU computing">

- These days it's more frequent to have SoC (System on a Chip) where the CPU and GPU are on the same chip & have access to the same memory space (e.g. Apple Silicon).
    - With shared memory, the CPU and GPU can directly access the same data without needing to copy it between separate memory pools.
    - Integrating the CPU and GPU on the same chip improves thermal management and reduces power consumption compared to systems with separate chips.
    - Still less powerful than typical discrete GPUs.
    - Well-suited for mobile devices and laptops where power efficiency is a priority.

### Traditional program structure
- CPU threads and GPU kernels
- Sequential or modestly parallel code runs on the CPU
- Massive parallelism runs on the GPU

<img src="../images/Screenshot 2024-04-26 at 16.54.01.png" alt="Traditional program structure">

#### Terminology
- Host <-> CPU
- Device <-> GPU

#### Recall: SPMD
> As seen here [SPMD](20_GPU_ARCHITECTURE.md#spmd-single-program--procedure-multiple-data)
- Single procedure/program, multiple data
    - This is a programming model rather than a hardware model.
- Each processing element executes the same procedure, but on different data.
    - Procedures can synchronize and communicate (barriers)
- Multiple instructions streams execute the same program
    - Each program
        1. Works on a different part of the data
        2. Can execute different control flow, at run-time 

### CUDA / OPENCL Programming Model
- SIMT or SPMD
- Bulk synchronous programming model
    - All threads in a block execute the same instruction at the same time
    - All threads in a block are synchronized at the end of each instruction
    - All threads in a block can communicate and share data
    - Threads in different blocks can't communicate
    - Global (coarse-grained) synchronization is possible between kernels
- The host (CPU typically) allocates memory, copies data and launches kernels
    - Grid: collection of blocks
    - Block: collection of threads
        - Within a block, shared memory and synchronization 
    - Thread: single execution unit
    <img src="../images/Screenshot 2024-04-26 at 17.28.22.png" alt="Memory hierarchy">
- The only way to synchronize threads inside a kernel is by terminating the kernel (global synchronization)
    - Inside a block, threads can synchronize using barriers
  - Hardware is free to schedule thread blocks and we do not have control over it (the order of execution of blocks is not guaranteed). Even though we know that the scheduling is done with round-robin.
  - This is the reason why that all the blocks have finish is by terminating the kernel.

<img src="../images/Screenshot 2024-04-26 at 17.25.14.png" alt="CUDA / OPENCL Programming Model">

#### Traditional program structure in CUDA
- Function prototypes

    ```cpp
    float serialFunction(...);
    __global__ void kernel(...);
    ```
- Main function
    1. **Allocate memory** space on the device -> `cudaMalloc(&d_in, bytes)` 
    2. Transfer data from **host to device** -> `cudaMemCpy(d_in, h_in, bytes, cudaMemcpyHostToDevice)`
    3. Execution configuration setup
        - \# of blocks
        - \# of threads per block
    4. **Kernel launch** -> `kernel<<<numBlocks, numThreadsPerBlock>>>(d_in, d_out)`
    5. **Transfer data back** from device to host -> `cudaMemCpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost)`
    6. Repeat as needed*

- Kernel `__global__ void kernel(type args, ...)`
    - To distinguish from host functions
    - Automatic variables transparently assigned to registers
    - Shared memory: `__shared__`
    - Intra-block synchronization: `__syncthreads()`

- Memory allocation
    - `cudaMalloc(&d_in, bytes)`
    - `cudaMemCpy(d_in, h_in, bytes, cudaMemcpyHostToDevice)`
    - `cudaMemCpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost)`
    - `cudaFree(d_in)`

- Kernel launch
    - `kernel<<<numBlocks, numThreadsPerBlock>>>(d_in, d_out)`

- Memory deallocation
    - `cudaFree(d_in)`

- Explicit synchronization
    - `cudaDeviceSynchronize()`
    - Where do we use it? We use it in the host code right after the kernel call to make sure that the kernel has finished executing. Why? It's the only way to know that the kernel has finished executing. The kernel call is asynchronous, so the host code will continue executing after the kernel call even though the kernel is still executing.

#### Indexing and memory access
- Images are 2D data structures
    - height x width
    - `Image[j][i], where 0<=j<height, 0<=i<width`
    <img src="../images/Screenshot 2024-04-26 at 17.45.27.png" alt="Indexing and memory access">

##### How is it actually stored in memory?
- **Row-major layout**
- `Image[j][i]` is stored in memory as `Image[j*width + i]`

<img src="../images/Screenshot 2024-04-26 at 17.46.44.png" alt="Row-major layout">

#### 1D Grid (One way to do it)
- One GPU thread per pixel
- Grid of blocks of threads
    - `gridDim.x` -> # of thread blocks that the grid contains in the x direction
    - `blockDim.x` -> # of threads that the block contains in the x direction
    - `blockIdx.x` -> index of the block in the grid in the x direction
    - `threadIdx.x` -> index of the thread in the block in the x direction

<img src="../images/Screenshot 2024-04-26 at 17.51.37.png" alt="1D Grid">

#### 2D Grid (Another way to do it)
- `gridDim.x` -> # of thread blocks that the grid contains in the x direction
- `gridDim.y` -> # of thread blocks that the grid contains in the y direction

<img src="../images/Screenshot 2024-04-26 at 17.55.49.png" alt="2D Grid">

### Review of GPU Architecture
- Introduced in 2007, the Tesla architecture marked a significant step towards GPGPU (General-Purpose computing on Graphics Processing Units). It includes an array of Streaming Processors (SPs) which are essentially small, efficient cores designed to handle multiple threads simultaneously.
 
    <img src="../images/Screenshot 2024-04-26 at 18.15.20.png" alt="Review of GPU Architecture SPs">

- Streaming Multiprocessor (SM)
    - Each SM contains several SPs (Streaming Processors) that perform the actual computations. SPs are capable of executing integer and floating-point operations.
- Blocks are divided into **warps**
    - SIMD unit (32 threads)
    - Uses scoreboarding (internally) to manage the execution of threads

    <img src="../images/Screenshot 2024-04-26 at 18.16.58.png" alt="Review of GPU Architecture SM (Fermi)">

- Streaming Multiprocessors (SM) or Compute units (CU): essentially clusters of SIMD pipelines which are organized to optimize parallel processing of data.
    - SIMD pipelines
- Streaming Processors (SP) or CUDA cores: often referred to as vector lanes, these cores are where the computations are carried out. Each core can handle one thread at a time.
- Number of **SMs x SPs** across generations
    *   **Tesla (2007)**: 30 SMs x 8 SPs per SM
    *   **Fermi (2010)**: 16 SMs x 32 SPs per SM
    *   **Kepler (2012)**: 15 SMs x 192 SPs per SM
    *   **Maxwell (2014)**: 24 SMs x 128 SPs per SM
    *   **Pascal (2016)**: 56 SMs x 64 SPs per SM
    *   **Volta (2017)**: 84 SMs x 64 SPs per SM
    *   **Turing (2018)**: 68 SMs x 64 SPs per SM
    *   **Ampere (2020)**: 108 SMs x 64 SPs per SM
    *   **Hopper (2022)**: 144 SMs x 128 SPs per SM
    *   **Lovelace (2022)**: 144 SMs x 128 SPs per SM

### Performance considerations
- **Main bottlenecks**
    - **Global memory access** (the main GPU memory) is often a significant bottleneck due to its relatively high latency and limited bandwidth compared to the compute capabilities of the GPU cores.
    - **CPU-GPU data transfers** over the PCIe bus can substantially affect performance, particularly for applications that require frequent data exchanges.
- **Memory access**
    - **Latency hiding**
        - Occupancy refers to the number of warps that are active on a multiprocessor at a given time. Higher occupancy can help hide the latency of memory access because while some threads are waiting for data to arrive from memory, others can be executing
    - **Memory coalescing** refers to the optimization technique where adjacent threads access consecutive memory addresses. When memory access is coalesced, multiple data elements can be loaded in a single memory transaction, which reduces the number of required memory accesses and improves bandwidth utilization.
    - **Data reuse**
        - **Shared memory usage** can significantly reduce the reliance on slower global memory. Shared memory is much faster but limited in size
- SIMD (Warp) utilization
    - **Branch divergence** occurs when threads of the same warp follow different execution paths due to conditional statements. This divergence can lead to serialization within the warp, where some threads are idle while others are executing, reducing overall efficiency.
    - **Memory access patterns**, crucial for maximizing throughput.
        - Strided accesses, where threads access memory locations that are spaced apart, can lead to poor utilization of memory bandwidth.
- **Atomic operations** ensure that a particular memory location is updated atomically, preventing race conditions between threads.
    - **Avoid them if possible**
    - **Use them sparingly**
    - **Serialization**: two threads trying to write to the same memory location. One must wait for the other to complete, which can serialize the operations and degrade performance.
        - Overlap of communication and computation to mitigate the impact of data transfer times.
            - Asynchronous data transfers can help overlap communication and computation, allowing the CPU to perform other tasks while the GPU is processing data.

#### Memory access - Latency hiding
- **FGMT** (Fine-Grained Multi-Threading) helps to hide the latency of long memory accesses by interleaving the execution of multiple threads or warps. When one warp stalls due to a memory request, another warp can take over the execution unit.
- **Occupancy**: It is defined as the ratio of active warps to the maximum number of warps that can be active per Streaming Multiprocessor (SM). Basically how many warps are active at the same time.
    - **Active warps**: warps that are currently executing and not stalled waiting for resources.
    - **Maximum warps**: total number of warps that an SM can support at any time.
    - **Maximum number of blocks per SM**: each can typically support up to 32 blocks.
    - **Register usage** & **Shared memory usage**: fixed amount of register and shared memory, often around 256KB for registers and 64KB for shared memory in modern architectures.

<img src="../images/Screenshot 2024-04-27 at 15.38.09.png" alt="Memory access - Latency hiding">

##### Example
Consider a scenario where each thread block in a GPU program requires 30KB of shared memory. Given that the typical shared memory available per SM is 64KB, you can calculate the number of blocks that can fit within one SM:

**Calculation**: Since each block requires 30KB, only two blocks can be accommodated within one SM (2 \* 30KB = 60KB < 64KB).
Crucial because fitting more blocks might exceed the available shared memory, leading to a reduction in occupancy and potential performance degradation.

- Occupancy Calculation
    - **Number of Threads per Block**: Defined by the programmer, this affects how threads are distributed across warps and blocks.
    - **Number of Registers per Thread**: Known at compile time, more registers per thread reduce the number of threads (and thus warps) that can be active at any one time.
    - **Shared Memory per Block**: Also defined by the programmer, this is the amount of shared memory each block requires. High shared memory usage can limit the number of blocks per SM.

### Memory access - Coalescing
- It involves structuring memory accesses by threads within a warp to ensure that they are as efficient as possible, ideally leading to peak memory bandwidth utilization.
- **Peak bandwidth** utilization occurs when all threads in a warp access one cache line

<img src="../images/Screenshot 2024-04-27 at 15.45.38.png" alt="Memory access - Coalescing">

> When threads in a warp access global memory, it is most efficient if they access consecutive memory addresses. This allows the GPU to combine these accesses into a single memory transaction instead of multiple, reducing the number of memory accesses and maximizing bandwidth utilization.

#### Example - Uncoalesced memory accesses
Threads access memory locations that are not adjacent, leading to multiple memory transactions. This is less efficient and increases the latency of memory operations.

<img src="../images/Screenshot 2024-04-27 at 15.47.46.png" alt="Uncoalesced memory accesses">

> T stands for thread
> 
> Time period stands for Iteration

This causes 4 memory transactions.

#### Example - Coalesced memory accesses
Threads in a warp access one cache line, this results in a single memory transaction, significantly reducing the memory access time and increasing the efficiency of the warp.

<img src="../images/Screenshot 2024-04-27 at 15.48.57.png" alt="Coalesced memory accesses">

> T stands for thread
> 
> Time period stands for Iteration

This causes 1 memory transaction to load the cache line.

#### AoS (Array of Structures) vs SoA (Structure of Arrays)

- **AoS (Array of Structures)**: Commonly preferred in CPU applications due to the locality of reference and data organization that fits well with the CPU cache line structure.
- **SoA (Structure of Arrays)**: More advantageous for GPUs as it aligns with the need for memory coalescing. By organizing data into SoA, each thread accesses a contiguous segment of memory, which is optimal for GPUs and helps in maximizing the memory bandwidth.

<img src="../images/Screenshot 2024-04-27 at 15.58.16.png" alt="AoS vs SoA">

<img src="../images/Screenshot 2024-04-27 at 16.00.20.png" alt="AoS vs SoA">

### Memory access - Data reuse
Particularly useful in applications like image processing, where the same data elements are accessed multiple times by different operations or iterations. Optimizing data reuse can significantly reduce the demand on memory bandwidth and improve overall performance.
- **Efficient memory use** involves accessing the same memory locations multiple times before discarding them. This is particularly efficient when neighboring threads access overlapping data segments, as in convolutional operations used in image processing and neural networks.

#### Example
Consider a typical filter operation in image processing, such as applying a Gaussian blur. Each pixel's new value is calculated using a combination of its own value and the values of its neighbors.

```cpp
for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
        sum += gauss[i][j] * Image[(i+row-1)*width + (j+col-1)];
    }
}
```
<p align="center">
    <img src="../images/Screenshot 2024-04-27 at 16.06.09.png" alt="Data reuse">
</p>

> Each thread computes a value based on a 3x3 region of the image. As the filter moves across the image, many of these regions will overlap, meaning subsequent operations will reuse some of the same data elements.
> 
> What about the next 9 elements? We're going to access the same 6 elements again. This is data reuse. We do not want again to access the global memory to get the same data, if we can reuse it we can optimize the performance of the program.
> Moving the filter one pixel to the right in a row means six out of the nine pixels in the new filter window are the same as in the previous window.

##### Optimization techniques for data reuse
- **Shared Memory Utilization**: Temporarily store data that will be reused in the shared memory of the GPU, which is much faster to access than global memory. This reduces latency and bandwidth usage on repeated accesses.
- **Tiling** by breaking down the image or data array into smaller blocks or tiles that fit into shared memory. Process each tile independently while maximizing the reuse of data loaded into the shared memory.
- **Plan Access Patterns:** Organize thread access patterns so that threads access shared memory efficiently, avoiding bank conflicts and ensuring coalesced access to global memory when loading and storing data.

<p align="center">
    <img src="../images/Screenshot 2024-04-27 at 16.13.42.png" alt="Tiling">
</p>

```cpp
__shared__ int l_data[(L_SIZE+2)*(L_SIZE+2)];
// ..
// load tile into share memory
__syncthreads(); // make sure that all threads have finished loading the data!
for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
        sum += gauss[i][j] * l_data[(i+l_row-1)*(L_SIZE+2) + (j+l_col-1)];
    }
}
```

### Shared memory
- Shared memory also had limits
- Is organized into multiple memory banks to allow concurrent access by multiple threads
- Typically, 32 banks in NVIDIA GPUs:
    - Successive 32-bit words are stored in successive banks
    - `Bank = Address % 32`
    - In an ideal scenario, each thread in a warp accesses a different bank, allowing all 32 threads to access memory concurrently without any delays.
- **Within-Warp Conflicts:** Bank conflicts occur when multiple threads in the same warp request data from the same memory bank simultaneously. This results in serialization of these memory accesses, degrading performance.
    - If each thread accesses a separate bank, there are no conflicts, and access is maximized.
    - If N threads access the same bank, they are serialized, effectively reducing the bandwidth by a factor of N.
- Bank conflicts are only possible within a warp
    - No bank conflicts between different warps, why? Because the access to the shared memory is scheduled at different times for different warps, whereas in the same warp, the access can be scheduled at the same time making the threads vulnerable to bank conflicts.
- Bank conflict free

    <img src="../images/Screenshot 2024-04-27 at 16.24.59.png" alt="Shared memory">

- N-way bank conflicts

    <img src="../images/Screenshot 2024-04-27 at 16.26.10.png" alt="N-way bank conflicts">

#### Reducing shared memory bank conflicts
- Bank conflicts are only possible within a warp
    - No bank conflicts between different warps. Different warps access shared memory at different scheduled times, thus not conflicting with each other.
- If strided accesses are needed, some optimization techniques can help
    - **Padding**: Altering the data structure to add extra elements can prevent multiple threads from accessing the same bank. This technique is useful when data structures are statically sized and indexed.
    - **Randomized mapping**: Changing the order of elements in shared memory or using a different addressing scheme can help distribute memory accesses more evenly across the banks.
    - **Hash functions**: Applying a hash function to compute the address can randomize accesses across banks, reducing the likelihood of conflicts.
##### Drawbacks of shared memory
- Optimizing shared memory requires a deep understanding of memory access patterns within kernels. Programmers must anticipate how threads will access memory and adjust the data layout or indexing accordingly.
- Effective optimization often requires knowledge about the number of banks and the specifics of the GPU architecture, which might not always be readily available or could change with new GPU generations.

### SIMD utilization
> Also seen [here](./20_GPU_ARCHITECTURE.md#conditional-control-flow-instructions)

Efficiency can be compromised by control flow divergence among threads within a warp.
- **Branch Divergence** occurs when threads within the same warp need to execute different execution paths based on their data. It leads to serial execution of divergent branches, reducing the effective utilization of GPU resources.

<p align="center">
    <img src="../images/Screenshot 2024-04-27 at 16.41.29.png" alt="SIMD utilization">
</p>

> In the image above causes 2 different execution paths, resulting in the twice the number of cycles to execute the code.

- **Hardware solutions** do exists, some advanced microarchitectures in CPUs mitigate branch divergence using techniques like branch prediction and speculative execution, GPUs typically do not implement these due to their design focused on massive parallelism and throughput.
- **Software Approaches** use software in a way that minimizes divergence is key. This involves understanding the common execution paths and structuring code to align with these paths as much as possible.

#### Example - Intra warp divergence
How threads within a warp can diverge based on simple conditions, leading to different functions being called based on thread indices
```cpp
Compute(threadIdx.x);

if (threadIdx.x % 2 == 0){
    Do_this(threadIdx.x);
} else {
    Do_that((threadIdx.x % 32) * 2 + 1);
}
```
<p align="center">
    <img src="../images/Screenshot 2024-04-27 at 16.43.35.png" alt="Intra warp divergence">
</p>

#### Example - Divergence-free execution
In this scenario, if the condition splits threads from different warps, each set of threads can execute concurrently without causing intra-warp divergence, thus maintaining full utilization of the SIMD unit.

```cpp
Compute(threadIdx.x);

if (threadIdx.x < 32){
    Do_this(threadIdx.x * 2);
} else {
    Do_that((threadIdx.x % 32) * 2 + 1);
}
```
<p align="center">
    <img src="../images/Screenshot 2024-04-27 at 16.44.56.png" alt="Divergence-free execution">
</p>

> If these conditional branches are handled by separate warps, there is no impact on the utilization within a single warp, ensuring efficient SIMD execution.

#### Vector reduction    

Is a common parallel algorithmic pattern where all elements of an array are combined to produce a single value, such as the sum, minimum, or maximum. Efficiently implementing vector reduction on GPUs can significantly impact the performance of many applications.

##### Naive mapping

In this method, threads reduce pairs of elements in multiple steps, where each thread combines two elements and writes back the result. The number of active threads halves in each subsequent iteration, leading to poor SIMD utilization as fewer threads remain active.

<img src="../images/Screenshot 2024-04-27 at 16.54.01.png" alt="Naive mapping">

> This image illustrates how threads become increasingly idle in later iterations, as they have fewer elements to combine.

```cpp
__shared float partialSum[]

unsigned int t = threadIdx.x;

for (unsigned int stride = 1; stride < blockDim.x; stride *= 2){
    __syncthreads();

    if (t % (2*stride) == 0){
        partialSum[t] += partialSum[t + stride];
    }
}
```

> This code demonstrates a typical reduction pattern where each thread progressively reduces elements spaced by increasing strides. However, this leads to increasing divergence and underutilization of the GPU's SIMD capabilities.

##### Divergence-free mapping

To improve SIMD utilization, it is preferable to ensure that all threads in a warp remain active without divergent execution paths. This approach modifies the loop to decrease the range of active threads more gradually and maintains more consistent thread activity.

<img src="../images/Screenshot 2024-04-27 at 17.10.39.png" alt="Divergence-free mapping">

> No 100% SIMD utilization, but better than the naive mapping.
>
> The code is a little bit more complex, but it's worth it.

```cpp
__shared float partialSum[]

unsigned int t = threadIdx.x;

for (int stride = blockDim.x; stride > 1; stride >> 1) {
    __syncthreads();

    if (t < stride) {
        partialSum[t] += partialSum[t + stride];
    }
}
```

> This code snippet uses a more sophisticated approach by halving the stride more smoothly, ensuring that the number of active threads decreases in a more controlled manner. It helps maintain a higher level of thread activity and reduces divergence.

### Atomic operations

Atomic operations are crucial for ensuring correct computations when multiple threads need to update the same memory location concurrently. They are widely used in parallel algorithms for tasks such as counting, accumulating sums, or implementing mutexes.
- When multiple threads attempt to update the same memory location simultaneously, without atomic operations, data corruption or inconsistencies can occur.
- CUDA: `int atomicAdd(int*, int)`
- PTX: `atom.shared.add.u32 %r25, [%rd14], %r24`
    - PTX is an intermediate representation of the CUDA code that is generated by the compiler. Useful to ensure continuity of the code between generations of GPUs.
- SASS: Specific Assembly Code for the particular GPU architecture
    ```asm
    ; Tesla, Fermi, Kepler architectures
    /*00a0*/ LDSLK P0, R9, [R8];
    /*00a8*/ @P0 IADD R10, R9, R7;
    /*00b0*/ @P0 STSCUL P1, [R8], R10;
    /*00b8*/ @!P0 BRA 0Xa0;
    ```

    > The above was not very efficient. The atomicAdd where compiled to a load, add, store, and a branch instruction. P0 & P1 (predicate registers) are 1 **bit per thread**, 0 or 1 depending on whether the thread managed to acquire the lock or not. So if 2 threads belonging to the same warp wanted to write to the same memory location, they have to contend for this lock, and only one of them will be able to write to the memory location. The thread that did not manage to acquire the lock would have retried the operation. -> **VERY INEFFICIENT** when multiple threads tried to write to the same memory location.

    > Newer GPU generations introduced more efficient atomic operations that reduce the need for locks and retries, thanks to hardware-level support for atomicity.

    ```asm
    ; Maxwell, Pascal, Volta
    /*01f8*/ ATOMS.ADD RZ, [R7], R11;
    ```

    > Native atomic operations for 32-bit integer and 32-bit and 64-bit atomicCAS (compare and swap) operations. 

- **Intra-Warp Conflict Degree** is the number of threads within a warp that try to update the same memory position can significantly impact performance. The conflict degree can range from 1 (no conflict) to 32 (all threads in a warp update the same location).
- **Serialization and Performance:** high conflict degrees lead to serialization of updates, which can severely degrade performance.

<img src="../images/Screenshot 2024-04-27 at 17.44.02.png" alt="Atomic operations">

#### Histogram calculation
Histograms count the number of data instances in disjoint categories (bins)

```cpp
for (each pixel i in image I) {
    Pixel = I[i]               // Read pixel
    Pixel`= Computation(Pixel) // Optional computation
    Histogram[Pixel`]++         // Vote in histogram bin
}
```

<img src="../images/Screenshot 2024-04-27 at 17.48.36.png" alt="Histogram calculation">

- Frequent conflicts in natural images because of the high number of pixels with the same value

##### Optimizing the histogram calculation
- **Privatization**: Per-block sub-histograms in shared memory, basically, instead of using global memory, we use shared memory to store the histogram. Each block has its own histogram in shared memory, and at the end, we combine all the histograms from all the blocks.

<img src="../images/Screenshot 2024-04-27 at 17.55.16.png" alt="Optimizing the histogram calculation">

### Data transfers between CPU and GPU
- One of the main bottlenecks in GPU computing
- Glued together by the **PCIe bus** or **NVLink**
    - These are the primary channels for data transfer between the CPU and GPU. While NVLink offers significantly higher bandwidth than traditional PCIe connections, the bandwidth (ranging from 300 to 900 GB/s) can still be a limiting factor in overall application performance.

<img src="../images/Screenshot 2024-04-27 at 17.58.47.png" alt="Data transfers between CPU and GPU">

#### Types of Data Transfers
- **Synchronous Transfers**: Operations like `cudaMemcpy()` block the CPU until the GPU finishes the data transfer. This can lead to inefficiencies if the CPU or GPU has to wait for the other to complete its tasks.
- **Asynchronous Transfers**: `cudaMemcpyAsync()` allows the CPU to continue processing other tasks while the GPU handles the data transfer, potentially overlapping with other GPU operations like kernel execution.
- **Streams:** These are sequences of operations that the GPU performs in order. Using multiple streams can greatly enhance the efficiency of data transfers and kernel executions by organizing them into independent sequences that can be processed concurrently.

#### Asynchronous data transfers
- Computation divided into nStreams
    - D input data instances, B blocks
    - nStreams
        - D/nStreams data instances
        - B/nStreams blocks

<img src="../images/Screenshot 2024-04-27 at 18.01.51.png" alt="Asynchronous data transfers">

> Not always possible

- Estimates
    - `t_e + t_T / nStreams`, where `t_e >= t_T (dominant kernel)`
    - `t_T+ t_e / nStreams`, where `t_T >= t_e (dominant transfers)`

> Little overhead added in managing the streams, but worth it.

##### Example - Video processing
- Applications with independent computation on different data instances can benefit from asynchronous transfers

<img src="../images/Screenshot 2024-04-27 at 18.05.06.png" alt="Video processing">

#### Unified Memory
- simplifies memory management by allowing the CPU and GPU to share the same memory space, eliminating the need for explicit data transfers.
- Particularly beneficial in applications where frequent data exchange between the CPU and GPU is necessary. It reduces the overhead of managing separate memory spaces and can lead to simpler and more efficient code.

## Summary
- GPU as an accelerator
    - Program structure
        - Bulk synchronous programming model
    - Memory hierarchy and memory management
    - Performance considerations
        - Memory access
            - Latency hiding: occupancy (TLP)
            - Memory coalescing
            - Data reuse: shared memory
        - SIMD utilization
        - Atomic operations
        - Data transfers

## Collaborative computing
Collaborative computing between CPUs and GPUs can be complex due to the fundamentally different architectures and processing capabilities of each.
However, advancements like Unified Memory and asynchronous kernel launches have significantly improved the synergy between these two components.

### Unified Memory
- The CPU and GPU share the same memory space. This eliminates the need for manual data transfers and helps manage data consistency automatically.
    - Since CUDA 6.0: unified memory
    - Since CUDA 8.0 + Pascal architecture: GPU page faults
    <img src="../images/Screenshot 2024-04-27 at 18.19.57.png" alt="Unified Memory">
- Easier programming
    - Developers no longer need to explicitly copy data between the CPU and GPU. Memory allocation with `cudaMallocManaged()` automatically ensures that the data is accessible from both the CPU and GPU.
    - No need to manage data consistency which reduces the risk of errors and simplifies the code.
    
#### Asynchronous kernel launches
- **Non-Blocking Computations**: Kernel launches are asynchronous, meaning the CPU can perform other tasks while the GPU is processing. This allows for more efficient use of system resources.

```cpp
// allocate input
malloc(input, ...);
cudaMallocManaged(&d_input, ...);
memcpy(d_input, input, ...);

// allocate output
malloc(output, ...);
cudaMallocManaged(&d_output, ...);

// launch kernel
gpu_kernel<<<...>>>(d_input, d_output);

// CPU can do things here!

// sync
cudaDeviceSynchronize();

// copy output to host memory
cudaMemcpy(output, d_output, ...);
```

#### Fine-grained heterogeneity
With architectures like Pascal and Volta, more sophisticated features such as
- **CPU-GPU memory coherence** which means that if the GPU updates a memory location, the change is immediately visible to the CPU and vice versa, without the need for explicit synchronization commands.
- **System-wide atomics** are supported which extend the concept of atomic operations to work across the entire system, meaning that these operations can be performed with consistency and atomicity no matter whether they are initiated by the CPU or the GPU.

Allows for fine-grained control over how data is accessed and modified, enhancing the capabilities for collaborative computing.
- **Reduced Complexity** so that developers no longer need to implement complex data transfer and synchronization mechanisms in their applications, simplifying code and reducing potential bugs.
- **Performance Enhancements** by reducing the need for data transfers and manual synchronization can significantly speed up applications, particularly those involving frequent interactions between the CPU and GPU.
- **Enhanced Flexibility** so that applications can more dynamically distribute tasks between the CPU and GPU based on their respective strengths and current load.

```cpp
// allocate input
cudaMallocManaged(&d_input, ...);

// allocate output
cudaMallocManaged(&d_output, ...);

// launch kernel
gpu_kernel<<<...>>>(d_input, d_output);

// CPU can do things here!
output[x] = input[y];

output[x+1].fetch_add(1);
```

> Since CUDA 8.0
> 
> - Unified memory `cudaMallocManaged(&h_in, in_size)`
> - System-wide atomics `old = atomicAdd_system(&h_out[x], inc)`

### Collaborative patterns

<img src="../images/Screenshot 2024-04-27 at 18.30.48.png" alt="Collaborative patterns">

<img src="../images/Screenshot 2024-04-27 at 18.35.20.png" alt="Collaborative patterns">
