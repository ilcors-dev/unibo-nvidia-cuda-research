# 21_GPU_PROGRAMMING

> https://www.youtube.com/watch?v=xz9DO-4Pkko

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

## Terminology

<img src="../images/Screenshot 2024-04-26 at 16.39.34.png" alt="Terminology">

## Why CUDA (GPUs) is better than SIMD
- As discussed in the [SIMD section](./19_SIMD.md), the main pain point is that SIMD is:
    - Not flexible (VLEN required)
    - Not easy to program
- GPUs instead are easier to program due to the SPMD programming model.
    - GPUs have democratized High Performance Computing (HPC).
    - Greater FLOPS/$, massively parallel chip on a commodity PC (no need of a super computer, you have a power house in your PC).
- Many workloads exhibit inherent parallelism
    - Matrices
    - Image processing
    - Deep neural networks

### Drawbacks
- New programming model
- Algorithms need to be re-implemented and rethought
- Still has some bottlenecks
    - CPU-GPU communication (PCIe bus, NVLink)
    - DRAM memory bandwidth (GDDR5, GDDR6, HBM2)
        - Even though modern gpu's memory bandwidth is very high (300-900 GB/s), it's still a bottleneck and far from the theoretical peak of the GPU.
        - The # of cores & how powerful these cores are is higher than the memory bandwidth -> GPUs are more powerful but this ratio is worse than 30 years ago.
        - Data layout

## CPU vs GPU
Different philosophies:
- CPU: a few **out-of-order** cores
- GPU: Many **in-order** FGMT (Fine Grained MultiThreading)

<img src="../images/Screenshot 2024-04-26 at 16.49.11.png" alt="CPU vs GPU">

## GPU computing
- Computation is offloaded to the GPU
- 3 steps
    1. CPU-GPU data transfer
    2. GPU kernel execution
    3. GPU-CPU data transfer

<img src="../images/Screenshot 2024-04-26 at 16.51.43.png" alt="GPU computing">

- These days it's more frequent to have SoC (System on a Chip) where the CPU and GPU are on the same chip & have access to the same memory space (e.g. Apple Silicon).
    - Still less powerful than typical discrete GPUs.

### Traditional program structure
- CPU threads and GPU kernels
    - Sequential or modestly parallel code runs on the CPU
    - Massive parallelism runs on the GPU

<img src="../images/Screenshot 2024-04-26 at 16.54.01.png" alt="Traditional program structure">

#### Terminology
- Host <-> CPU
- Device <-> GPU

#### Recall: SPMD
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

##### 1D Grid (One way to do it)
- One GPU thread per pixel
- Grid of blocks of threads
    - `gridDim.x` -> # of thread blocks that the grid contains in the x direction
    - `blockDim.x` -> # of threads that the block contains in the x direction
    - `blockIdx.x` -> index of the block in the grid in the x direction
    - `threadIdx.x` -> index of the thread in the block in the x direction

<img src="../images/Screenshot 2024-04-26 at 17.51.37.png" alt="1D Grid">

##### 2D Grid (Another way to do it)
- `gridDim.x` -> # of thread blocks that the grid contains in the x direction
- `gridDim.y` -> # of thread blocks that the grid contains in the y direction

<img src="../images/Screenshot 2024-04-26 at 17.55.49.png" alt="2D Grid">

### Review of GPU Architecture
- Streaming Processor Array (SPs) - Tesla architecture
 
<img src="../images/Screenshot 2024-04-26 at 18.15.20.png" alt="Review of GPU Architecture SPs">

- Streaming Multiprocessor (SM)
    - Streaming Processors (SPs)
- Blocks are divided into **warps**
    - SIMD unit (32 threads)
    - Uses scoreboarding (internally) to manage the execution of threads

<img src="../images/Screenshot 2024-04-26 at 18.16.58.png" alt="Review of GPU Architecture SM (Fermi)">

- Streaming Multiprocessors (SM) or Compute units (CU)
    - SIMD pipelines
- Streaming Processors (SP) or CUDA cores
    - Vector lanes
- Number of **SMs x SPs** across generations
    - Tesla (2007):     30 x 8
    - Fermi (2010):     16 x 32
    - Kepler (2012):    15 x 192
    - Maxwell (2014):   24 x 128
    - Pascal (2016):    56 x 64
    - Volta (2017):     84 x 64
    - Turing (2018):    68 x 64
    - Ampere (2020):    108 x 64
    - Hopper (2022):    144 x 64
    - Lovelace (2022):  144 x 64

### Performance considerations
- Main bottlenecks
    - Global memory access
    - CPU-GPU data transfers
- Memory access
    - Latency hiding
        - Occupancy -> based on fine-grained parallelism architecture of the GPU
    - Memory coalescing -> how do we access memory?
    - Data reuse
        - Shared memory usage
- SIMD (Warp) utilization
    - Branch divergence
    - Memory access patterns
- Atomic operations
    - Avoid them if possible
    - Use them sparingly
    - Serialization -> two threads trying to write to the same memory location
        - Overlap of communication and computation

#### Memory access - Latency hiding
- **FGMT** can hide long latency operations (e.g., memory access)
- **Occupancy**: ratio of active warps to maximum warps
    - **Active warps**: warps that are not stalled
    - **Maximum warps**: total number of warps that can be active
    - 
