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
- **Occupancy**: ratio of active warps to maximum warps (might change based on the architecture, typically increases with newer architectures)
    - **Active warps**: warps that are not stalled
    - **Maximum warps**: total number of warps that can be active per SM, typically 64
    - **Maximum number of blocks per SM**: typically 32
    - **Register usage**: typically 256KB
    - **Shared memory usage**: typically 64KB

<img src="../images/Screenshot 2024-04-27 at 15.38.09.png" alt="Memory access - Latency hiding">

##### Example
Imagine, a program where each thread block needs to use 30KB, how many blocks can we fit in an SM? Only 2 blocks because 2 * 30KB = 60KB over 64KB (shared memory usage).

- Occupancy Calculation
    - Number of threads per block (defined by the programmer)
    - Number of registers per thread (known at compile time)
    - Shared memory per block (defined by the programmer)

### Memory access - Coalescing
- When accessing global memory, we want to make sure that concurrent threads access nearby memory locations
- **Peak bandwidth** utilization occurs when all threads in a warp access one cache line

<img src="../images/Screenshot 2024-04-27 at 15.45.38.png" alt="Memory access - Coalescing">

#### Uncoalesced memory accesses

<img src="../images/Screenshot 2024-04-27 at 15.47.46.png" alt="Uncoalesced memory accesses">

> T stands for thread
> 
> Time period stands for Iteration

This causes 4 memory transactions.

#### Coalesced memory accesses

<img src="../images/Screenshot 2024-04-27 at 15.48.57.png" alt="Coalesced memory accesses">

> T stands for thread
> 
> Time period stands for Iteration

This causes 1 memory transaction to load the cache line.

#### AoS (Array of Structures) vs SoA (Structure of Arrays)

<img src="../images/Screenshot 2024-04-27 at 15.58.16.png" alt="AoS vs SoA">

- CPUs prefer AoS, GPUs prefer SoA

<img src="../images/Screenshot 2024-04-27 at 16.00.20.png" alt="AoS vs SoA">

### Memory access - Data reuse
- Same memory locations accessed by neighboring threads

```cpp
for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
        sum += gauss[i][j] * Image[(i+row-1)*width + (j+col-1)];
    }
}
```

<img src="../images/Screenshot 2024-04-27 at 16.06.09.png" alt="Data reuse">

- The image above is a typical **filter** operation used in image processing, neural networks, etc.
- We're accessing these 9 elements in the image and do the computation. What about the next 9 elements? We're going to access the same 6 elements again. This is data reuse. We do not want again to access the global memory to get the same data, if we can reuse it we can optimize the performance of the program.

#### Tiling
- To take advantage of data reuse, we can use tiling, where we divide the input into **tiles** that can be loaded into **share memory**

<img src="../images/Screenshot 2024-04-27 at 16.13.42.png" alt="Tiling">

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
- Is an interleaved (banked) memory. If we have a warp of 32 threads, each thread accesses a different bank (ideal scenario). This allows for concurrent access to shared memory.
- Typically, 32 banks in NVIDIA GPUs:
    - Successive 32-bit words are stored in successive banks
    - Bank = Address % 32
- Bank conflicts are only possible within a warp
    - No bank conflicts between different warps, why? Because the access to the shared memory is scheduled at different times for different warps, whereas in the same warp, the access can be scheduled at the same time making the threads vulnerable to bank conflicts.
- Bank conflict free

    <img src="../images/Screenshot 2024-04-27 at 16.24.59.png" alt="Shared memory">

- N-way bank conflicts

    <img src="../images/Screenshot 2024-04-27 at 16.26.10.png" alt="N-way bank conflicts">

#### Reducing shared memory bank conflicts
- Bank conflicts are only possible within a warp
    - No bank conflicts between different warps
- If strided accesses are needed, some optimization techniques can help
    - **Padding**: add extra elements to the shared memory array to avoid conflicts
    - **Randomized mapping**: change the order of the elements in the shared memory array to avoid conflicts
    - **Hash functions**: use hash functions to map addresses to banks instead of a simple linear mapping
- Drawbacks, as a programmer you need to know the access pattern of the threads to optimize the shared memory access. You do not really know how memory is accessed.

### SIMD utilization
- Control flow problem in GPUs / SIMT
    - Branch divergence: occurs when threads inside warps branch to different execution paths

<img src="../images/Screenshot 2024-04-27 at 16.41.29.png" alt="SIMD utilization">

> In the image above causes 2 different execution paths, resulting in the twice the number of cycles to execute the code.

- Today there are some hardware related solutions which GPUs still do not implement
- What we can do is find a way to write programs in a nicer way to avoid these kind of problems

- Intra warp **divergence**

    ```cpp
    Compute(threadIdx.x);

    if (threadIdx.x % 2 == 0){
        Do_this(threadIdx.x);
    } else {
        Do_that((threadIdx.x % 32) * 2 + 1);
    }
    ```

    <img src="../images/Screenshot 2024-04-27 at 16.43.35.png" alt="Intra warp divergence">

- **Divergence-free** execution

    ```cpp
    Compute(threadIdx.x);

    if (threadIdx.x < 32){
        Do_this(threadIdx.x * 2);
    } else {
        Do_that((threadIdx.x % 32) * 2 + 1);
    }
    ```

    <img src="../images/Screenshot 2024-04-27 at 16.44.56.png" alt="Divergence-free execution">

    > In this case, if these two parts belong to different warps, divergence no longer exists -> 100% utilization of the SIMD unit.

#### Vector reduction    
- Is the process of combining all elements of an array into a single value

##### Naive mapping

<img src="../images/Screenshot 2024-04-27 at 16.54.01.png" alt="Naive mapping">

> Threads, iteration after iteration, are more and more divergent (far away), low SIMD utilization.

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

##### Divergence-free mapping
- All active threads belong to the same warp
- In the example before, we do not really care in which order the calculations are done

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

### Atomic operations
- The problem with atomic operations occurs when threads in the same warp when try to write to the same memory location.
- Atomic operations are needed when threads might update the same memory locations at the same time
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

    > Got more efficient with the newer generations

    ```asm
    ; Maxwell, Pascal, Volta
    /*01f8*/ ATOMS.ADD RZ, [R7], R11;
    ```

    > Native atomic operations for 32-bit integer and 32-bit and 64-bit atomicCAS (compare and swap) operations.

- We define the intra-warp conflict degree as the number of threads in a warp that update the same memory position
- The conflict degree can be between 1 and 32

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
- Even though the bandwidth is high, it's still a bottleneck (300-900 GB/s nowadays)
- Synchronous and asynchronous data transfers
    - Synchronous: `cudaMemcpy()`
    - Asynchronous: `cudaMemcpyAsync()`
- Streams (Command queues)
    - Sequence of operations that are performed in order
        - CPU-GPU data transfers
        - Kernel execution
            - D input data instances, B blocks
        - GPU-CPU data transfers
    - Default stream
    <img src="../images/Screenshot 2024-04-27 at 17.58.47.png" alt="Data transfers between CPU and GPU">

        > Good thing is that the data that I'm transferring (in some cases) in this part of the transfer is going to be done in an independent way from the kernel execution -> Asynchronous data transfers.

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

##### Example, Video processing
- Applications with independent computation on different data instances can benefit from asynchronous transfers

<img src="../images/Screenshot 2024-04-27 at 18.05.06.png" alt="Video processing">

Nowadays we have **Unified Memory** where the CPU and GPU share the same memory space. This is very useful because we do not have to transfer data between the CPU and GPU, the data is already there. This is very useful for applications where the CPU and GPU need to share data.

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
The collaboration between the CPU and GPU is difficult

The only thing that the CPU can do is to launch the kernel and wait for it to finish, and then transfer the data back to the CPU. The CPU cores are not doing nothing while the GPU is doing the computation.

### Unified Memory
- The CPU and GPU share the same memory space
    - Since CUDA 6.0: unified memory
    - Since CUDA 8.0 + Pascal architecture: GPU page faults
    <img src="../images/Screenshot 2024-04-27 at 18.19.57.png" alt="Unified Memory">
- Easier programming
    - No need to manage data transfers
    - No need to manage data consistency
    - `cudaMallocManaged()`
    
#### Kernel launches are asynchronous
- CPU can work while the GPU is working
- Traditionally, this is the most efficient way to exploit heterogeneity

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
- Fine-grain heterogeneity becomes possible with Pascal / Volta architecture
- Pascal / Volta unified memory
    - CPU-GPU memory coherence
    - System-wide atomics

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

#### Since CUDA 8.0
- Unified memory `cudaMallocManaged(&h_in, in_size)`
- System-wide atomics `old = atomicAdd_system(&h_out[x], inc)`

### Collaborative patterns

<img src="../images/Screenshot 2024-04-27 at 18.30.48.png" alt="Collaborative patterns">

<img src="../images/Screenshot 2024-04-27 at 18.35.20.png" alt="Collaborative patterns">
