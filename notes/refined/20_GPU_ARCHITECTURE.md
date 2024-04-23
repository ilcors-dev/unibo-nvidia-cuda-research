# 20_GPU architecture

> https://www.youtube.com/watch?v=UFD8K-lprbQ

## Overview of GPU Architecture
- GPUs are both vector processors and array processors, a mix of the two. Both types fall under the SIMD (Single Instruction Multiple Data) category.
- **Memory banking** is the division of memory into multiple units to allow more simultaneous data access.

## Limits of SIMD
- SIMD faces limitations if the data being processed is not vectorizable, such as:
  - Linked lists
  - Any algorithm that is not vectorizable, i.e., data elements are not independent.

## Performance Considerations
- Where SIMD is applicable, there is a notable performance gain. However, performance improvements are still bound by Amdahl's Law, which applies to programs only partially parallelized; the speedup is limited by the sequential portion of the program.
- Modern processors include SIMD extensions and can switch between serial and SIMD instruction modes.

## Historical Context: Intel Pentium MMX
### Intel Pentium MMX operations - 90s
- Concept: one instruction operates on multiple data elements simultaneously -> SIMD!
- No VLEN register.
- Opcode determines the data type:
  - 8 -> 8-bit bytes
  - 4 -> 16-bit words
  - 2 -> 32-bit doublewords
  - 1 -> 64-bit quadwords
- STRIDE is always 1.

## Deep Dive into GPUs
- GPUs are essentially SIMD engines under the hood.
- The pipeline operates like a SIMD (array processor) pipeline.
- Programming is not done via SIMD instructions but through **threads**.

### What are threads in GPU terminology?
- In GPU parlance, a thread is a single sequence of programmed instructions that operates on an independent set of data.
- Logical vs. Physical Resources: A thread is not directly a piece of physical hardware. Instead, it is a logical execution unit that the GPU's hardware can manage. Threads are mapped to physical hardware resources when they are scheduled for execution.
- Execution Context: Each thread has its own execution context, which includes:
Registers: Each thread has access to a set of registers.
Local memory (if any): Threads may also have access to small amounts of local memory specific to that thread.
Program Counter: Each thread does not independently possess a physical program counter. Instead, all threads in a warp share the same instruction stream and thus follow the same program counter for their execution path.

### Programming Model vs. Execution Model
- **Programming model**: How the programmer writes the code, can be:
  - Sequential (von Neumann)
  - Data parallel (SIMD)
  - Dataflow
  - Multi-threaded (MIMD, SPMD)
- **Execution model**: How the hardware executes the code
  - Out-of-order execution
  - Vector processor
  - Array processor
  - Dataflow processor
  - Multiprocessor
  - Multithreaded processor

### GPUs as SIMD Machines
- In practice, a GPU is a SIMD machine without the need for SIMD instructions; it is programmed using THREADs (SPMD programming model).
- Each thread executes the same code but on different data.
- Each thread maintains its own context and can be used, reset, and executed independently.
- A set of threads executing the same instruction are dynamically grouped into **WARPs (wavefront)** by the hardware.
- GPUs can be viewed as SIMD machines not exposed to the programmer (SIMT = single instruction multiple threads).

### What are WARPs?
- WARPs are groups of threads that execute the same instruction. Typically, 32 threads are grouped into a WARP.

#### How do they operate
- Uniform Instruction Execution: All threads in a warp execute the same instruction at once but on different pieces of data. This design leverages the data parallelism that GPUs are optimized for, such as in graphics rendering or scientific computations where the same operations are performed over a large set of data elements.
- Handling Divergence: If threads within a warp need to execute different instructions (due to conditional branching, for instance), this causes what's known as "warp divergence." The GPU handles divergence by serially executing each branch path needed by any thread in the warp, while other threads wait (idle). This can lead to inefficiencies and is one of the challenges in optimizing GPU code.

### Warp and Thread Management:

*   **Shared Execution**: Because all threads in a warp execute the same instruction simultaneously, they can be thought of as sharing a single program counter at the level of the warp. This means if threads need to execute different instructions because of conditional branching, the warp must serialize these branches, handling each outcome path one at a time, which can lead to inefficiencies.
*   **Resource Allocation**: The CUDA cores (hardware execution units) do not belong to any specific thread permanently. They are allocated dynamically to threads as warps are scheduled to execute. This means the hardware resources are shared and managed across potentially thousands of threads.

### SIMD vs SIMT Execution Model
- **SIMD**: a single sequential instruction stream -> [VLD, VLD, VADD, VST], VLEN
- **SIMT**: multiple scalar instruction streams -> threads dynamically grouped into **warps**, effectively removing the need for a VLEN, VMASK in SIMD -> [LD, LD, ADD, ST], NumThreads

### Advantages of SIMT
- In SIMT, the programmer can continue to program in the von Neumann model; the hardware takes care of the rest.
- Each thread can be treated separately -> can execute more threads independently aka MIMD processing.
- Can flexibly group threads into warps to maximize the benefits of SIMD.

### Fine-grained Multi-threading of Warps
- Warp of 32 threads.
- If there are 32k iterations & 1 iteration/thread -> 1k warps (each warp has 32 threads).
- Each warp can be interleaved on the same pipeline.

### GPU vs. CPU
- GPUs do not do branch prediction or check on data dependencies.
- A GPU essentially schedules warps in the pipeline.
- The pipeline remains very simple:
  - One instruction per thread at a time (no interlocking).
  - Interleaving warp execution to mask latencies.

### SIMD vs SIMT
- SIMD: VADD A,B -> C
- SIMT: ADD A[tid], B[tid] -> C[tid]
  - tid = threadId
  - The structure can remain the same as a SIMD processor, but using **tid**.

#### Memory Access in GPU Architecture
- The same instruction in different threads uses the **tid** as an index to access different data.
  - <img alt="GPU Memory Access Diagram" src="../images/Pasted image 20240421164115.png" />
- When programming, it is necessary to partition the data across different threads.
- For maximum performance, the memory must have sufficient bandwidth.

### Warp Exposure to Programmers
- Warps are not exposed to the programmer.
- **CPU threads & GPU kernels**:
  - Sequential or minimally parallelized sections on the CPU.
  - High parallelism sections on the GPU: **blocks of threads**.
  - Serial code makes sense to run on the CPU because it is better at it.
    - <img alt="CPU vs GPU code Execution" src="../images/Pasted image 20240421165556.png" />
- GPUs have been very successful also because the code for a CPU is very similar to that for a GPU.
  - <img alt="GPU Architecture Overview" src="../images/Pasted image 20240421165810.png" />
  - <img alt="Detailed GPU Architecture" src="../images/Pasted image 20240421165900.png" />

#### From Blocks to Warps in GPUs
- GPU core = a SIMD pipeline.
  - Streaming processor (SP)
  - Many such SIMD processors
    - Streaming multiprocessor (SM)
      - <img alt="Streaming Multiprocessor Architecture" src="../images/Pasted image 20240421170114.png" />
- Blocks are divided into WARPs.
  - SIMD / SIMT unit (32 threads)
    - <img alt="Warp Configuration" src="../images/Pasted image 20240421170146.png)" />
#### Warp-based SIMD vs. Traditional SIMD
- Traditional SIMD is executed on a single thread.
  - Sequential instruction execution -> lock-step operations in a SIMD instruction.
  - The programming model must be SIMD (no extra threads) -> the software must know the vector length.
- WARP based SIMD, more scalar threads executing in a SIMD-like manner (the same instruction executed by all threads).
  - No lock step.
  - Each thread can be treated individually (different warps) -> the programming model IS NOT SIMD.
    - The software does not need to know the VLEN.
    - Multithreading and dynamic grouping of threads are possible.
  - Scalar ISA.
  => Essentially an SPMD programming model implemented on SIMD hardware.

#### SPMD (Single Program Multiple Data)
- It is a programming model, not an architectural structure.
- Each functional unit executes the same procedure but on different data.
  - The procedures can be synchronized at certain points in the program (e.g., barriers).
- Multiple execution streams run the same program.
  - Each program/procedure:
    - Works on different data.
    - Can execute a different control-flow path at runtime (!).

#### Flexibility of SIMT in Grouping Threads into Warps
- <img alt="SIMT Thread Grouping" src="../images/Pasted image 20240421171233.png" />
- <img alt="Dynamic Warp Formation" src="../images/Pasted image 20240421171318.png" />
- If there are many threads, they can be:
  - Grouped more threads that are at the same PC.
  - Grouped into a single warp dynamically.
  - The result is a reduction in "divergence" -> SIMD utilization increases.
    - SIMD utilization: the fraction of SIMD lanes executing a useful operation (e.g., executing an active thread).
- The idea is to merge threads that are executing the same instruction (i.e., at the same PC) after executing a branch.
- Essentially, create new warps with waiting warps to improve SIMD utilization.
  - <img alt="Advanced Warp Management" src="../images/Pasted image 20240421172104.png" />
  - In a complex example...
    - <img alt="Complex Warp Management" src="../images/Pasted image 20240421172335.png" />
  - Can threads be moved to different lines of the pipeline? -> NO

#### Example of a GPU - NVIDIA GeForce GTX 285
- 240 stream processors.
- SIMT execution.
- 30 cores.
- 8 SIMD FU per core.
  - By today's standards, it is SMALL.
    - <img alt="NVIDIA GeForce GTX" src="../images/Pasted image 20240421172837.png" />
  - 32 threads in a warp & 32 warps -> 1024 threads that can be used thus requiring 64 KB of storage for the threads (registers).
    - <img alt="Thread and Warp" src="../images/Pasted image 20240421173033.png" />

#### Tensor Cores
- What are TENSOR CORES -> essentially, they are cores specialized for performing matrix operations in an optimized manner. They are specialized cores.
- Even in tensor cores, there are SIMD processors.

### SIMD vs SIMT Summary
SIMD (Single Instruction, Multiple Data) and SIMT (Single Instruction, Multiple Threads) architectures both exploit data parallelism but in different ways, leading to distinct limitations and advantages in various computing scenarios.

### Limits of SIMD Architecture

1.  **Data Dependency and Divergence:**
    
    *   SIMD units require operations to be uniform and synchronized across all data elements within a vector. When data dependencies or divergent execution paths (like conditional branches) occur, the SIMD model struggles because all processing units must execute the same operation at the same time. Divergent paths lead to idle processing units, which reduces efficiency.
2.  **Memory Alignment and Access Patterns:**
    
    *   Efficient SIMD execution requires data to be well-aligned in memory. Non-aligned data accesses can lead to performance penalties. Additionally, SIMD prefers continuous memory access patterns, and irregular patterns (like indirect addressing or sparse data access) can significantly reduce performance.
3.  **Overhead of Managing Vector Lengths:**
    
    *   SIMD architectures are often dependent on fixed vector sizes (vector length). Programs must manage operations on data sizes that do not perfectly fit into these lengths, requiring additional logic to handle leftover data elements.
4.  **Limited Flexibility in Programming:**
    
    *   Programming for SIMD can be more complex, especially when dealing with algorithms that do not naturally map to vector operations. Programmers often need to restructure code extensively or use intrinsic functions to make full use of SIMD capabilities, increasing the complexity of development and maintenance.
5.  **Hardware Utilization:**
    
    *   SIMD units can be underutilized if the application does not have enough parallelism to fill the SIMD lanes, which is a common issue with applications having varying degrees of parallelizable workloads.

### Limits of SIMT Architecture

1.  **Thread Divergence:**
    
    *   In SIMT architectures like those in GPUs, thread divergence remains a challenge. When threads within the same warp choose different execution paths due to conditional branching, the GPU must execute each path serially, which diminishes the benefits of parallel execution.
2.  **Resource Contention:**
    
    *   SIMT architectures can suffer from resource contention when multiple threads attempt to access memory or other shared resources simultaneously, leading to potential bottlenecks and reduced performance.
3.  **Complex Memory Hierarchy Management:**
    
    *   SIMT architectures typically involve a complex memory hierarchy including local, shared, and global memory. Efficiently managing data across these memories is crucial for performance but adds to programming complexity.
4.  **Synchronization Costs:**
    
    *   While SIMT allows each thread to execute independently, synchronization mechanisms are necessary to coordinate threads, especially when they need to share data. Implementing synchronization correctly is crucial but can be error-prone and impact performance.
5.  **Software Complexity:**
    
    *   Although SIMT provides more flexibility than SIMD, it still requires careful programming to avoid performance pitfalls such as warp divergence and inefficient memory access patterns. The need to manage execution at the thread level can complicate software design and optimization.

#### Comparison and Context

While SIMD is more restrictive due to its need for uniform operations across all data elements, SIMT provides more flexibility by allowing each thread to execute independently. However, both architectures need careful management of memory access patterns and data alignment to prevent performance degradation. The choice between SIMD and SIMT typically depends on the specific requirements and constraints of the application, as well as the underlying hardware's capabilities.
