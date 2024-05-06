# 19_SIMD architecture

> https://www.youtube.com/watch?v=gkMaO3yJMz0

- [19\_SIMD architecture](#19_simd-architecture)
    - [SIMD vs Out-of-Order Execution:](#simd-vs-out-of-order-execution)
      - [SIMD](#simd)
      - [Out-of-Order Execution](#out-of-order-execution)
      - [Comparison](#comparison)
      - [Data Processing in SIMD Architectures:](#data-processing-in-simd-architectures)
      - [Example of Operational Differences:](#example-of-operational-differences)
    - [Vector Stride and Pipeline:](#vector-stride-and-pipeline)
      - [Advantages of Vector Processors:](#advantages-of-vector-processors)
      - [Disadvantages of Vector Processors:](#disadvantages-of-vector-processors)
    - [Amdahl's Law](#amdahls-law)
    - [Vector Registers and Functional Units:](#vector-registers-and-functional-units)
      - [Loading / Storing vectors from \& to memory](#loading--storing-vectors-from--to-memory)
        - [How do we achieve this with a memory that takes more than 1 cycle to access?](#how-do-we-achieve-this-with-a-memory-that-takes-more-than-1-cycle-to-access)
      - [Memory banking](#memory-banking)
        - [An example](#an-example)
    - [Vector Chaining](#vector-chaining)
    - [Vector stripmining](#vector-stripmining)
    - [Gather / Scatter operations](#gather--scatter-operations)
      - [Gather example](#gather-example)
      - [Scatter example](#scatter-example)
    - [Conditional operations](#conditional-operations)
      - [1 Example](#1-example)
      - [2 Example](#2-example)
      - [Implementation of masked operations](#implementation-of-masked-operations)
      - [Simple](#simple)
      - [Density-Time](#density-time)
    - [Array vs Vector processors, again](#array-vs-vector-processors-again)
      - [Vector instruction level parallelism](#vector-instruction-level-parallelism)
    - [In summary](#in-summary)


*   **SIMD (Single Instruction, Multiple Data)** architecture is a type of parallel computing architecture that performs the same operation on multiple data points simultaneously.
*   In a SIMD architecture, multiple processing elements execute the same operation on different pieces of data at the same time, thereby achieving data-level parallelism.
*   Two common forms of SIMD architectures are **vector processors** and **array processors**.
    *   **Array Processors**: In array processors, instructions operate on multiple data elements simultaneously using different processing elements. This form of SIMD architecture is characterized by having multiple processing elements that perform the same operation at the same time across different sets of data.
    *   **Vector Processors**: Vector processors use a single instruction to perform operations on data stored in vector registers. These registers can hold multiple data elements, and a single instruction can operate on all these elements sequentially or in parallel, depending on the architecture.
    <img src="../images/Screenshot 2024-04-24 at 21.36.37.png" alt="SIMD architecture"/>
*   A **GPU** represents a hybrid of array and vector processor architectures.
*   The **SIMD (Single Instruction, Multiple Data) architecture** utilizes registers that store vectors, which are arrays of N elements of M bits each, instead of single scalar values.

### SIMD vs Out-of-Order Execution:
#### SIMD
**Pros:**

*  **Parallel Data Processing**: SIMD allows a single instruction to simultaneously perform the same operation on multiple data points. This is particularly beneficial for tasks that can be parallelized at the data level, such as vector and matrix operations common in graphics, multimedia processing, and scientific computations.
*  **Efficiency**: By executing one instruction across many data elements, SIMD reduces the instruction cycle overhead, leading to better utilization of processing power for suitable tasks.
*  **Energy Efficiency**: Executing multiple operations simultaneously can be more energy-efficient than processing them serially, which is particularly valuable in mobile and embedded applications.

**Cons:**
*   **Specialized Use Cases**: SIMD is most effective when operations are uniform and can be applied simultaneously to multiple pieces of data. It is less useful for general-purpose computing where each data element might require different operations.
*   **Programming Complexity**: Utilizing SIMD often requires explicit programming effort, including managing data alignment and handling cases where data sizes do not match SIMD register widths.

#### Out-of-Order Execution
**Pros:**
*   **Increased CPU Utilization**: Out-of-order execution allows CPUs to make more efficient use of processor cycles by executing instructions as resources become available, rather than adhering strictly to the program order.
*   **Latency Hiding**: It helps in hiding the latency of slower operations like memory fetch and long arithmetic operations by rearranging the execution order to keep the execution units busy.
*   **General Purpose**: This approach is beneficial for a wide range of applications, as it automatically optimizes execution without special programming requirements.

**Cons:**
*   **Complexity**: The CPU design becomes significantly more complex with out-of-order execution, which can increase the cost and power consumption of the processor.
*   **Diminishing Returns**: For highly sequential code or when dependencies between instructions are too tight, the advantages of out-of-order execution can be limited.

#### Comparison
*   SIMD and out-of-order execution are complementary techniques that address different aspects of parallelism and efficiency in computing.
*   SIMD is well-suited for tasks with data-level parallelism, while out-of-order execution is more effective for improving the utilization of CPU resources and handling complex instruction dependencies.

#### Data Processing in SIMD Architectures:

*   **Vector processors** handle data serially from vector registers:
    *   Each cycle processes one element from the vector through a functional unit (FU) until all vector elements are processed.
*   **Array processors** handle entire vectors simultaneously:
    *   Performs operations like LOAD, ADD, and STORE on all elements at the same time, necessitating multiple FUs.

#### Example of Operational Differences:

*   **To add two vectors of 32 elements:**
    *   An **array processor** would perform 32 LOADs, 32 ADDs, and 32 STOREs concurrently, requiring 32 FUs.
    *   A **vector processor** would sequentially perform:
        *   LOAD0, then LOAD1 and ADD0, followed by LOAD2, ADD1, and STORE0, continuing in this fashion, utilizing fewer resources but at a slower rate.

<img src="../images/Screenshot 2024-04-24 at 21.36.37.png" alt="SIMD architecture"/>

### Vector Stride and Pipeline:

*   **Vector stride** refers to the memory distance between consecutive elements in a vector. A stride of 1, where elements are contiguous, is optimal for performance.
*   Vector instructions benefit from longer pipelines due to:
    *   Lack of data dependencies, eliminating dependency checks.
    *   Absence of pipeline interlocks and control flows between vector elements.
    *   Predictable strides that simplify data prefetching and caching.

<img src="../images/Screenshot 2024-04-24 at 21.45.34.png" alt="SIMD architecture"/>

#### Advantages of Vector Processors:

* **High Data Throughput**: Processes multiple data elements simultaneously, enhancing performance for data-intensive tasks.
* **Efficient Use of Resources**: Amortizes instruction decoding overhead across multiple data elements, reducing the total number of instructions and saving processing power.
    * Reduces instruction fetch bandwidth requirements
    * Amortizes instruction fetch & control overhead over many data -> leads to high energy efficiency per operation
    * Fewer loops & therefore fewer branches
* **Reduced Power Consumption**: Fewer instructions and efficient parallel processing lower energy usage, beneficial in energy-sensitive environments.
* **Simplified Programming for Parallelism**: Inherent hardware parallelism simplifies coding for applications suited to vector operations, like matrix computations.
* **Optimized Performance for Specific Applications**: Excels in applications involving large vectors or matrices, significantly outperforming scalar processors in these cases.
* **Pipelining**: Employs deep pipelining to process different stages of vector operations simultaneously, boosting instruction throughput.
    * No intra-vector dependencies -> no hardware interlocking -> no pipeline stalls to prevent hazards.
    * No control flow within a vector
    * Predictable strides for efficient data prefetching and caching. 
* **Scalability**: Scales effectively with added hardware resources, handling larger vectors or more operations without major power or complexity increases.
* **Handling Large Data Sets**: Ideal for modern applications involving large datasets, such as machine learning and big data analysis.
    * predictable memory access pattern 

#### Disadvantages of Vector Processors:

*   Dependence on parallelizable data, suitable only for regular parallelism.
    * Becomes very inefficient if parallelism is irregular   
*   Ineffective for data structures like linked lists, where the next element's position is unpredictable.
*   Memory bandwidth can limit performance due to the large volume of data accessed per instruction.

To quote:
> To program a vector machine, the compiler or hand coder must make the data structures in the code fit nearly exactly the regular structure built into the hardware. That's hard to do in first place, and just as hard to change. One tweak, and the low-level code has to be rewritten by a very smart and dedicated programmer who knows the hardware and often the subtleties of the application area.

### Amdahl's Law
-  Amdahl's Law is a fundamental principle in parallel computing that quantifies the potential speedup achievable by parallelizing a computation.
-  It states that the overall speedup of a program is limited by the fraction of the program that cannot be parallelized.
-  The law is expressed as a formula:
    -  **Speedup = 1 / [(1 - P) + (P / N)]**
    -  Where:
        -  **P** is the fraction of the program that can be parallelized.
        -  **N** is the number of processors or cores available for parallel execution.

### Vector Registers and Functional Units:

*   **Vector registers** store multiple M-bit elements:
    *   Controlled by **vector control registers**:
        *   **VLEN** (vector length)
        *   **VSTR** (vector stride)
        *   **VMASK** (mask for conditional operations, enabling selective processing based on specified conditions, such as VMASK\[i\] = (V\_k\[i\] == 0)).
*   **Vector functional units (FU)** can be deeply pipelined, exploiting the independent processing of elements to enhance throughput.

#### Loading / Storing vectors from & to memory
- Multiple elements need to be loaded/stored from/to memory in a single instruction.
- The elements are separated by a fixed stride.

##### How do we achieve this with a memory that takes more than 1 cycle to access?
- Bank the memory
- Interleave the memory accesses across the banks

#### Memory banking
- Memory is divided into multiple banks, that can be accessed simultaneously.
- Banks share address and data buses (to reduce memory chip pins required)
- This way can start and complete one bank access per cycle
- Can sustain N concurrent accesses if all N go to different banks

<img src="../images/Screenshot 2024-04-25 at 15.13.15.png" alt="Banking"/>

##### An example
- For i = 0 to 49
- C[i] = (A[i] + B[i]) / 2

```assembly
    MOVI R0 = 50            1
    MOVA R1 = A             1
    MOVA R2 = B             1
    MOVA R3 = C             1
X:
    LD R4 = MEM[R1++]       11  ;autoincrement addressing
    LD R5 = MEM[R2++]       11
    ADD R6 = R4 + R5        4
    SHFR R7 = R6 >> 1       1
    ST MEM[R3++] = R7       11
    DECBNZ R0, X            2   ;decrement and branch if NZ
```

-> 304 dynamic instructions
- Assuming scalar execution time in-order with 1 bank
    - first 2 loads in the loop cannot be pipelined -> 2*11 cycles
    - 4+50*40 = 2004 cycles
- Assuming scalar execution time in-order processor with 1 bank with 2 memory ports (which can be used concurrently) or 2 banks
    - first 2 loads in the loop can be pipelined -> 1 + 11 cycles
    - 4+50*30 = 1504 cycles

- Since the loop can be vectorized (since each iteration is independent of any other)

```assembly
    MOVI VLEN = 50              1
    MOVI VSTR = 1               1
    VLD V0 = A                  11 + VLEN – 1
    VLD V1 = B                  11 + VLEN – 1
    VADD V2 = V0 + V1           4 + VLEN – 1
    VSHFR V3 = V2 >> 1          1 + VLEN – 1
    VST C = V3                  11 + VLEN – 1
```

- Assuming 
    - **no chaining** (no vector data forwarding -> output of a vector FU cannot be used as input to another vector FU in the same cycle)
    - entire vector registers need to be ready before the operation can start
    - 1 memory port (one address generator) per bank
    - 16 banks (word-interleaved: consecutive elements of an array are stored in consecutive banks)

-> 1+1+11+49+11+49+4+49+1+49+11+49 = 285 cycles

<img src="../images/Screenshot 2024-04-25 at 15.27.40.png" alt="example vector"/>

- why 16 banks?
    - 11-cycle memory access latency
    - having 16 (> 11) banks ensures there are enough banks to overlap enough memory operations to cover memory latency 
- assuming unit stride of 1 (VSTR = 1)
- what if stride > 1?
    - how can we endure we can access 1 element per cycle when memory latency is 11 cycles? **VECTOR CHAINING**

### Vector Chaining
- **Vector chaining** allows the output of one vector operation to be used as the input for the next operation in the same cycle.

<img src="../images/Screenshot 2024-04-25 at 15.30.40.png" alt="vector chaining"/>

- In the previous example, assuming vector chaining is possible
    - 1+1+11+49+11+49+49+11 = 182 cycles

### Vector stripmining
- Occurs, for example, when # data elements > # elements in a vector register
- The loop can be broken down into # elements in the vector register
    - E.g. 527 data elements, 64-element VREGs
    - 8 iteration where VLEN = 64
    - 1 iteration where VLEN = 15 (the VLEN must be changed at runtime)

### Gather / Scatter operations
- Occurs when the vector data is not stored in a strided fashion (irregular memory access to a vector)
- Indirection will be used to combine / pack elements into vector registers
#### Gather example
```c
for (i=0; i<N; i++)
    A[i] = B[i] + C[D[i]]
```

```
LV vD, rD       # Load indices in D vector
LVI vC, rC, vD  # Load indirect from rC base <- **Gathering**
LV vB, rB       # Load B vector
ADDV.D vA,vB,vC # Do add
SV vA, rA       # Store result
```

#### Scatter example

```
Index Vector                Data Vector (to Store)              Stored vector (in memory)
0                           3.14                                Base+0  3.14
2                           6.5                                 Base+1  X
6                           71.2                                Base+2  6.5
7                           2.71                                Base+3  X
                                                                Base+4  X
                                                                Base+5  X
                                                                Base+6  71.2
                                                                Base+7  2.71
```

### Conditional operations
- **Masking** is used to conditionally execute operations on vector elements
- VMask register is a bit mask determining which data element should not be acted upon

#### 1 Example
```c
for (i=0; i<N; i++)
    if (a[i] != 0) then b[i]=a[i]*b[i]
```

```
VLD V0 = A
VLD V1 = B
VMASK = (V0 != 0)
VMUL V1 = V0 * V1
VST B = V1
```
- Named, predicated execution in some architectures. Execution is predicated on mask bit

#### 2 Example
```c
for (i = 0; i < 64; ++i)
    if (a[i] >= b[i]) 
        c[i] = a[i]
    else 
        c[i] = b[i]
```

```
A       B       VMASK
1       2       0
2       2       1
3       2       1
4       10      0
-5      -4      0
0       -3      1
6       5       1
-7      -8      1
```

#### Implementation of masked operations
#### Simple
- Execute all N operations, turn off result WB according to the mask
#### Density-Time
- scan mask vector and only execute elements with non-zero masks

### Array vs Vector processors, again
- Array vs vector processors distinction is a "purist's" distinction
- Most "modern" SIMD processors are hybrids of the two
    - exploiting data parallelism in both time and space
    - [GPUs](./20_GPU_ARCHITECTURE.md) are a prime example
    
<img src="../images/Screenshot 2024-04-25 at 15.49.27.png" alt="array vs vector"/>

#### Vector instruction level parallelism
- Can overlap execution of multiple vector instructions
    - Example, machine has 32 elements per vector register and 8 lanes
    - Example, with 24 operations/cycle (on steady state) while issuing 1 vector instruction/cycle
> What is a lane? Refer to the individual execution units within a processor that can independently execute operations on different elements of a vector simultaneously. Each lane operates on a separate data element but under the same instruction.
<img src="../images/Screenshot 2024-04-25 at 15.52.55.png" alt="vector instruction level parallelism"/>

### In summary
- Vector / SIMD machines are good at exploiting regular data-level parallelism
    - Same operation performed on many data elements
    - Improve performance, simplify design (no intra-vector dependencies)
- Performance improvement limited by vectorizability of code
    - Scalar operations limit vector machine performance
    - Amdahl's law
- Many existing ISAs include (vector-like) SIMD operations
    - Intel MMX / SSEn / AVX, PowerPC AltiVec, ARM Advanced SIMD (NEON)
