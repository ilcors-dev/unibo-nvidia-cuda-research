# 19_SIMD architecture

> https://www.youtube.com/watch?v=gkMaO3yJMz0
*   Historically, **vector processors** were more common because replicating multiple functional units (FUs) on a chip was spatially challenging. Today, advancements in chip design have mitigated these space constraints, making **array processors** more prevalent.
*   A **GPU** represents a hybrid of array and vector processor architectures.
*   The **SIMD (Single Instruction, Multiple Data) architecture** utilizes registers that store vectors, which are arrays of N elements of M bits each, instead of single scalar values.

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

#### Vector Stride and Pipeline:

*   **Vector stride** refers to the memory distance between consecutive elements in a vector. A stride of 1, where elements are contiguous, is optimal for performance.
*   Vector instructions benefit from longer pipelines due to:
    *   Lack of data dependencies, eliminating dependency checks.
    *   Absence of pipeline interlocks and control flows between vector elements.
    *   Predictable strides that simplify data prefetching and caching.

#### Disadvantages of Vector Processors:

*   Dependence on parallelizable data, suitable only for regular parallelism.
*   Ineffective for data structures like linked lists, where the next element's position is unpredictable.
*   Memory bandwidth can limit performance due to the large volume of data accessed per instruction.

#### Vector Registers and Functional Units:

*   **Vector registers** store multiple M-bit elements:
    *   Controlled by **vector control registers**:
        *   **VLEN** (vector length)
        *   **VSTR** (vector stride)
        *   **VMASK** (mask for conditional operations, enabling selective processing based on specified conditions, such as VMASK\[i\] = (V\_k\[i\] == 0)).
*   **Vector functional units (FU)** can be deeply pipelined, exploiting the independent processing of elements to enhance throughput.

### Practical Application:

*   Ideal for tasks with high data parallelism such as scientific computations and multimedia applications, offering energy efficiency but potentially at a slower rate compared to array processors.
*   **Extensive Use in Graphics Rendering**: SIMD architectures are extensively used in the field of graphics rendering where operations over large sets of pixels or vertices are performed simultaneously. This application leverages the architecture's ability to perform the same operation on multiple data points efficiently, significantly speeding up the rendering process.
*   **Applications in Machine Learning**: With the rise of machine learning and deep learning, SIMD architectures have found a new role in accelerating operations in neural networks, particularly in the training and inference stages. The parallel processing capabilities allow for rapid matrix and vector calculations that are fundamental to these technologies.
*   **Telecommunications**: SIMD processors are crucial in digital signal processing (DSP) used in telecommunications. They process multiple data samples simultaneously, enhancing the speed and efficiency of data transmission and signal processing tasks such as filtering, modulation, and coding.
*   **Bioinformatics and Computational Biology**: SIMD architectures facilitate faster processing of large datasets common in genomics and proteomics, helping to align sequences and analyze genetic data more swiftly than traditional scalar processors could achieve.
