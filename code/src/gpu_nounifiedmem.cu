#include <iostream>
#include <vector>
#include <chrono>

__global__ void saxpy(int n, float a, float* x, float* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = a * x[i] + y[i];
    }
}

int main() {
    int n = 500000000; // Smaller for testing
    float a = 2.0;

    std::vector<float> x(n, 1.0);
    std::vector<float> y(n, 1.0);

    float* x_dev, * y_dev;

    cudaError_t cudaStatus;

    // Allocate the required memory on the GPU
    cudaStatus = cudaMalloc(&x_dev, n * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaMalloc failed for x_dev!" << std::endl;
        return 1;
    }

    cudaStatus = cudaMalloc(&y_dev, n * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaMalloc failed for y_dev!" << std::endl;
        cudaFree(x_dev);
        return 1;
    }

    // Transfer the data from the CPU to the GPU
    cudaMemcpy(x_dev, x.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(y_dev, y.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    // How many threads there may be in a block
    int blockSize = 256;

    // How many blocks are necessary to cover all the datapoints
    int numBlocks = (n + blockSize - 1) / blockSize;

    std::cout << "for " << n << " elements, " << numBlocks << " blocks are needed" << std::endl;

    // Measure performance
    auto start = std::chrono::high_resolution_clock::now();

    // Launch the kernel
    saxpy<<<numBlocks, blockSize>>>(n, a, x_dev, y_dev);
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "saxpy launch failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        cudaFree(x_dev);
        cudaFree(y_dev);
        return 1;
    }

    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    // Copy the result back to the host (CPU)
    cudaMemcpy(y.data(), y_dev, n * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms\n";

    cudaFree(x_dev);
    cudaFree(y_dev);

    return 0;
}
