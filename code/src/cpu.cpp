#include <iostream>
#include <vector>
#include <chrono>

// SAXPY operation: Y = a * X + Y
void saxpy(int n, float a, std::vector<float>& x, std::vector<float>& y) {
    for (int i = 0; i < n; ++i) {
        y[i] = a * x[i] + y[i];
    }
}

int main() {
    int n = 500000000; // Adjust the size as needed
    float a = 2.0; // Scalar multiplier

    // Vectors x and y
    std::vector<float> x(n, 1.0); // Initialize x with 1.0
    std::vector<float> y(n, 2.0); // Initialize y with 2.0

    // Measure performance
    auto start = std::chrono::high_resolution_clock::now();

    // Perform SAXPY operation
    saxpy(n, a, x, y);

    auto end = std::chrono::high_resolution_clock::now();

    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms\n";

    return 0;
}
