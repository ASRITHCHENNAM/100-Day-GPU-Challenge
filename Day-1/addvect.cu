#include <iostream>
#include <cmath>        // For ceil
#include <cuda_runtime.h>

// Kernel to add two vectors
__global__ void vectorAdd(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
        printf("Thread %d: A[%d] = %.2f, B[%d] = %.2f, C[%d] = %.2f\n",
               i, i, A[i], i, B[i], i, C[i]);
    }
}

int main() {
    const int N = 10;
    float A[N], B[N], C[N];

    // Initialize host arrays A and B
    for (int i = 0; i < N; ++i) {
        A[i] = i * 1.0f;
        B[i] = (N - i) * 1.0f;
    }

    // Print input vectors
    std::cout << "Input Vector A: ";
    for (int i = 0; i < N; ++i) std::cout << A[i] << " ";
    std::cout << "\n";

    std::cout << "Input Vector B: ";
    for (int i = 0; i < N; ++i) std::cout << B[i] << " ";
    std::cout << "\n";

    // Device pointers
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));
    cudaMalloc(&d_c, N * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_a, A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, B, N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    int blocksize = 256;
    int gridsize = (N + blocksize - 1) / blocksize;
    std::cout << "Launching kernel with grid size " << gridsize << " and block size " << blocksize << "\n";
    vectorAdd<<<gridsize, blocksize>>>(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();  // Wait for GPU to finish

    // Copy result back to host
    cudaMemcpy(C, d_c, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Print result vector C
    std::cout << "Output Vector C = A + B: ";
    for (int i = 0; i < N; ++i)
        std::cout << C[i] << " ";
    std::cout << "\n";

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}


