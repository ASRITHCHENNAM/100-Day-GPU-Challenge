#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

__global__ void MatrixAdd_B(const float* matA, const float* matB, float* matC, int size) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= size || col >= size) return;

    matC[row * size + col] = matA[row * size + col] + matB[row * size + col];
}

int main() {
    const int size = 10;
    float *matA, *matB, *matC;

    // Host memory allocation
    matA = (float *)malloc(size * size * sizeof(float));
    matB = (float *)malloc(size * size * sizeof(float));
    matC = (float *)malloc(size * size * sizeof(float));

    // Initialize matrices
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            matA[i * size + j] = 1.0f;
            matB[i * size + j] = 2.0f;
            matC[i * size + j] = 0.0f;
        }
    }

    float *d_matA, *d_matB, *d_matC;
    cudaMalloc((void **)&d_matA, size * size * sizeof(float));
    cudaMalloc((void **)&d_matB, size * size * sizeof(float));
    cudaMalloc((void **)&d_matC, size * size * sizeof(float));

    cudaMemcpy(d_matA, matA, size * size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_matB, matB, size * size * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockDim(32, 16);
    dim3 gridDim(ceil(size / 32.0f), ceil(size / 16.0f));

    MatrixAdd_B<<<gridDim, blockDim>>>(d_matA, d_matB, d_matC, size);
    cudaDeviceSynchronize();

    cudaMemcpy(matC, d_matC, size * size * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Matrix C (Result):\n");
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            printf("%.2f ", matC[i * size + j]);
        }
        printf("\n");
    }

    printf("\nMatrix A:\n");
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            printf("%.2f ", matA[i * size + j]);
        }
        printf("\n");
    }

    printf("\nMatrix B:\n");
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            printf("%.2f ", matB[i * size + j]);
        }
        printf("\n");
    }

    // Free device and host memory
    cudaFree(d_matA);
    cudaFree(d_matB);
    cudaFree(d_matC);
    free(matA);
    free(matB);
    free(matC);

    return 0;
}
