#include <iostream>

__global__ void vectorMatrixMult(const float* matrixA, const float* vectorB, float* resultVector, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        float sum = 0.0f;
        for (int j = 0; j < size; j++) {
            sum += matrixA[i * size + j] * vectorB[j];
        }
        resultVector[i] = sum;
    }
}

int main() {
    const int size = 10;
    float *matrixA, *vectorB, *resultVector;

    // Allocate host memory
    matrixA = (float *)malloc(size * size * sizeof(float));
    vectorB = (float *)malloc(size * sizeof(float));
    resultVector = (float *)malloc(size * sizeof(float));

    // Initialize matrixA and vectorB
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            matrixA[i * size + j] = 1.0f;
        }
        vectorB[i] = 2.0f;
        resultVector[i] = 0.0f;
    }

    float *d_matrixA, *d_vectorB, *d_resultVector;
    cudaMalloc(&d_matrixA, size * size * sizeof(float));
    cudaMalloc(&d_vectorB, size * sizeof(float));
    cudaMalloc(&d_resultVector, size * sizeof(float));

    cudaMemcpy(d_matrixA, matrixA, size * size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vectorB, vectorB, size * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    vectorMatrixMult<<<gridSize, blockSize>>>(d_matrixA, d_vectorB, d_resultVector, size);

    cudaDeviceSynchronize();

    cudaMemcpy(resultVector, d_resultVector, size * sizeof(float), cudaMemcpyDeviceToHost);

    // Print matrixA
    printf("Matrix A:\n");
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            printf("%.2f ", matrixA[i * size + j]);
        }
        printf("\n");
    }

    // Print resultVector
    printf("Result Vector (C = A * B):\n");
    for (int i = 0; i < size; i++) {
        printf("%.2f ", resultVector[i]);
    }
    printf("\n");

    // Print vectorB
    printf("Vector B:\n");
    for (int i = 0; i < size; i++) {
        printf("%.2f ", vectorB[i]);
    }
    printf("\n");

    // Free device and host memory
    cudaFree(d_matrixA);
    cudaFree(d_vectorB);
    cudaFree(d_resultVector);
    free(matrixA);
    free(vectorB);
    free(resultVector);

    return 0;
}
