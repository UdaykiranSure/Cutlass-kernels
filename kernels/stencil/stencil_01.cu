#include <cuda_runtime.h>
#include <cstdio>
#include <iostream>
#include <vector>

#include "timer.h"

#define BLOCK_DIM 8




__global__ void stencil_kernel(float *in, float *out, unsigned int N){
    int i = blockIdx.z * blockDim.z + threadIdx.z;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    
    float c0 = 1;
    float c1 = 1;

    if( i >=0 && i < N && j >= 0 && j < N && k >= 0 && k < N){
        out[i *N*N + j *N + k] = c0 * in[i * N*N + j*N + k] + 
                                 c1 * ( in[(i-1) * N*N + j*N + k] + 
                                        in[(i+1) * N*N + j*N + k] +
                                        in[i * N*N + (j-1)*N + k] + 
                                        in[i * N*N + (j+1)*N + k] + 
                                        in[i * N*N + j*N + (k-1)] + 
                                        in[i * N*N + j*N + (k+1)] );


    }

}


void stencil_gpu(float *in, float *out, unsigned int N){
    Timer timer;

    //allocate gpu memory
    startTime(&timer);
    float *input, *output;
    cudaMalloc(&input, N*N*N*sizeof(float));
    cudaMalloc(&output, N*N*N*sizeof(float));
    stopTime(&timer);
    printElapsedTime(timer, "Cuda Allocation Time");

    //Copy Host to Device
    startTime(&timer);
    cudaMemcpy(input, in, N * N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copy to GPU Time");

    //Kernel Execution
    startTime(&timer);
    dim3 numThreadsPerBlock(BLOCK_DIM, BLOCK_DIM, BLOCK_DIM);
    dim3 numBlocks((N + BLOCK_DIM - 1)/BLOCK_DIM,(N + BLOCK_DIM - 1)/BLOCK_DIM, (N + BLOCK_DIM - 1)/BLOCK_DIM);
    stencil_kernel<<<numBlocks, numThreadsPerBlock>>>(input, output, N);
    stopTime(&timer);
    printElapsedTime(timer, "Kernel Time:");

    //copy data from gpu
    startTime(&timer);
    cudaMemcpy(out, output, N * N * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copy from GPU Time");

    // Free GPU Time
    startTime(&timer);
    cudaFree(input);
    cudaFree(output);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Deallocation Time");

}


int main() {
    const unsigned int N = 1024;

    std::vector<float> input(N * N * N);
    std::vector<float> output(N * N * N, 0.0f);


    // Initialize input matrix
    for (unsigned int depth = 0;depth < N; ++depth){
        for (unsigned int row = 0; row < N; ++row) {
            for (unsigned int col = 0; col < N; ++col) {
                input[depth * N * N + row * N + col] = 1.0f;
            }
        }
    }


    // Run GPU convolution
    stencil_gpu(
        input.data(),
        output.data(),
        N
    );

    // // Print a small section of the output
    // std::cout << "\nOutput (top-left 5x5):\n";

    // for (unsigned int row = 0; row < 5 && row < height; ++row) {
    //     for (unsigned int col = 0; col < 5 && col < width; ++col) {
    //         std::cout << output[row * width + col] << "\t";
    //     }
    //     std::cout << '\n';
    // }

    return 0;
}