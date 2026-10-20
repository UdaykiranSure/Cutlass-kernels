#include <cuda_runtime.h>
#include <cstdio>
#include <iostream>
#include <vector>

#include "timer.h"

#define BLOCK_DIM 1024
#define COARSE_FACTOR 4

// Recution using segementation + memory coalescing + thread coarsening


__global__ void reduction_kernel(float* in, float* partial_sums, unsigned int N){
    unsigned int segment = blockIdx.x * blockDim.x*2*COARSE_FACTOR;
    unsigned int i = segment + threadIdx.x;

    __shared__ float in_s[BLOCK_DIM*sizeof(float)];
    int sum = 0.0f;
    for(int tile = 0; tile < 2*COARSE_FACTOR; tile += 1){
        sum += in[i + tile*BLOCK_DIM];
    }
    in_s[threadIdx.x] = sum;
    __syncthreads();
    
    for(int stride = BLOCK_DIM/2; stride >= 1; stride /= 2){
        if(threadIdx.x < stride){
            in_s[threadIdx.x] += in_s[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if(threadIdx.x == 0){
        partial_sums[blockIdx.x] = in_s[threadIdx.x];
    }

}




float reduction_gpu(float *in, unsigned int N){
    Timer timer;

    //allocate gpu memory
    startTime(&timer);
    float *input_d;
    cudaMalloc(&input_d, N*sizeof(float));
    stopTime(&timer);
    printElapsedTime(timer, "Cuda Allocation Time");

    //Copy Host to Device
    startTime(&timer);
    cudaMemcpy(input_d, in, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copy to GPU Time");

    //allocate partial sums
    startTime(&timer);
    const unsigned int numThreadsPerBlock = BLOCK_DIM;
    const unsigned int numElementsPerBlock = 2 * BLOCK_DIM * COARSE_FACTOR;
    unsigned int numBlocks = (N + numElementsPerBlock - 1)/numElementsPerBlock;
    float *partialSums = (float*) malloc(numBlocks*sizeof(float));
    float *partialSums_d;
    cudaMalloc(&partialSums_d, numBlocks*sizeof(float));
    stopTime(&timer);
    printElapsedTime(timer, "allocate partial sums");


    //Kernel Execution
    startTime(&timer);
    reduction_kernel<<<numBlocks, numThreadsPerBlock>>>(input_d, partialSums_d, N);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Kernel Time:");

    //copy data from gpu
    startTime(&timer);
    cudaMemcpy(partialSums, partialSums_d, N* sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copy from GPU Time");

    // Free GPU Time
    startTime(&timer);
    cudaFree(input_d);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Deallocation Time");

    int sum = 0.0;
    for(int i = 0; i < numBlocks; ++i){
        sum += partialSums[i];
    }
    return sum;

}


int main() {
    const unsigned int N = std::pow(2, 26);

    std::vector<float> input(N);


    // Initialize input matrix
    for (unsigned int i = 0; i < N; ++i){
            input[i] = 1.0f;
    }


    // Run GPU convolution
    reduction_gpu(
        input.data(),
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