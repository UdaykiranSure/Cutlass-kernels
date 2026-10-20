#include <cuda_runtime.h>
#include <cstdio>
#include <iostream>
#include <vector>

#include "timer.h"

#define BLOCK_DIM 16
#define IN_TILE_DIM BLOCK_DIM
#define OUT_TILE_DIM IN_TILE_DIM-2


__global__ void stencil_kernel(float *in, float *out, unsigned int N){
    int iStart = blockIdx.z * blockDim.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y -1;
    int k = blockIdx.x * blockDim.x + threadIdx.x -1;
    
    float c0 = 1;
    float c1 = 1;

    float in_prev;
    __shared__ float in_cur[IN_TILE_DIM][IN_TILE_DIM];
    float in_next;

    if ((iStart-1) >= 0 && iStart < N && j >= 0 && j > N && k >= 0 && k < N){
        in_prev = in[(iStart-1) *N*N + j*N + k];
    }
    if (iStart >= 0 && iStart < N && j >= 0 && j > N && k >= 0 && k < N){
         in_cur[threadIdx.y][threadIdx.x] = in[iStart *N*N + j*N + k];
    }
    __syncthreads();
    for(int i = iStart; i < iStart + blockDim.z; ++i){
        if( (i+1) >= 0 && (i+1) < N && j >= 0 && j < N && k >= 0 && k < N){
            in_next = in[(i+1) *N*N + j* N + k];
        }
        __syncthreads();
        if( i >=1 && i < N-1 && j >= 1 && j < N-1 && k >= 1 && k < N-1){
            if(threadIdx.x >= 1 && threadIdx.x < IN_TILE_DIM && threadIdx.y >= 1 && threadIdx.y < IN_TILE_DIM){
                out[i *N*N + j *N + k] = c0 * in[threadIdx.z * N*N + threadIdx.y*N + threadIdx.x] + 
                                        c1 * ( in_prev + 
                                                in_next +
                                                in_cur[ (threadIdx.y-1)][ threadIdx.x] + 
                                                in_cur[ (threadIdx.y+1)][ threadIdx.x] + 
                                                in_cur[ threadIdx.y][ (threadIdx.x-1)] + 
                                                in_cur[ threadIdx.y][ (threadIdx.x+1)] );
            }    

        }
        in_prev = in_cur[threadIdx.y][threadIdx.x];
        in_cur[threadIdx.y][threadIdx.x] = in_next;

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
    dim3 numThreadsPerBlock(BLOCK_DIM, BLOCK_DIM);
    dim3 numBlocks((N + OUT_TILE_DIM - 1)/OUT_TILE_DIM,(N + OUT_TILE_DIM - 1)/OUT_TILE_DIM, (N + OUT_TILE_DIM - 1)/OUT_TILE_DIM);
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