#include <cuda_runtime.h>
#include <cstdio>
#include <iostream>
#include <vector>

#include "common.h"
#include "timer.h"

// Convolution with tiling

#define OUT_TILE_DIM 32
int IN_TILE_DIM = OUT_TILE_DIM + 2*MASK_RADIUS ;


__constant__ float mask_c[MASK_DIM][MASK_DIM];

__global__ void convolution_kernel(float *input, float *output, unsigned int width, unsigned int height){
    int inRow = blockIdx.y * blockDim.y + threadIdx.y - MASK_RADIUS;
    int inCol = blockIdx.x * blockDim.x + threadIdx.x - MASK_RADIUS;

    __shared__ float in_s[IN_TILE_DIM][IN_TILE_DIM];

    if(inRow >= 0 && inRow < width  && inCol >= 0 && inCol < height){
        in_s[threadIdx.y][threadIdx.x] = input[inRow * width + inCol];
    }
    else{
        in_s[threadIdx.y][threadIdx.x] = 0.0f;
    }
    __syncthreads();
    if(threadIdx.x < OUT_TILE_DIM && threadIdx.y < OUT_TILE_DIM){
        int outRow = blockIdx.y * blockDim.y + threadIdx.y;
        int outCol = blockIdx.x * blockDim.x + threadIdx.x;
        if(outRow < height && outCol < width){
            float sum = 0.0f;
            for(int maskRow = 0; maskRow < MASK_DIM; ++maskRow){
                for(int maskCol = 0; maskCol < MASK_DIM; ++maskCol){
                    int inRow = threadIdx.y + maskRow;
                    int inCol = threadIdx.x + maskCol;
                    if(inRow < height && inRow >= 0 &&
                    inCol < width && inCol >= 0){
                        sum += mask_c[maskRow][maskCol]*in_s[inRow][inCol];
                    }
                }
            }
            output[outRow * width + outCol] = sum;
        }
    }
}



void convolution_gpu(float mask[][MASK_DIM], float* input, float* output, unsigned int width, unsigned int height){
    Timer timer;

    //Allocate GPU memory
    startTime(&timer);
    float *input_d, *output_d;
    cudaMalloc(&input_d, width * height * sizeof(float));
    cudaMalloc(&output_d, width * height * sizeof(float));
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Allocation Time");

    //Copy data to GPU
    startTime(&timer);
    cudaMemcpy(input_d, input, width * height * sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copy to GPU Time");

    // Copy mask to Constant Memory
    startTime(&timer);
    cudaMemcpyToSymbol(mask_c, mask, MASK_DIM*MASK_DIM*sizeof(float));
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "copy to GPU Constant Memory time");

    //call kernel
    startTime(&timer);
    dim3 numThreadsPerBlock(IN_TILE_DIM, IN_TILE_DIM);
    dim3 numBlocks((width + OUT_TILE_DIM -1)/OUT_TILE_DIM, (height + OUT_TILE_DIM - 1)/OUT_TILE_DIM );
    convolution_kernel<<<numBlocks, numThreadsPerBlock>>>(input_d, output_d, width, height);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Kernel time");
    
    //copy data from gpu
    startTime(&timer);
    cudaMemcpy(output, output_d, width * height * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copy from GPU Time");

    // Free GPU Time
    startTime(&timer);
    cudaFree(input_d);
    cudaFree(output_d);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Deallocation Time");

}

int main() {
    const unsigned int width = 1024;
    const unsigned int height = 1024;

    std::vector<float> input(width * height);
    std::vector<float> output(width * height, 0.0f);

    float mask[MASK_DIM][MASK_DIM];

    // Initialize input matrix
    for (unsigned int row = 0; row < height; ++row) {
        for (unsigned int col = 0; col < width; ++col) {
            input[row * width + col] = 1.0f;
        }
    }

    // Initialize mask
    for (int row = 0; row < MASK_DIM; ++row) {
        for (int col = 0; col < MASK_DIM; ++col) {
            mask[row][col] = 1.0f;
        }
    }

    // Run GPU convolution
    convolution_gpu(
        mask,
        input.data(),
        output.data(),
        width,
        height
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