#ifndef GRAYSCALE_WEIGHTED_FILTER_H
#define GRAYSCALE_WEIGHTED_FILTER_H

#include "../image.h"
#include "util.h"

stbi_uc* grayWeight(stbi_uc* input_image, int width, int height, int channels);
__global__ void grayWeightKernel(stbi_uc* input_image, stbi_uc* output_image, int width, int height, int channels, int total_threads);

stbi_uc* grayWeight(stbi_uc* input_image, int width, int height, int channels) {
    int image_size = channels * width * height * sizeof(stbi_uc);
    stbi_uc* d_input_image;
    stbi_uc* d_output_image;
    stbi_uc* h_output_image = (stbi_uc*) malloc(image_size);

    cudaMallocManaged(&d_input_image, image_size);
    cudaMallocManaged(&d_output_image, image_size);
    cudaMemcpy(d_input_image, input_image, image_size, cudaMemcpyHostToDevice);

    int total_threads = width * height;
    int threads = min(THREADS_PER_BLOCK, total_threads);
    int blocks = (threads == total_threads) ? 1 : total_threads / THREADS_PER_BLOCK;

    printf("Blocks %d, threads %d\n", blocks, threads);
    grayWeightKernel<<<blocks, threads>>>(d_input_image, d_output_image, width, height, channels, total_threads);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output_image, d_output_image, image_size, cudaMemcpyDeviceToHost);
    return h_output_image;
}

__global__ void grayWeightKernel(stbi_uc* input_image, stbi_uc* output_image, int width, int height, int channels, int total_threads) {
    
    const int thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    
    if (thread_id >= total_threads) {
        return;
    }

    // Declare coordinates based on thread_id and image dimensions.
    int x_coordinate = thread_id % height;
    int y_coordinate = thread_id / width;

    Pixel inPixel, outPixel;

    getPixel(input_image, width, x_coordinate, y_coordinate, &inPixel);

    double alpha = inPixel.r*0.21f + inPixel.g*0.72f + inPixel.b*0.07f;

    outPixel.r = int(alpha);
    outPixel.g = int(alpha);
    outPixel.b = int(alpha);
    outPixel.a = inPixel.a;

    setPixel(output_image, width, x_coordinate, y_coordinate, &outPixel);

}

#endif