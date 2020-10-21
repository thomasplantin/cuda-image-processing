#ifndef GRAYSCALE_FILTER_H
#define GRAYSCALE_FILTER_H

#include "../image.h"
#include "util.h"

stbi_uc* gray(stbi_uc* input_image, int width, int height, int channels);
__global__ void grayKernel(stbi_uc* input_image, stbi_uc* output_image, int width, int height, int channels, int total_threads);

stbi_uc* gray(stbi_uc* input_image, int width, int height, int channels) {
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
    grayKernel<<<blocks, threads>>>(d_input_image, d_output_image, width, height, channels, total_threads);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output_image, d_output_image, image_size, cudaMemcpyDeviceToHost);
    return h_output_image;
}

__global__ void grayKernel(stbi_uc* input_image, stbi_uc* output_image, int width, int height, int channels, int total_threads) {
    
    const int thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    
    if (thread_id >= total_threads) {
        return;
    }

    // Declare coordinates based on thread_id and image dimensions.
    int x_coordinate = thread_id % height;
    int y_coordinate = thread_id / width;

    Pixel inPixel, outPixel;

    getPixel(input_image, width, x_coordinate, y_coordinate, &inPixel);

    int alpha = (inPixel.r + inPixel.g + inPixel.b) / 3;

    outPixel.r = alpha;
    outPixel.g = alpha;
    outPixel.b = alpha;
    outPixel.a = inPixel.a;

    setPixel(output_image, width, x_coordinate, y_coordinate, &outPixel);

}

#endif