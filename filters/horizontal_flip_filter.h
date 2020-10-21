#ifndef HORIZONTAL_FLIP_FILTER_H
#define HORIZONTAL_FLIP_FILTER_H

#include "../image.h"
#include "util.h"

stbi_uc* horizontalFlip(stbi_uc* input_image, int width, int height, int channels);
__global__ void horizontalFlip(stbi_uc* input_image, stbi_uc* output_image, int width, int height, int channels, int total_threads);

stbi_uc* horizontalFlip(stbi_uc* input_image, int width, int height, int channels) {
    int image_size = channels * width * height * sizeof(stbi_uc);
    stbi_uc* d_input_image;
    stbi_uc* d_output_image;
    stbi_uc* h_output_image = (stbi_uc*) malloc(image_size);

    cudaMallocManaged(&d_input_image, image_size);
    cudaMallocManaged(&d_output_image, image_size);
    cudaMemcpy(d_input_image, input_image, image_size, cudaMemcpyHostToDevice);

    int total_threads = width * height;
    int threads = min(MAX_THREADS, total_threads);
    int blocks = (threads == total_threads) ? 1 : total_threads / MAX_THREADS;

    printf("Blocks %d, threads %d\n", blocks, threads);
    horizontalFlip<<<blocks, threads>>>(d_input_image, d_output_image, width, height, channels, total_threads);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output_image, d_output_image, image_size, cudaMemcpyDeviceToHost);
    return h_output_image;
}

__global__ void horizontalFlip(stbi_uc* input_image, stbi_uc* output_image, int width, int height, int channels, int total_threads) {
    const int thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    if (thread_id >= total_threads) {
        return;
    }

    int y_coordinate = thread_id / width;
    int old_x_coordinate = thread_id % height;
    int new_x_coordinate = (width - 1) - old_x_coordinate;
    Pixel pixel;

    getPixel(input_image, width, old_x_coordinate, y_coordinate, &pixel);
    setPixel(output_image, width, new_x_coordinate, y_coordinate, &pixel);
}

#endif