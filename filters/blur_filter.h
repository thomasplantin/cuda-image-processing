#ifndef BLUR_FILTER_H
#define BLUR_FILTER_H

#include "../image.h"
#include "util.h"

stbi_uc* blur(stbi_uc* input_image, int width, int height, int channels);
__global__ void blurKernel(stbi_uc* input_image, stbi_uc* output_image, int width, int height, int channels, int total_threads, int padded_width, int padded_height, int* mask, int mask_size);

stbi_uc* blur(stbi_uc* input_image, int width, int height, int channels) {
    int mask_size = 5;
    int mask[mask_size*mask_size];
    for(int i = 0; i < mask_size*mask_size; i++) {
        mask[i] = 1;
    }
    int* d_mask;

    cudaMallocManaged(&d_mask, mask_size * mask_size * sizeof(int));
    cudaMemcpy(d_mask, mask, mask_size * mask_size * sizeof(int), cudaMemcpyHostToDevice);

    int padded_width = width;
    int padded_height = height;
    stbi_uc* padded_image = zeroPadImage(input_image, padded_width, padded_height, channels, mask_size);
    
    int image_size = channels * padded_width * padded_height * sizeof(stbi_uc);
    stbi_uc* d_input_image;
    stbi_uc* d_output_image;
    stbi_uc* h_output_image = (stbi_uc*) malloc(image_size);
    for (int i = 0; i < padded_width * padded_height; i++) {
        h_output_image[i] = input_image[i];
    }

    cudaMallocManaged(&d_input_image, image_size);
    cudaMallocManaged(&d_output_image, image_size);
    cudaMemcpy(d_input_image, padded_image, image_size, cudaMemcpyHostToDevice);
    imageFree(padded_image);

    int total_threads = width * height;
    int threads = min(THREADS_PER_BLOCK, total_threads);
    int blocks = (threads == total_threads) ? 1 : total_threads / THREADS_PER_BLOCK;

    printf("Blocks %d, threads %d\n", blocks, threads);
    blurKernel<<<blocks, threads>>>(d_input_image, d_output_image, width, height, channels, total_threads, padded_width, padded_height, d_mask, mask_size);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output_image, d_output_image, image_size, cudaMemcpyDeviceToHost);
    return h_output_image;
}

__global__ void blurKernel(stbi_uc* input_image, stbi_uc* output_image, int width, int height, int channels, int total_threads, int padded_width, int padded_height, int* mask, int mask_size) {
    
    const int thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    
    if (thread_id >= total_threads) {
        return;
    }

    int padding_size = mask_size / 2;

    int y_coordinate = (thread_id / width) + padding_size;
    int x_coordinate = (thread_id % height) + padding_size;

    Pixel current_pixel;
    double red = 0;
    double blue = 0;
    double green = 0;
    double alpha = 0;
    double blur_coef = 1.0f/(mask_size*mask_size);
    for (int i = 0; i < mask_size; i++) {
        for (int j = 0; j < mask_size; j++) {
            getPixel(input_image, padded_width, x_coordinate - padding_size + i, y_coordinate - padding_size + j, &current_pixel);
            int mask_element = mask[i * mask_size + j];

            red += current_pixel.r * mask_element * blur_coef;
            green += current_pixel.g * mask_element * blur_coef;
            blue += current_pixel.b * mask_element * blur_coef;
            alpha += current_pixel.a * mask_element * blur_coef;
        }
    }

    Pixel pixel;
    if (red < 0) {
        pixel.r = 0;
    } else if (red > 255) {
        pixel.r = 255;
    } else {
        pixel.r = int(red);
    }
    if (green < 0) {
        pixel.g = 0;
    } else if (green > 255) {
        pixel.g = 255;
    } else {
        pixel.g = int(green);
    }
    if (blue < 0) {
        pixel.b = 0;
    } else if (blue > 255) {
        pixel.b = 255;
    } else {
        pixel.b = int(blue);
    }
    if (alpha < 0) {
        pixel.a = 0;
    } else if (alpha > 255) {
        pixel.a = 255;
    } else {
        pixel.a = int(alpha);
    }

    setPixel(output_image, width, x_coordinate, y_coordinate, &pixel);
    
}

#endif