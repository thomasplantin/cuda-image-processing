#ifndef UTIL_H
#define UTIL_H

#include "../image.h"

const int MAX_THREADS = 1024;       // Nader's threads per block
const int THREADS_PER_BLOCK = 256;  // Thomas' threads per block

stbi_uc* zeroPadImage(stbi_uc* input_image, int &width, int &height, int channels, int filter_size);
__device__ bool isOutOfBounds(int x, int y, int image_width, int image_height);

stbi_uc* zeroPadImage(stbi_uc* input_image, int &width, int &height, int channels, int filter_size) {
    int half_filter_size = filter_size / 2;
    int padded_width = width + 2 * half_filter_size;
    int padded_height = height + 2 * half_filter_size;

    stbi_uc* padded_image = (stbi_uc*) malloc(channels * padded_width * padded_height * sizeof(stbi_uc));

    Pixel zero_pixel = { .r = 0, .g = 0, .b = 0, .a = 0 };
    Pixel other_pixel;

    for (int i = 0; i < padded_width; i++) {
        for (int j = 0; j < padded_height; j++) {
            if (i < half_filter_size || i > padded_width - half_filter_size || j < half_filter_size || j > padded_width - half_filter_size) {
                setPixel(padded_image, padded_width, i, j, &zero_pixel);
            } else {
                getPixel(input_image, width, i - half_filter_size, j - half_filter_size, &other_pixel);
                setPixel(padded_image, padded_width, i, j, &other_pixel);
            }
        }
    }

    width = padded_width;
    height = padded_height;

    return padded_image;
}

__device__ bool isOutOfBounds(int x, int y, int image_width, int image_height) {
    if (x < 0 || y < 0 || x >= image_width || y >= image_height) {
        return true;
    } else {
        return false;
    }
}

#endif