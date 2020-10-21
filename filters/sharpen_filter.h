#ifndef SHARPEN_FILTER_H
#define SHARPEN_FILTER_H

#include "../image.h"
#include "convolve.h"

int sharpen_mask_3[] = { 0, -1,  0,
                        -1,  5, -1,
                         0, -1,  0};

int sharpen_mask_dimension_3 = 3;

stbi_uc* sharpen(stbi_uc* input_image, int width, int height, int channels);

stbi_uc* sharpen(stbi_uc* input_image, int width, int height, int channels) {
    Memory memory = Global;
    stbi_uc* output = convolve(input_image, width, height, channels, sharpen_mask_3, sharpen_mask_dimension_3, memory);

    return output;
}

#endif