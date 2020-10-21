#ifndef LOAD_IMAGE_H
#define LOAD_IMAGE_H

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image/stb_image_write.h"

typedef struct Pixel {
    stbi_uc r;
    stbi_uc g;
    stbi_uc b;
    stbi_uc a;
} Pixel;

stbi_uc* loadImage(const char* path_to_image, int* width, int* height, int* channels) {
    return stbi_load(path_to_image, width, height, channels, STBI_rgb_alpha);
}

void writeImage(const char* path_to_image, stbi_uc* image, int width, int height, int channels) {
    stbi_write_png(path_to_image, width, height, channels, image, width * channels);
}

void imageFree(stbi_uc* image) {
    stbi_image_free(image);
}

__host__ __device__ void getPixel(stbi_uc* image, int width, int x, int y, Pixel* pixel) {
    const stbi_uc* p = image + (STBI_rgb_alpha * (y * width + x));
    pixel->r = p[0];
    pixel->g = p[1];
    pixel->b = p[2];
    pixel->a = p[3];
}

__host__ __device__ void getPixel(const stbi_uc* image, int width, int x, int y, Pixel* pixel) {
    const stbi_uc* p = image + (STBI_rgb_alpha * (y * width + x));
    pixel->r = p[0];
    pixel->g = p[1];
    pixel->b = p[2];
    pixel->a = p[3];
}

__host__ __device__ void setPixel(stbi_uc* image, int width, int x, int y, Pixel* pixel) {
    stbi_uc* p = image + (STBI_rgb_alpha * (y * width + x));
    p[0] = pixel->r;
    p[1] = pixel->g;
    p[2] = pixel->b;
    p[3] = pixel->a;
}

__host__ __device__ void printPixel(Pixel* pixel) {
    printf("r = %u, g = %u, b = %u, a = %u\n", pixel->r, pixel->g, pixel->b, pixel->a);
}

#endif