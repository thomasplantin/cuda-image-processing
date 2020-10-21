#ifndef CONVOLVE_H
#define CONVOLVE_H

#include "../image.h"
#include "util.h"

void checkCudaErrorAux(const char *file, unsigned line, const char *statement, cudaError_t err) {
    if (err == cudaSuccess) {
        return;
    }
    fprintf(stderr, "%s returned %s(%d) at %s:%u\n", statement, cudaGetErrorString(err), err, file, line);
    exit(1);
}

#define CUDA_CHECK_RETURN(value) checkCudaErrorAux(__FILE__,__LINE__, #value, value)

typedef enum { Global, Shared, Constant, Texture } Memory;

texture<uchar4, cudaTextureType2D, cudaReadModeElementType> texture_reference;

__constant__ int constant_mask[100];
__constant__ int constant_mask_dimension; // maximum 10

stbi_uc* convolve(const stbi_uc* input_image, int width, int height, int channels, const int* mask, int mask_dimension, Memory memory);
stbi_uc** convolveBatch(stbi_uc** input_image, int input_size, int width, int height, int channels, const int* mask, int mask_dimension, Memory memory, bool batch);

__global__ void convolve(const stbi_uc* input_image, stbi_uc* output_image, const int width, const int height, const int* mask, const int mask_dimension);
__global__ void convolveSharedMemory(const stbi_uc* input_image, stbi_uc* output_image, const int width, const int height, const int* mask, const int mask_dimension, const int shared_memory_width);
__global__ void convolveConstantMemory(const stbi_uc* input_image, stbi_uc* output_image, const int width, const int height);
__global__ void convolveTextureMemory(const stbi_uc* input_image, stbi_uc* output_image, const int width, const int height, const int* mask, const int mask_size);

stbi_uc* convolve(const stbi_uc* input_image, int width, int height, int channels, const int* mask, int mask_dimension, Memory memory) {
    int* d_mask;

    cudaMallocManaged(&d_mask, mask_dimension * mask_dimension * sizeof(int));
    cudaMemcpy(d_mask, mask, mask_dimension * mask_dimension * sizeof(int), cudaMemcpyHostToDevice);
    
    // For constant memory
    cudaMemcpyToSymbol(constant_mask, mask, mask_dimension * mask_dimension * sizeof(int));
    cudaMemcpyToSymbol(constant_mask_dimension, &mask_dimension, sizeof(int));

    // For texture memory
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
    uchar4 *d_vector;
    cudaMalloc(&d_vector, channels * width * height * sizeof(uchar4));
    cudaBindTexture2D(NULL, texture_reference, d_vector, channelDesc, width, height, width * sizeof(uchar4));

    int image_size = channels * width * height * sizeof(stbi_uc);
    stbi_uc* d_input_image;
    stbi_uc* d_output_image;
    stbi_uc* h_output_image = (stbi_uc*) malloc(image_size);

    for (int i = 0; i < width * height * channels; i++) {
        h_output_image[i] = input_image[i];
    }

    cudaMallocManaged(&d_input_image, image_size);
    cudaMallocManaged(&d_output_image, image_size);
    cudaMemcpy(d_input_image, input_image, image_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_output_image, input_image, image_size, cudaMemcpyHostToDevice);

    int total_threads = width * height;
    int threads = min(MAX_THREADS, total_threads);

    int square_root = (int) sqrt(threads);
    dim3 block(square_root, square_root);
    dim3 grid;
    grid.x = (width + block.x - 1) / block.x;
    grid.y = (width + block.y - 1) / block.y;

    // For shared memory
    int shared_memory_size = threads * channels * sizeof(stbi_uc); 

    float time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    switch (memory) {
        case Global:
            convolve<<<grid, block>>>(d_input_image, d_output_image, width, height, d_mask, mask_dimension);
        break;
        case Shared:
            convolveSharedMemory<<<grid, block, shared_memory_size>>>(d_input_image, d_output_image, width, height, d_mask, mask_dimension, square_root);
        break;
        case Constant:
            convolveConstantMemory<<<grid, block>>>(d_input_image, d_output_image, width, height);
        break;
        case Texture:
            convolveTextureMemory<<<grid, block>>>(d_input_image, d_output_image, width, height, d_mask, mask_dimension);
        break;
        default:
            printf("No kernel launched\n");
    }
    cudaEventRecord(stop); 
    cudaEventSynchronize(stop); 
    cudaEventElapsedTime(&time, start, stop);
    printf("Time: %f\n", time);

    cudaMemcpy(h_output_image, d_output_image, image_size, cudaMemcpyDeviceToHost);
    return h_output_image;
}

stbi_uc** convolveBatch(stbi_uc** input_images, int input_size, int width, int height, int channels, const int* mask, int mask_dimension, Memory memory, bool batch) {
    int* d_mask;

    cudaMallocManaged(&d_mask, mask_dimension * mask_dimension * sizeof(int));
    cudaMemcpy(d_mask, mask, mask_dimension * mask_dimension * sizeof(int), cudaMemcpyHostToDevice);
    
    // For constant memory
    cudaMemcpyToSymbol(constant_mask, mask, mask_dimension * mask_dimension * sizeof(int));
    cudaMemcpyToSymbol(constant_mask_dimension, &mask_dimension, sizeof(int));

    // For texture memory
    // cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
    // uchar4 *d_vector;
    // cudaMalloc(&d_vector, channels * width * height * sizeof(uchar4));
    // cudaBindTexture2D(NULL, texture_reference, d_vector, channelDesc, width, height, width * sizeof(uchar4));

    cudaStream_t* streams = (cudaStream_t*) malloc(input_size * sizeof(cudaStream_t));
    for (int i = 0; i < input_size; i++) {
        cudaStreamCreate(&streams[i]);
        if (!batch) {
            streams[i] = 0;
        }
    }


    int image_size = channels * width * height * sizeof(stbi_uc);
    stbi_uc** d_input_images;
    stbi_uc** d_output_images;
    stbi_uc** h_output_images = (stbi_uc**) malloc(input_size * sizeof(stbi_uc*));
    stbi_uc** temp_input_images = (stbi_uc**) malloc(input_size * sizeof(stbi_uc*));
    stbi_uc** temp_output_images = (stbi_uc**) malloc(input_size * sizeof(stbi_uc*));
    CUDA_CHECK_RETURN(cudaMallocManaged(&d_input_images, input_size * sizeof(stbi_uc*)));
    CUDA_CHECK_RETURN(cudaMallocManaged(&d_output_images, input_size * sizeof(stbi_uc*)));
    for (int i = 0; i < input_size; i++) {
        h_output_images[i] = (stbi_uc*) malloc(image_size);
        for (int j = 0; j < width * height * channels; j++) {
            h_output_images[i][j] = 0;
        }
        
        CUDA_CHECK_RETURN(cudaMallocManaged(&temp_input_images[i], image_size));
        CUDA_CHECK_RETURN(cudaMallocManaged(&temp_output_images[i], image_size));
        CUDA_CHECK_RETURN(cudaMemcpy(temp_input_images[i], input_images[i], image_size, cudaMemcpyHostToDevice));
        CUDA_CHECK_RETURN(cudaMemcpy(temp_output_images[i], input_images[i], image_size, cudaMemcpyHostToDevice));
    }
    CUDA_CHECK_RETURN(cudaMemcpy(d_input_images, temp_input_images, input_size * sizeof(stbi_uc*), cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(d_output_images, temp_output_images, input_size * sizeof(stbi_uc*), cudaMemcpyHostToDevice));

    int total_threads = width * height;
    int threads = min(MAX_THREADS, total_threads);

    int square_root = (int) sqrt(threads);
    dim3 block(square_root, square_root);
    dim3 grid;
    grid.x = (width + block.x - 1) / block.x;
    grid.y = (width + block.y - 1) / block.y;


    // For shared memory
    int shared_memory_size = threads * channels * sizeof(stbi_uc); 

    for (int i = 0; i < input_size; i++) {
        switch (memory) {
            case Global:
                convolve<<<grid, block, 0, streams[i]>>>(d_input_images[i], d_output_images[i], width, height, d_mask, mask_dimension);
            break;
            case Shared:
                convolveSharedMemory<<<grid, block, shared_memory_size, streams[i]>>>(d_input_images[i], d_output_images[i], width, height, d_mask, mask_dimension, square_root);
            break;
            case Constant:
                convolveConstantMemory<<<grid, block, 0, streams[i]>>>(d_input_images[i], d_output_images[i], width, height);
            break;
            case Texture:
                convolveTextureMemory<<<grid, block, 0, streams[i]>>>(d_input_images[i], d_output_images[i], width, height, d_mask, mask_dimension);
            break;
            default:
                printf("No kernel launched\n");
        }
    }
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    CUDA_CHECK_RETURN(cudaMemcpy(h_output_images, d_output_images, sizeof(stbi_uc*), cudaMemcpyDeviceToHost));

    for (int i = 0; i < input_size; i++) {
        CUDA_CHECK_RETURN(cudaMemcpyAsync(h_output_images[i], d_output_images[i], image_size, cudaMemcpyDeviceToHost, streams[i]));
        // CUDA_CHECK_RETURN(cudaStreamDestroy(streams[i]));
    }
    cudaDeviceSynchronize();
    return h_output_images;    
}

__global__ void convolve(const stbi_uc* input_image, stbi_uc* output_image, const int width, const int height, const int* mask, const int mask_dimension) {
    const int x_coordinate = blockIdx.x * blockDim.x + threadIdx.x;
    const int y_coordinate = blockIdx.y * blockDim.y + threadIdx.y;
    
    int half_mask_size = mask_dimension / 2;

    Pixel current_pixel;
    int red = 0;
    int blue = 0;
    int green = 0;
    // int alpha = 0;
    for (int i = 0; i < mask_dimension; i++) {
        for (int j = 0; j < mask_dimension; j++) {
            int current_x_global = x_coordinate - half_mask_size + i;
            int current_y_global = y_coordinate - half_mask_size + j;
            if (isOutOfBounds(current_x_global, current_y_global, width, height)) {
                continue;
            }
            getPixel(input_image, width, current_x_global, current_y_global, &current_pixel);
            int mask_element = mask[i * mask_dimension + j];

            red += current_pixel.r * mask_element;
            green += current_pixel.g * mask_element;
            blue += current_pixel.b * mask_element;
        }
    }

    Pixel pixel;
    if (red < 0) {
        pixel.r = 0;
    } else if (red > 255) {
        pixel.r = 255;
    } else {
        pixel.r = red;
    }
    if (green < 0) {
        pixel.g = 0;
    } else if (green > 255) {
        pixel.g = 255;
    } else {
        pixel.g = green;
    }
    if (blue < 0) {
        pixel.b = 0;
    } else if (blue > 255) {
        pixel.b = 255;
    } else {
        pixel.b = blue;
    }

    setPixel(output_image, width, x_coordinate, y_coordinate, &pixel);
}

__global__ void convolveSharedMemory(const stbi_uc* input_image, stbi_uc* output_image, const int width, const int height, const int* mask, const int mask_dimension, const int shared_memory_width) {
    const int x_coordinate = blockIdx.x * blockDim.x + threadIdx.x;
    const int y_coordinate = blockIdx.y * blockDim.y + threadIdx.y;

    extern __shared__ stbi_uc data[];
    int half_mask_size = mask_dimension / 2;
    int shmem_x = threadIdx.x;
    int shmem_y = threadIdx.y;

    Pixel current_pixel;
    getPixel(input_image, width, x_coordinate, y_coordinate, &current_pixel);
    setPixel(data, shared_memory_width, shmem_x, shmem_y, &current_pixel);
    __syncthreads();

    int red = 0;
    int blue = 0;
    int green = 0;
    // int alpha = 0;
    for (int i = 0; i < mask_dimension; i++) {
        for (int j = 0; j < mask_dimension; j++) {
            int current_x_global = x_coordinate - half_mask_size + i;
            int current_y_global = y_coordinate - half_mask_size + j;
            int current_x = shmem_x - half_mask_size + i;
            int current_y = shmem_y - half_mask_size + j;

            if (isOutOfBounds(current_x_global, current_y_global, width, height)) {
                continue;
            }
            if (isOutOfBounds(current_x, current_y, shared_memory_width, shared_memory_width)) {
                getPixel(input_image, width, current_x_global, current_y_global, &current_pixel);
            } else {
                getPixel(data, shared_memory_width, current_x, current_y, &current_pixel);
            }
            
            int mask_element = mask[i * mask_dimension + j];

            red += current_pixel.r * mask_element;
            green += current_pixel.g * mask_element;
            blue += current_pixel.b * mask_element;
        }
    }

    Pixel pixel;
    if (red < 0) {
        pixel.r = 0;
    } else if (red > 255) {
        pixel.r = 255;
    } else {
        pixel.r = red;
    }
    if (green < 0) {
        pixel.g = 0;
    } else if (green > 255) {
        pixel.g = 255;
    } else {
        pixel.g = green;
    }
    if (blue < 0) {
        pixel.b = 0;
    } else if (blue > 255) {
        pixel.b = 255;
    } else {
        pixel.b = blue;
    }

    setPixel(output_image, width, x_coordinate, y_coordinate, &pixel);
}

__global__ void convolveConstantMemory(const stbi_uc* input_image, stbi_uc* output_image, const int width, const int height) {
    const int x_coordinate = blockIdx.x * blockDim.x + threadIdx.x;
    const int y_coordinate = blockIdx.y * blockDim.y + threadIdx.y;

    int half_mask_size = constant_mask_dimension / 2;

    Pixel current_pixel;
    int red = 0;
    int blue = 0;
    int green = 0;
    // int alpha = 0;
    for (int i = 0; i < constant_mask_dimension; i++) {
        for (int j = 0; j < constant_mask_dimension; j++) {
            int current_x_global = x_coordinate - half_mask_size + i;
            int current_y_global = y_coordinate - half_mask_size + j;
            if (isOutOfBounds(current_x_global, current_y_global, width, height)) {
                continue;
            }
            getPixel(input_image, width, current_x_global, current_y_global, &current_pixel);
            int mask_element = constant_mask[i * constant_mask_dimension + j];

            red += current_pixel.r * mask_element;
            green += current_pixel.g * mask_element;
            blue += current_pixel.b * mask_element;
        }
    }

    Pixel pixel;
    if (red < 0) {
        pixel.r = 0;
    } else if (red > 255) {
        pixel.r = 255;
    } else {
        pixel.r = red;
    }
    if (green < 0) {
        pixel.g = 0;
    } else if (green > 255) {
        pixel.g = 255;
    } else {
        pixel.g = green;
    }
    if (blue < 0) {
        pixel.b = 0;
    } else if (blue > 255) {
        pixel.b = 255;
    } else {
        pixel.b = blue;
    }

    setPixel(output_image, width, x_coordinate, y_coordinate, &pixel);
}

__global__ void convolveTextureMemory(const stbi_uc* input_image, stbi_uc* output_image, const int width, const int height, const int* mask, const int mask_dimension) {
    const int x_coordinate = blockIdx.x * blockDim.x + threadIdx.x;
    const int y_coordinate = blockIdx.y * blockDim.y + threadIdx.y;
    
    int half_mask_size = mask_dimension / 2;

    int red = 0;
    int blue = 0;
    int green = 0;
    // int alpha = 0;
    for (int i = 0; i < mask_dimension; i++) {
        for (int j = 0; j < mask_dimension; j++) {
            int current_x_global = x_coordinate - half_mask_size + i;
            int current_y_global = y_coordinate - half_mask_size + j;

            if (isOutOfBounds(current_x_global, current_y_global, width, height)) {
                continue;
            }
            uchar4 current_pixel = tex2D(texture_reference, x_coordinate, y_coordinate);
            int mask_element = mask[i * mask_dimension + j];

            red += current_pixel.x * mask_element;
            green += current_pixel.y * mask_element;
            blue += current_pixel.z * mask_element;
        }
    }

    Pixel pixel;
    if (red < 0) {
        pixel.r = 0;
    } else if (red > 255) {
        pixel.r = 255;
    } else {
        pixel.r = red;
    }
    if (green < 0) {
        pixel.g = 0;
    } else if (green > 255) {
        pixel.g = 255;
    } else {
        pixel.g = green;
    }
    if (blue < 0) {
        pixel.b = 0;
    } else if (blue > 255) {
        pixel.b = 255;
    } else {
        pixel.b = blue;
    }

    setPixel(output_image, width, x_coordinate, y_coordinate, &pixel);
}

#endif