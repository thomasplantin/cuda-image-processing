# image-processing-cuda

This repository contains the codebase to run various parallel GPU based algorithms for image processing. Some of the algorithms implemented are image blurring, image flipping, and more. These parallel algorithms are run on a GPU using CUDA.

### Test Cases

To run our test cases, run
```chmod u+x ex_single.sh```
```chmod u+x ex_batch.sh```

To run our single image mode test cases, run
```ex_single.sh```

This will run all of our filters on the image "images/lena_rgb.png".
Seven images should be generated in a new directory called output.
Each image is the output of the filter it is named after. To check
the correctness of the output, compare each image with the corresponding
image in the expected_output directory. You can also compare each
output image with the original "images/lena_rgb.png".

To run our batch images mode test cases, run
```ex_batch.sh```

First, this make 100 copies of "images/lena_rgb.png" to images/batch/
This will help us generate a large number of images. Second, the script
will filter them with the edge detection filter in batch mode. It will
read all images, filter them, and write them back. Writing these images
will take some time. Compare any image in the output/batch/ directory to
"expected_output/edge.png" to check if the output is correct.

If you prefer not to run our scripts, you can run each command in the
scripts one by one instead.

### Repository structure

main.cu simply parses arguments and calls the necessary filters
stb_image/ contains the image library we used.
image.h is our wrapper for the image library.
filters/ contains each filter implemented in a header file
filters/convolve.h is called from every convolution filter.
Other filters have their own kernels.
expected_output/ stores the expected_output of each filter
images/lena_rgb.png is the input image for all our test cases.

### Introduction

Note: You must have the ability to run CUDA files on your end in order to render any of the work in this repository. For more information about CUDA, please visit this link: https://developer.nvidia.com/about-cuda

Before any filters can be applied, the `main.cu` file must be compiled. To do that, open your terminal and run the following command from the root directory of this project:

```nvcc main.cu```

You can ignore any warnings that are printed to the console. A file named `a.out` should now be stored in the root directory.

To apply a filter to an image, please follow the next steps:
* Import an image of your choice in the `images` directory, or just use one of the images already there.
* From the root directory, run `a.out` with the following arguments (see filter arguments in the table below):
```./a.out path_to_image_input path_to_image_output filter_arg```
* Conversely, you can run the following command:
```sbatch runs/filter_run``` (please check the `runs` directory to see which file you should call).

### Table 1: Filters and their arguments
|      Filter     |  Filter Arg |
|:---------------:|:-----------:|
| Horizontal Flip | hflip       |
| Vertical Flip   | vflip       |
| Sharpening      | sharpen     |
| Blurring        | blur        |

