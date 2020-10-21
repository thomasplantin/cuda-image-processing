nvcc main.cu
mkdir -p output/

./a.out images/lena_rgb.png output/vertical_flip.png vflip single
./a.out images/lena_rgb.png output/horizontal_flip.png hflip single
./a.out images/lena_rgb.png output/blur.png blur single
./a.out images/lena_rgb.png output/sharpen.png sharpen single
./a.out images/lena_rgb.png output/edge.png edge single
./a.out images/lena_rgb.png output/grayscale.png gray single
./a.out images/lena_rgb.png output/weighted_grayscale.png grayweight single
