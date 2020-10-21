nvcc main.cu
mkdir -p images/batch
for i in {1..100}; do cp images/lena_rgb.png "images/batch/$i.png"; done

mkdir -p output/batch
./a.out images/batch/ output/batch/ edge batch 