#include <string>
#include <random>

#include "merge_sort.h"

using namespace std;

#define custom_min(a, b) (a < b ? a : b)
// Based on https://github.com/kevin-albert/cuda-mergesort/blob/master/mergesort.cu

__host__ std::tuple<dim3, dim3, int, bool> parseCommandLineArguments(int argc, char** argv) 
{
    int numElements = 32;
    bool maxLongs = false;
    dim3 threadsPerBlock;
    dim3 blocksPerGrid;

    threadsPerBlock.x = 32;
    threadsPerBlock.y = 1;
    threadsPerBlock.z = 1;

    blocksPerGrid.x = 8;
    blocksPerGrid.y = 1;
    blocksPerGrid.z = 1;

    for (int i = 1; i < argc; i++) {
        if (argv[i][0] == '-' && argv[i][1] && !argv[i][2]) {
            char arg = argv[i][1];
            unsigned int* toSet = 0;
            switch(arg) {
                case 'x':
                    toSet = &threadsPerBlock.x;
                    break;
                case 'y':
                    toSet = &threadsPerBlock.y;
                    break;
                case 'z':
                    toSet = &threadsPerBlock.z;
                    break;
                case 'X':
                    toSet = &blocksPerGrid.x;
                    break;
                case 'Y':
                    toSet = &blocksPerGrid.y;
                    break;
                case 'Z':
                    toSet = &blocksPerGrid.z;
                    break;
                case 'n':
                    i++;
                    numElements = stoi(argv[i]);
                    break;
                case 'l':
                    maxLongs = true;
                    break;
            }
            if (toSet) {
                i++;
                *toSet = (unsigned int) strtol(argv[i], 0, 10);
            }
        }
    }
    return {threadsPerBlock, blocksPerGrid, numElements, maxLongs};
}

__host__ long *generateRandomLongArray(int numElements, bool maxLongs)
{
    // allocate memory needed for the unsorted data
    long *randomLongs = static_cast<long*>(malloc(sizeof(long) * numElements));

    // create a random engine and distribution
    std::default_random_engine engine(std::random_device{}());
    auto d = std::uniform_int_distribution<long>(-1000, 1000); // default for readablity
    if (maxLongs) {
        d.param(std::uniform_int_distribution<long>::param_type{
          std::numeric_limits<long>::min(), std::numeric_limits<long>::max()
        });
    }

    // fill the allocated memory with random long values
    for (size_t i = 0; i < numElements; ++i) {
        randomLongs[i] = d(engine);
    }

    return randomLongs;
}

__host__ void printHostMemory(long *host_mem, int num_elments)
{
    // Output results
    for(int i = 0; i < num_elments; i++)
    {
        printf("%ld ",host_mem[i]);
    }
    printf("\n");
}

__host__ int main(int argc, char** argv) 
{

    auto[threadsPerBlock, blocksPerGrid, numElements, maxLongs] = parseCommandLineArguments(argc, argv);

    long *data = generateRandomLongArray(numElements, maxLongs);

    printf("Unsorted data: ");
    printHostMemory(data, numElements);

    mergesort(data, numElements, threadsPerBlock, blocksPerGrid);

    // printf("\n");
    printf("Sorted data: ");
    printHostMemory(data, numElements);

    // cleanup
    free(data);
}

__host__ std::tuple <long* ,long*> allocateMemory(long* data, int numElements)
{
    // device data
    long* D_data;
    long* D_swp;

    // stack constants
    size_t size = sizeof(long)*numElements;
    
    // allocate the two arrays
    cudaMalloc((void**)&D_data, size);
    cudaMalloc((void**)&D_swp, size);

    // copy from our input list into the first array
    cudaMemcpy(D_data, data, size, cudaMemcpyHostToDevice); // (dev, host, size ..)

    return {D_data, D_swp}; // D is for device data ptr
}

__host__ void mergesort(long* data, long size, dim3 threadsPerBlock, dim3 blocksPerGrid) {

    auto [D_data, D_swp] = allocateMemory(data, size);

    long* A = D_data; // input
    long* B = D_swp;  // output

    long nThreads = threadsPerBlock.x * threadsPerBlock.y * threadsPerBlock.z *
                    blocksPerGrid.x * blocksPerGrid.y * blocksPerGrid.z;

    // slice up the list and give pieces of it to each thread, letting the pieces grow
    // bigger and bigger until the whole list is sorted
    for (int width = 2; width < (size << 1); width <<= 1) { // bitshift << 1 is the same as *= 2
        long slices = size / ((nThreads) * width) + 1;
        gpu_mergesort<<<blocksPerGrid, threadsPerBlock>>>(A, B, size, width, slices);

        // switch the input & output arrays
        long* temp = A;
        A = B;
        B = temp;
    }
    cudaDeviceSynchronize(); // synchronize before copying
    cudaMemcpy(data, A, sizeof(long)*size, cudaMemcpyDeviceToHost);

    // free the GPU memory
    cudaFree(D_data);
    cudaFree(D_swp);
}

// calculate the id of the current thread
__device__ unsigned int getIdx() {
    int x;
    return threadIdx.x +
           threadIdx.y * (x  = threadIdx.x) +
           threadIdx.z * (x *= threadIdx.y) +
           blockIdx.x  * (x *= threadIdx.z) +
           blockIdx.y  * (x *= blockIdx.z) +
           blockIdx.z  * (x *= blockIdx.y);
}

// Kernel call to mergesort
__global__ void gpu_mergesort(long* source, long* dest, long size, long width, long slices) {
    unsigned int idx = getIdx();
    // start is the width of the merge sort data span * the thread index * number of slices that this kernel will sort
    long start = width * idx * slices;
    long middle;
    long end;

    for (long slice = 0; slice < slices; slice++) {
        // break if we are out of array size
        if (start >= size) break;

        // calculate start & end of the array to be sorted
        middle = custom_min(start + static_cast<long>(0.5 * width), size);
        end = custom_min(start + width, size);

        // kernel call to merge sort
        gpu_bottomUpMerge(source, dest, start, middle, end);
        start += width;
    }
}

__device__ void gpu_bottomUpMerge(long* source, long* dest, long start, long middle, long end) {
    long i = start;
    long j = middle;
    long k = start;
    // merge sort algorithm
    while (i < middle && j < end) {
        if (source[i] < source[j]) {
            dest[k++] = source[i++];
        } else {
            dest[k++] = source[j++];
        }
    }
    // continue to populate
    while (i < middle) {
        dest[k++] = source[i++];
    }
    while (j < end) {
        dest[k++] = source[j++];
    }
}