
// GPU Rank Sort
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <cstdlib>
#include <time.h>
using namespace std;

__host__ void sortOnHost(int n, int* h_a, int* h_b) {

    for (int i = 0; i < n; i++) {
        int rank = 0;
        for (int j = 0; j < n; j++) {
            if (h_a[i] > h_a[j])
                rank++;
        }
        h_b[rank] = h_a[i];
    }
}

__global__ void sortOnDevice1(int n, int* d_a, int* d_b) {
    // Rank sort but we access global memory e.g d_a[threadId], d_a[j]
    // iteration i computed by threadId
    int threadId = threadIdx.x + blockIdx.x * blockDim.x;

    if (threadId < n) {
        int rank = 0;
        for (int j = 0; j < n; j++) {
            // d_a[threadId] accessed, slow.
            if (d_a[threadId] > d_a[j])
                rank++;
        }
        // d_a[threadId] accessed again, slow!!!
        d_b[rank] = d_a[threadId];

    }
}

__global__ void sortOnDevice2(int n, int* d_a, int* d_b) {
    // Again but only access global memory once
    int threadId = threadIdx.x + blockIdx.x * blockDim.x;

    if (threadId < n) {
        int rank = 0, elem = d_a[threadId]; // Accessed once 
        for (int j = 0; j < n; j++) {
            if (elem > d_a[j]) // Yes, we are accessing it here but we need to for comparison
                rank++;
        }
        d_b[rank] = elem;

    }
}

__global__ void sortOnDevice3(int n, int* d_a, int* d_b) {
    // Fastest approach using shared memory in the GPU
    int threadId = threadIdx.x + blockIdx.x * blockDim.x;

    // Shared memory, scoped to the block of threads
    extern __shared__ int share_a[];

    // copy d_a to share_a
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        share_a[i] = d_a[i];
    }

    // Wait for shared memory to be filled up
    __syncthreads();

    if (threadId < n) {
        int rank = 0, elem = share_a[threadId]; // Pull from fast shared memory :)
        for (int j = 0; j < n; j++) {
            if (elem > share_a[j]) // Comparison using fast shared memory
                rank++;
        }
        d_b[rank] = elem;

    }
}


int main(int argc, char** argv) {
    // n number of elements in the array
    int n = 100000, blockSize = 512;

    // allocate the arrays and initialise h_a
    int* h_a = (int*)malloc(n * sizeof(int));
    int* h_b = (int*)malloc(n * sizeof(int));

    int* d_a, * d_b;
    cudaMalloc((void**)&d_a, n * sizeof(int));
    cudaMalloc((void**)&d_b, n * sizeof(int));

    // Fill in reverse to ensure nothing is sorted.
    for (int i = 0; i < n; i++)
        h_a[i] = n - i;

    clock_t time1, time2;

    // copy h_a to d_a
    cudaMemcpy(d_a, h_a, n * sizeof(int), cudaMemcpyHostToDevice);

    /*
        HOST SORT
    */

    time1 = clock();
    sortOnHost(n, h_a, h_b);
    time2 = clock();
    printf("Host Sort: %lf\n", 1.0 * (time2 - time1) / CLOCKS_PER_SEC);

    /*
        GPU SORT
    */
    int gridSize = (n + blockSize - 1) / blockSize;
    time1 = clock();
    sortOnDevice1 << <gridSize, blockSize >> > (n, d_a, d_b);
    cudaDeviceSynchronize(); // Wait for GPU to finish
    time2 = clock();
    printf("GPU Sort using Global Memory: %lf\n", 1.0 * (time2 - time1) / CLOCKS_PER_SEC);

    time1 = clock();
    sortOnDevice2 << <gridSize, blockSize >> > (n, d_a, d_b);
    cudaDeviceSynchronize();
    time2 = clock();
    printf("GPU Sort storing global memory value in local variable: %lf\n", 1.0 * (time2 - time1) / CLOCKS_PER_SEC);
    
    time1 = clock();
    sortOnDevice3 << <gridSize, blockSize >> > (n, d_a, d_b);
    cudaDeviceSynchronize(); 
    time2 = clock();
    printf("GPU Sort Shared Memory: %lf\n", 1.0 * (time2 - time1) / CLOCKS_PER_SEC);

    // copy d_b to h_b
    cudaMemcpy(h_b, d_b, n * sizeof(int), cudaMemcpyDeviceToHost);

    // Print out first 100 elements to make sure they are sorted
    //for(int i = 0; i < 100; i++)
        //printf("%d ", h_b[i]);

    // No memory leaks this time around, i want my games to still have some VRAM...
    cudaFree(d_a);
    cudaFree(d_b);

    scanf("%d");
    return 0;
}