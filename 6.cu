#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "utility_functions.cuh"
#include "sorting_functions.cuh"


/*
    * @brief Sort an array, using the GPU
    * @param M the array to sort
    * @param d is the size of the array to sort (power of 2)
    * @param number_of_sub_arrays (power of 2) M is divided into sub arrays that are sorted then merged, 
    * this is the number of sub-arrays to create and it should be chosen with the number of blocks/threads in mind
    * @param n_blocks The number of GPU blocks, this determines the subdivision of M and the number of threads per block
    * @param n_threads The number of threads per block
*/
void Sort_k(int* M, const size_t d, const unsigned int n_blocks) {
//  ============================== CPU sorting variables ================================
    // number of threads per block
    unsigned int n_threads = d/n_blocks;

    // this is the size of the sub-arrays in M_GPU
    size_t size_sub_arrays = d/n_blocks;

    //  M_GPU is a pointer of pointer : this is the number of pointer/sub-arrays
    size_t size_M_GPU = n_blocks;

    //  This is the number of pointer/sub-arrays for next iteration
    size_t size_M_GPU_merged = size_M_GPU/2;

//  Used in the loops;
    size_t j;
//  =========================== Splitting M into sub-arrays =============================
    int** M_splitted = (int**) malloc(n_blocks*sizeof(int*));
    for (size_t i=0;i<n_blocks;i++) {
        M_splitted[i] = M+i*(size_sub_arrays);
    }
//  ============================= Sorting the sub arrays ================================
//  Here, each block sorts an array
    SortSmall(M_splitted, n_blocks, size_sub_arrays);

    free(M_splitted);

//  ============================= Preparing the GPU arrays ==============================
    //  Single dimension arrays, that store contiguously all the values in M
    int* Actual_M_GPU;
    int* Actual_M_GPU_merged;

    cudaMalloc(&Actual_M_GPU, d * sizeof(int));
    cudaMemcpy(Actual_M_GPU, M, d * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&Actual_M_GPU_merged, d * sizeof(int));


    //  2-dimensional arrays that are on the GPU for the merging
    int* * M_GPU;
    int* * M_GPU_merged;
    cudaMalloc(&M_GPU, size_M_GPU * sizeof(int*));
    cudaMalloc(&M_GPU_merged, size_M_GPU_merged * sizeof(int*));

    // this is necessary to fill M_GPU and M_GPU_merged
    int** tempo_array;
    tempo_array = (int**) malloc(size_M_GPU * sizeof(int*));


    j = 0;
    for (size_t i=0; i<d; i+=size_sub_arrays) {
        tempo_array[j] = Actual_M_GPU + i;
        j++;
    }
    cudaMemcpy(M_GPU, tempo_array, size_M_GPU * sizeof(int*), cudaMemcpyHostToDevice);

    size_sub_arrays = size_sub_arrays*2;
    j=0;
    for (size_t i=0; i<d; i+=size_sub_arrays) {
        tempo_array[j] = Actual_M_GPU_merged + i;
        j++;
    }
    cudaMemcpy(M_GPU_merged, tempo_array, size_M_GPU_merged * sizeof(int*), cudaMemcpyHostToDevice);

    void* swap_pointer;
//  ============================= Merging the small arrays ==============================
    while ( size_sub_arrays <= n_threads ) {
        mergeSmallBatch_k<<<n_blocks, n_threads>>>((const int**)M_GPU, (const int**) &(M_GPU[size_M_GPU_merged]), 
        M_GPU_merged, size_M_GPU, size_sub_arrays/2);

        swap_pointer = (void*) Actual_M_GPU;
        Actual_M_GPU = Actual_M_GPU_merged;
        Actual_M_GPU_merged = (int*) swap_pointer;

        swap_pointer = (void*) M_GPU;
        M_GPU = M_GPU_merged;
        M_GPU_merged = (int**) swap_pointer;


        size_M_GPU = size_M_GPU_merged;
        size_M_GPU_merged /= 2;
        size_sub_arrays *= 2;
        j=0;
        for (size_t i=0; i<d; i+=size_sub_arrays) {
            tempo_array[j] = Actual_M_GPU_merged + i;
            j++;
        }
        cudaMemcpy(M_GPU_merged, tempo_array, size_M_GPU_merged * sizeof(int*), cudaMemcpyHostToDevice);
    }

// ============================== Merging the large arrays ==============================
    while ( size_M_GPU > 1 ) {
        mergeBigBatch_k<<<n_blocks,n_threads>>>( (const int**) M_GPU, (const int**) &(M_GPU[size_M_GPU_merged]), M_GPU_merged, 
        size_M_GPU_merged, size_sub_arrays/2, size_sub_arrays);

        swap_pointer = (void*) Actual_M_GPU;
        Actual_M_GPU = Actual_M_GPU_merged;
        Actual_M_GPU_merged = (int*) swap_pointer;

        swap_pointer = (void*) M_GPU;
        M_GPU = M_GPU_merged;
        M_GPU_merged = (int**) swap_pointer;


        size_M_GPU = size_M_GPU_merged;
        size_M_GPU_merged /= 2;
        size_sub_arrays *= 2;
        j=0;
        for (size_t i=0; i<d; i+=size_sub_arrays) {
            tempo_array[j] = Actual_M_GPU_merged + i;
            j++;
        }
        cudaMemcpy(M_GPU_merged, tempo_array, size_M_GPU_merged * sizeof(int*), cudaMemcpyHostToDevice);
    }

//  ================================ Cleaning and end ===================================
    cudaMemcpy(M, Actual_M_GPU, d * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(Actual_M_GPU);
    cudaFree(Actual_M_GPU_merged);
    cudaFree(M_GPU);
    cudaFree(M_GPU_merged);
    free(tempo_array);
}


int main(void) {
    size_t n_tries = 1000;
    size_t d;
    size_t nblocks = 512;

//  Timing variables
    float elapsed_time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float total_time;

    int* M;

    for (d=512; d<=4096*4096; d*=2) {
        GenerateUnsortedRandomArray(&M, d);
        total_time = 0.0;
        for (size_t try_number=0;try_number<n_tries;try_number++) {
            ReRandomizeArray(M, d);

            cudaEventRecord(start, 0);

            Sort_k( M, d, nblocks);

            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&elapsed_time, start, stop);

            total_time += elapsed_time;

            if (IsSorted(M, d)) {
                continue;
            } else {
                printf("M isn't sorted...\n");
                exit(EXIT_FAILURE);
            }
        }
        free(M);
        printf("d : %zu, time for %zu runs : %2f s, average time : %2f s\n", d,  n_tries, total_time / 1000.0, (total_time / 1000.0) / n_tries);
    }


    return 0;
}