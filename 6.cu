#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "utility_functions.h"


/*
    * @brief Sort an array, using the GPU
    * @param M the array to sort
    * @param d is the size of the array to sort (power of 2)
    * @param number_of_sub_arrays (power of 2) M is divided into sub arrays that are sorted then merged, 
    * this is the number of sub-arrays to create and it should be chosen with the number of blocks/threads in mind
    * @param n_blocks The number of GPU blocks, this determines the subdivision of M and the number of threads per block
    * @param n_threads The number of threads per block
*/
void Sort_k(const int* M, const size_t d, const size_t n_blocks) {
//  ============================== CPU sorting variables ================================
    size_t n_threads = d/n_blocks;
    size_t size_sub_arrays = d/n_blocks;
    size_t size_M_GPU = n_blocks;
    size_t size_M_GPU_merged = size_M_GPU/2;
//  TODO : change the number of blocks/threads if too big, raise error if too small
//  
//  ============================== GPU variables creation ===============================



//  Creating M_GPU
    testCUDA(cudaMalloc(&M_GPU, n_blocks * sizeof(int*)));
    for (size_t i=0;i<n_blocks;i++) {
        testCUDA(cudaMalloc(tempo_array+i, (size_sub_arrays) * sizeof(int)));
    }
    testCUDA(cudaMemcpy(M_GPU, tempo_array, n_blocks * sizeof(int*), cudaMemcpyHostToDevice));




    for (int i = 0; i < N; i++){
        testCUDA(cudaMemcpy(&M[i], &M_GPU[i], (NA + NB) * sizeof(int), cudaMemcpyDeviceToHost));
    }
//  =========================== Splitting M into sub-arrays =============================
    int** M_splitted = (int**) malloc(n_blocks*sizeof(int*));
    for (size_t i=0;i<n_blocks;i++) {
        M_splitted[i] = M+i*n_threads;
    }
//  ============================= Sorting the sub arrays ================================
//  Here, each block sorts an array
    SortSmall(M_splitted, n_block, n_threads);

//  TODO : free M_splitted

//  ============================= Preparing the GPU arrays ==============================
    //  Single dimension arrays, that store contiguously all the values in M
    int* Actual_M_GPU;
    int* Actual_M_GPU_merged;

    testCUDA(cudaMalloc(&Actual_M_GPU, d * sizeof(int)));
    testCUDA(cudaMemcpy(M, Actual_M_GPU, d * sizeof(int), cudaMemcpyDeviceToHost));

    testCUDA(cudaMalloc(&Actual_M_GPU_merged, d * sizeof(int)));


    //  2-dimensionnal arrays that are on the GPU for the merging
    int* * M_GPU;
    int* * M_GPU_merged;
    testCUDA(cudaMalloc(&M_GPU, size_M_GPU * sizeof(int*)));
    testCUDA(cudaMalloc(&M_GPU_merged, size_M_GPU_merged * sizeof(int*)));

    // this is necessary to fill M_GPU and M_GPU_merged
    void* tempo_array;
    tempo_array = malloc(d*sizeof(int*));
    for (size_t i=0; i<d; i+=size_sub_arrays) {
        tempo_array[i] = Actual_M_GPU + i;
    }
    testCUDA(cudaMemcpy(M_GPU, tempo_array, size_M_GPU * sizeof(int*), cudaMemcpyDeviceToHost));

    size_sub_arrays = size_sub_arrays*2;
    for (size_t i=0; i<d; i+=size_sub_arrays) {
        tempo_array[i] = Actual_M_GPU_merged + i;
    }
    testCUDA(cudaMemcpy(M_GPU_merged, tempo_array, size_M_GPU_merged * sizeof(int*), cudaMemcpyDeviceToHost));

    int** swap_pointer;
//  ============================= Merging the small arrays ==============================
    while ( size_sub_arrays <= n_threads ) {
        mergeSmallBatch_k(M_GPU, &(M_GPU[size_M_GPU_merged]), M_GPU_merged, size_MPU_merged, , size_sub_arrays);

        swap_pointer = Actual_M_GPU;
        Actual_M_GPU = Actual_M_GPU_merged;
        Actual_M_GPU_merged = swap_pointer;
        swap_pointer = M_GPU;
        M_GPU = M_GPU_merged;
        M_GPU_merged = swap_pointer;

        for (size_t i=0; i<d; i+=size_sub_arrays) {
            tempo_array[i] = Actual_M_GPU + i;
        }
        testCUDA(cudaMemcpy(M_GPU, tempo_array, size_M_GPU * sizeof(int*), cudaMemcpyDeviceToHost));
        size_sub_arrays *= 2;
        for (size_t i=0; i<d; i+=size_sub_arrays) {
            tempo_array[i] = Actual_M_GPU_merged + i;
        }
        testCUDA(cudaMemcpy(M_GPU_merged, tempo_array, size_M_GPU_merged * sizeof(int*), cudaMemcpyDeviceToHost));

        size_M_GPU = size_M_GPU_merged;
        size_M_GPU_merged /= 2;
    }

// ============================== Merging the large arrays ==============================
    while ( size_M_GPU > 1 ) {
        mergeBigBatch_k(M_GPU, &(M_GPU[size_M_GPU_merged]), M_GPU_merged, size_MPU_merged, , size_sub_arrays);

        swap_pointer = Actual_M_GPU;
        Actual_M_GPU = Actual_M_GPU_merged;
        Actual_M_GPU_merged = swap_pointer;
        swap_pointer = M_GPU;
        M_GPU = M_GPU_merged;
        M_GPU_merged = swap_pointer;

        for (size_t i=0; i<d; i+=size_sub_arrays) {
            tempo_array[i] = Actual_M_GPU + i;
        }
        testCUDA(cudaMemcpy(M_GPU, tempo_array, size_M_GPU * sizeof(int*), cudaMemcpyDeviceToHost));
        size_sub_arrays *= 2;
        for (size_t i=0; i<d; i+=size_sub_arrays) {
            tempo_array[i] = Actual_M_GPU_merged + i;
        }
        testCUDA(cudaMemcpy(M_GPU_merged, tempo_array, size_M_GPU_merged * sizeof(int*), cudaMemcpyDeviceToHost));

        size_M_GPU = size_M_GPU_merged;
        size_M_GPU_merged /= 2;

    }

//  ================================ Cleaning and end ===================================

    int** M = (int**) malloc(N * sizeof(int*));
    int** tempo = (int**) malloc(N * sizeof(int*));
    testCUDA(cudaMemcpy(tempo, M_GPU, N * sizeof(int*), cudaMemcpyDeviceToHost));
    printf("M created\n");
    for (size_t i=0;i<N;i++) {
        M[i] = (int*) malloc(d * sizeof(int));
        testCUDA(cudaMemcpy(M[i], tempo[i], d * sizeof(int), cudaMemcpyDeviceToHost));
    }
    printf("M filled\n");

}


int main(void) {
    printf("TODO : Delete testCUDA\n");
    printf("TODO : number_of_sub_arrays/2\n");
    printf("TODO : Delete testCUDA\n");
    printf("TODO : Delete testCUDA\n");
    printf("TODO : Delete testCUDA\n");




    return 0;
}