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
void Sort_k(const int* M, const size_t d, const size_t n_blocks, const size_t n_threads) {
//  ============================== CPU sorting variables ================================
    size_t n_threads = ;
    size_t size_sub_arrays = d/n_blocks;
//  TODO : change the number of blocks/threads if too big, raise error if too small
//  
//  ============================== GPU variables creation ===============================
    int** M_GPU;

    void* tempo_array;
    tempo_array = malloc(n_blocks*sizeof(int*));

//  Creating M_GPU
    testCUDA(cudaMalloc(&M_GPU, n_blocks * sizeof(int*)));
    for (size_t i=0;i<n_blocks;i++) {
        testCUDA(cudaMalloc(tempo_array+i, (size_sub_arrays) * sizeof(int)));
    }
    testCUDA(cudaMemcpy(M_GPU, tempo_array, n_blocks * sizeof(int*), cudaMemcpyHostToDevice));


    SortSmall(M, const size_t* NM, n_block, n_threads);


    for (int i = 0; i < N; i++){
        testCUDA(cudaMemcpy(&M[i], &M_GPU[i], (NA + NB) * sizeof(int), cudaMemcpyDeviceToHost));
    }

//  ============================= Sorting the sub arrays ================================
//  Here, each block sorts an array
//  TODO : blocks are preferably of size > 32 : add a check, that changes number_of_sub_arrays, if it's too high, and prints a warning


//  ============================= Merging the small arrays ==============================
    int** M_GPU_merged;

    while ( size_sub_arrays <= n_threads ) {
        free(tempo_array);
        tempo_array = malloc(n_blocks/2*sizeof(int*));
        testCUDA(cudaMalloc(&M_GPU_merged, n_blocks/2 * sizeof(int*)));
        for (size_t i=0;i<n_blocks/2;i++) {
            testCUDA(cudaMalloc(tempo_array+i, 2 * size_sub_arrays * sizeof(int)));
        }
        testCUDA(cudaMemcpy(M_GPU_merged, tempo_array, n_blocks/2 * sizeof(int*), cudaMemcpyHostToDevice));

        //testCUDA(cudaFree(M_GPU[i]));
        // cudafree(M_GPU)
        M_GPU = M_GPU_merged;
    }

// ============================== Merging the large arrays ==============================


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