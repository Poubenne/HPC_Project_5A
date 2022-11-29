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
//    printf("n_threads : %u\n", n_threads);
    // this is the size of the sub-arrays in M_GPU
    size_t size_sub_arrays = d/n_blocks;
//    printf("size_sub_arrays : %zu\n", size_sub_arrays);
    //  M_GPU is a pointer of pointer : this is the number of pointer/sub-arrays
    size_t size_M_GPU = n_blocks;
//    printf("size_M_GPU : %zu\n", size_M_GPU);
    //  This is the number of pointer/sub-arrays for next iteration
    size_t size_M_GPU_merged = size_M_GPU/2;
//    printf("size_M_GPU_merged : %zu\n", size_M_GPU_merged);

//  Used in the loops;
    size_t j;
//  =========================== Splitting M into sub-arrays =============================
    int** M_splitted = (int**) malloc(n_blocks*sizeof(int*));
    for (size_t i=0;i<n_blocks;i++) {
        M_splitted[i] = M+i*n_threads;
    }
//  ============================= Sorting the sub arrays ================================
//  Here, each block sorts an array
    SortSmall(M_splitted, n_blocks, n_threads);

//  TODO : free M_splitted
    //free(M_splitted);

//  ============================= Preparing the GPU arrays ==============================
    //  Single dimension arrays, that store contiguously all the values in M
    int* Actual_M_GPU;
    int* Actual_M_GPU_merged;

    testCUDA(cudaMalloc(&Actual_M_GPU, d * sizeof(int)));
    testCUDA(cudaMemcpy(M, Actual_M_GPU, d * sizeof(int), cudaMemcpyDeviceToHost));

    testCUDA(cudaMalloc(&Actual_M_GPU_merged, d * sizeof(int)));


    //  2-dimensional arrays that are on the GPU for the merging
    int* * M_GPU;
    int* * M_GPU_merged;
    testCUDA(cudaMalloc(&M_GPU, size_M_GPU * sizeof(int*)));
    testCUDA(cudaMalloc(&M_GPU_merged, size_M_GPU_merged * sizeof(int*)));

    // this is necessary to fill M_GPU and M_GPU_merged
    int** tempo_array;
    tempo_array = (int**) malloc(size_M_GPU * sizeof(int*));


    j = 0;
    for (size_t i=0; i<d; i+=size_sub_arrays) {
        tempo_array[j] = Actual_M_GPU + i;
        j++;
    }
    testCUDA(cudaMemcpy(M_GPU, tempo_array, size_M_GPU * sizeof(int*), cudaMemcpyHostToDevice));

    size_sub_arrays = size_sub_arrays*2;
    j=0;
    for (size_t i=0; i<d; i+=size_sub_arrays) {
        tempo_array[j] = Actual_M_GPU_merged + i;
        j++;
    }
    testCUDA(cudaMemcpy(M_GPU_merged, tempo_array, size_M_GPU_merged * sizeof(int*), cudaMemcpyHostToDevice));

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
        j=0;
        for (size_t i=0; i<d; i+=size_sub_arrays) {
            tempo_array[j] = Actual_M_GPU + i;
            j++;
        }
        testCUDA(cudaMemcpy(M_GPU, tempo_array, size_M_GPU * sizeof(int*), cudaMemcpyHostToDevice));
        size_sub_arrays *= 2;
        j=0;
        for (size_t i=0; i<d; i+=size_sub_arrays) {
            tempo_array[j] = Actual_M_GPU_merged + i;
            j++;
        }
        testCUDA(cudaMemcpy(M_GPU_merged, tempo_array, size_M_GPU_merged * sizeof(int*), cudaMemcpyHostToDevice));
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
        j=0;
        for (size_t i=0; i<d; i+=size_sub_arrays) {
            tempo_array[j] = Actual_M_GPU + i;
            j++;
        }
        testCUDA(cudaMemcpy(M_GPU, tempo_array, size_M_GPU * sizeof(int*), cudaMemcpyHostToDevice));
        size_sub_arrays *= 2;
        j=0;
        for (size_t i=0; i<d; i+=size_sub_arrays) {
            tempo_array[j] = Actual_M_GPU_merged + i;
            j++;
        }
        testCUDA(cudaMemcpy(M_GPU_merged, tempo_array, size_M_GPU_merged * sizeof(int*), cudaMemcpyHostToDevice));
    }

//  ================================ Cleaning and end ===================================
    // TODO : free


    testCUDA(cudaMemcpy(M, Actual_M_GPU, d * sizeof(int), cudaMemcpyDeviceToHost));
}


int main(void) {
    printf("TODO : Delete testCUDA\n");
    printf("TODO : number_of_sub_arrays/2\n");
    printf("TODO : change n_blocks to size sub_arrays and add number_of_blocks\n");
    printf("TODO : sorting_functions.h -> .cuh\n");
    printf("TODO : delete the useless tests\n");
    printf("TODO : m_gpu à l'itération suivante est juste m_gpu_merged de cette itération\n");

    size_t size = 4096*4096;
    size_t nblocks = 512;

    int* M;
    GenerateUnsortedRandomArray(&M, size);


    Sort_k( M, size, nblocks);

    if (IsSorted(M, size)) {
        printf("M is sorted!\n");
    } else {
        printf("M isn't sorted...\n");
    }

    return 0;
}