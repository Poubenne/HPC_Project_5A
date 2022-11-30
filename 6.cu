#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "utility_functions.cuh"
#include "sorting_functions.cuh"


/*
    * @brief Sort an array, using the GPU
    * @param M the array to sort
    * @param d is the size of the array to sort (power of 2)
*/
void Sort_k(int* M, const size_t d) {
//  ============================== CPU sorting variables ================================
//  theoretical limits : (65535 =>) 32768 blocks * 1024 threads
//  compute capability 3 : 2147483648-1 => 1073741824
    unsigned int maximum_number_of_blocks = 32768;
    unsigned int maximum_number_of_threads = 1024;

    if ( d >= maximum_number_of_blocks*maximum_number_of_threads ) {return;}
    
    unsigned int initial_subarrays_size = 1024;
    unsigned int initial_split_size = d/initial_subarrays_size;
    int** M_splitted;
//  if possible : (m.size < 2097120 =32*65535 and m.size > 32)
    if ( d < initial_subarrays_size*maximum_number_of_blocks ) {// if the array is small enough
        if ( d > initial_subarrays_size ) { //  if the array is big enough
            //  subdivise M into 32-sized arrays, and sort them

            M_splitted = (int**) malloc(initial_split_size*sizeof(int*));
            for (size_t i=0;i<initial_split_size;i++) {
                M_splitted[i] = M+i*(initial_subarrays_size);
            }

            SortSmall(M_splitted, initial_split_size, initial_subarrays_size);
            free(M_splitted);
        } else {//M is actually small, so we don't actually care, we sort with a single block
            SortSmall(&M, 1, d);
        }
    } else {//  M is too big, so we split into 1024-sized arrays
        unsigned int initial_subarrays_size = 1024;
        unsigned int initial_split_size = d/initial_subarrays_size;
        //  subdivise M into 1024-sized arrays, and sort them

        M_splitted = (int**) malloc(initial_split_size*sizeof(int*));
        for (size_t i=0;i<initial_split_size;i++) {
            M_splitted[i] = M+i*(initial_subarrays_size);
        }

        SortSmall(M_splitted, initial_split_size, initial_subarrays_size);
        free(M_splitted);
    }

//  then, merge the arrays, using max-threaded blocks
    // number of threads per block

    unsigned int n_threads;
    if ( d < maximum_number_of_threads ) {
        n_threads = d;
    } else {
        n_threads = maximum_number_of_threads;
    }
    unsigned int n_blocks = d/n_threads;

    // this is the size of the sub-arrays in M_GPU
    size_t size_sub_arrays = initial_subarrays_size;

    //  M_GPU is a pointer of pointer : this is the number of pointer/sub-arrays
    size_t size_M_GPU = initial_split_size;

    //  This is the number of pointer/sub-arrays for next iteration
    size_t size_M_GPU_merged = size_M_GPU/2;

//  Used in the loops;
    size_t j;
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

//  Timing variables
    float elapsed_time;
    float quicksort_time;
    cudaEvent_t start, stop;
//  no cudatest in the final version
//  testCUDA(cudaMalloc(&M_GPU, N * sizeof(int*)));
//  testCUDA(cuda_function);
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float total_time;

    int* M;

    for (d=4; d<=64*4096; d*=2) {
        GenerateUnsortedRandomArray(&M, d);
        cudaEventRecord(start, 0);

        QuickSort(M, d);

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&quicksort_time, start, stop);
        total_time = 0.0;
        for (size_t try_number=0;try_number<n_tries;try_number++) {
            ReRandomizeArray(M, d);

            cudaEventRecord(start, 0);

            Sort_k( M, d);

            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&elapsed_time, start, stop);

            total_time += elapsed_time;

            if (IsSorted(M, d)) {
                continue;
            } else {
                printf("M, of size %zu, isn't sorted...\n", d);
                exit(EXIT_FAILURE);
            }
        }
        free(M);
        printf("d : %zu, time for %zu runs : %2f s, average time : %2f s\n\t(Quicksort took %2fs)\n", d,  n_tries, total_time / 1000.0, (total_time / 1000.0) / n_tries, quicksort_time/1000.0);
            //printf("%2f, ",(total_time / 1000.0) / n_tries);
    }


    return 0;
}