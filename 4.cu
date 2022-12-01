#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "utility_functions.cuh"


/*
    * @brief Sorts small arrays with an exact number of blocks/threads
*/
__global__ void SortSmall_k(int **M, int j, int k) {
// Each block sorts one array

//  This is the thread's position in the array
    int tidx = threadIdx.x;

// This is the id of the table that the thread is going to help sorting
    int bid = blockIdx.x;

    //index of i + j
    int swap_index = tidx^j;

    // The threads with the lowest ids sort the array.
    if ((swap_index)>tidx) {
        if ((tidx&k)==0) {
            // Sort the array in ascending order
            if (M[bid][tidx]>M[bid][swap_index]) {
                // Swap index i and i+j
                int temp = M[bid][tidx];
                M[bid][tidx] = M[bid][swap_index];
                M[bid][swap_index] = temp;
            }
        }
        if ((tidx&k)!=0) {
            // Sort the array in descending order
            if (M[bid][tidx]<M[bid][swap_index]) {
                // Swap index i and i+j
                int temp = M[bid][tidx];
                M[bid][tidx] = M[bid][swap_index];
                M[bid][swap_index] = temp;
            }
        }
    }
}

/*
    * @brief Sorts small arrays
    * @param M The array of small arrays
    * @param N The number of small arrays
    * @param d The size of the small arrays (they all have the same)
    * @return A float corresponding to the time taken by the sort
*/
float SortSmall(int **M, size_t N, size_t d) {
//Sorts a group of small arrays using GPU parallelized Bitonic Sort

    int **M_GPU;
    cudaMalloc(&M_GPU, N * sizeof(int*));
    int** tempo_array;
    tempo_array = (int**) malloc(N*sizeof(int*));
    for (int i = 0; i < N; i++){
        cudaMalloc(&tempo_array[i], d * sizeof(int));
        cudaMemcpy(tempo_array[i], M[i], d * sizeof(int), cudaMemcpyHostToDevice);
    }    
    cudaMemcpy(M_GPU, tempo_array, N*sizeof(int*), cudaMemcpyHostToDevice);

//  Timing variables
    float elapsed_time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int j, k;
    cudaEventRecord(start, 0);
    // Progressively increase the size of major bitonic sorted array with each iteration
    for (k = 2; k <= d; k *= 2) {
        //Decrease the sorting step for each iteration to be able to correctly sort major size k arrays from current iteration
        for (j=k/2; j>0; j/=2) {
            SortSmall_k<<<N, d>>>(M_GPU, j, k);
        }
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);


    for (size_t i=0;i<N;i++) {
        cudaMemcpy(M[i], tempo_array[i], d * sizeof(int), cudaMemcpyDeviceToHost);
    }

    for (int i = 0; i < N; i++){
        cudaFree(tempo_array[i]);
    }
    free(tempo_array);
    cudaFree(M_GPU);

    return elapsed_time;
}

int main(void) {
    size_t N = 1024;
    size_t d;
    size_t n_tries = 1000;
    int** array;
    
    array = (int**) malloc(N*sizeof(int*));
    
    float total_time;
    
    printf("Measuring the performance on different d values :\n");
    for (d=4; d<=1024; d*=2) {
        
        total_time = 0.0;
        for (size_t i=0;i<n_tries;i++) {
            for (size_t i=0;i<N;i++) {
                GenerateUnsortedRandomArray(array+i, d);
            }

            total_time += SortSmall(array, N, d);

            for (size_t i=0;i<N;i++) {
                if (! IsSortedAscending(array[i],d) ) {
                    printf("A sub-array isn't sorted... i = %zd\n", i);
                    exit(EXIT_FAILURE);
                }
            }

            for (size_t i=0;i<N;i++) {
                free(array[i]);
            }
        }
        printf("d : %zu, time for %zu runs : %2f s, average time : %2f s\n", d,  n_tries, total_time / 1000.0, (total_time / 1000.0) / n_tries);
    }
    printf("Done without errors!\n");



    free(array);


}