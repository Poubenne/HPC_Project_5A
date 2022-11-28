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


// HERE DUNNO WHY THIS IS NOT NECESSARY
//    if (tidx >= NM[bid] || k >= NM[bid])
    //if (tidx >= NM[bid])
    //    return;


    int ixj = tidx^j;

    /* The threads with the lowest ids sort the array. */
    if ((ixj)>tidx) {
        if ((tidx&k)==0) {
        /* Sort ascending */
        if (M[bid][tidx]>M[bid][ixj]) {
            /* exchange(i,ixj); */
            int temp = M[bid][tidx];
            M[bid][tidx] = M[bid][ixj];
            M[bid][ixj] = temp;
        }
        }
        if ((tidx&k)!=0) {
        /* Sort descending */
        if (M[bid][tidx]<M[bid][ixj]) {
            /* exchange(i,ixj); */
            int temp = M[bid][tidx];
            M[bid][tidx] = M[bid][ixj];
            M[bid][ixj] = temp;
        }
        }
    }
}

/*
    * @brief Sorts small arrays
    * @param M The array of small arrays
    * @param N The number of small arrays
    * @param d The size of the small arrays (they all have the same)
*/
void SortSmall(int **M, size_t N, size_t d) {
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

    int j, k;
    /* Major step */
    for (k = 2; k <= d; k <<= 1) {
        /* Minor step */
        for (j=k>>1; j>0; j=j>>1) {
            SortSmall_k<<<N, d>>>(M_GPU, j, k);
        }
    }

    for (size_t i=0;i<N;i++) {
        cudaMemcpy(M[i], tempo_array[i], d * sizeof(int), cudaMemcpyDeviceToHost);
    }

    for (int i = 0; i < N; i++){
        cudaFree(tempo_array[i]);
    }
    cudaFree(M_GPU);
}

int main(void) {
    size_t N = 1024;
    size_t d = 64;
    int** array;
    
    array = (int**) malloc(N*sizeof(int*));
    
    float TimerV;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    
    
    printf("Measuring the performance on different d values :\n\tWarning! the performance measure takes into account the memory allocation time\n");
    for (d=4; d<=1024; d*=2) {
        for (size_t i=0;i<N;i++) {
            GenerateUnsortedRandomArray(array+i, d);
        }

        cudaEventRecord(start, 0);
        SortSmall(array, N, d);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&TimerV, start, stop);
        printf("d : %zu, time : %2f s\n", d,TimerV / 1000.0);

        for (size_t i=0;i<N;i++) {
            if (! IsSortedAscending(array[i],d) ) {
                printf("A sub-array isn't sorrted... i = %zd\n", i);
                exit(EXIT_FAILURE);
            }
        }

        for (size_t i=0;i<N;i++) {
            free(array[i]);
        }
    }
    printf("Done without errors!\n");



    free(array);


}