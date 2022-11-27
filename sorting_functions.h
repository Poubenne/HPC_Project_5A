#include <stdlib.h>
#include <stdio.h>
#include <time.h>

/*
    Header files shouldn't contain code, but this is a small project, so whatever
*/




//  ===================================================   Sorting small arrays   ==================================================================
__global__ void SortSmall_k(int **M, const size_t NM, int j, int k){
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
    * @brief Sorts a group of small arrays using GPU parallelized Bitonic Sort
    * @param M contain the arrays to sort
    * @param n_arrays The number of arrays to sort, it's also the number of blocks
    * @param size_arrays The size of the arrays to sort (they all have the same), it's also the number of threads per block
*/
void SortSmall(int **M, const size_t n_arrays, const size_t size_arrays) {
    int **M_GPU;
    int** tempo_array;

    tempo_array = (int**) malloc(n_arrays*sizeof(int*));
    testCUDA(cudaMalloc(&M_GPU, n_arrays * sizeof(int*)));
    
    for (int i = 0; i < n_arrays; i++){
        testCUDA(cudaMemcpy(tempo_array[i], M[i], size_arrays * sizeof(int), cudaMemcpyHostToDevice));
    }
    testCUDA(cudaMemcpy(M_GPU, tempo_array, n_arrays*sizeof(int*), cudaMemcpyHostToDevice));

    int j, k;
    /* Major step */
    for (k = 2; k <= size_arrays; k <<= 1) {
        /* Minor step */
        for (j=k>>1; j>0; j=j>>1) {
            SortSmall_k<<<n_arrays, size_arrays>>>(M_GPU, size_arrays, j, k);
//            SortSmall_k<<<n_arrays, size_arrays>>>(M_GPU, size_arrays, j, k);
//            SortSmall_k<<<N, NTPB>>>(M_GPU, NM_GPU, j, k);
        }
    }

    for (size_t i=0;i<n_arrays;i++) {
//        testCUDA(cudaMemcpy(M[i], tempo_array[i], size_arrays * sizeof(int), cudaMemcpyDeviceToHost));
        testCUDA(cudaMemcpy(M[i], M_GPU[i], size_arrays * sizeof(int), cudaMemcpyDeviceToHost));
    }

//TODO
/*  
    for (int i = 0; i < N; i++){
        //testCUDA(cudaFree(tempo_array+i));
        testCUDA(cudaFree(M_GPU[i]));
    }
    puts("Check");

    free(tempo_array);
    //free(tempo);
    testCUDA(cudaFree(NM_GPU));
    testCUDA(cudaFree(M_GPU));
*/
}

//  =========================================================== Merging small arrays ================================================================
/*
    * @brief Merges two batches of same sized sorted arrays to create a single batch of sorted arrays
    * @param A First batch of sorted arrays
    * @param B Second batch of sorted arrays
    * @param M The batch of merged arrays
    * @param n_arrays (power of 2) The number of arrays per batch
    * @param size_arrays (power of 2) The size of the arrays, since they all have the same
*/
__global__ void mergeSmallBatch_k(const int** A, const int** B, int** M, const size_t n_arrays, const size_t size_arrays) {
//  Since the number of threads per block is a multiple of d, that means a single block can merge several arrays

//  TODOTEST : shared , or global
    size_t d = 2*size_arrays;

//  This is the thread's position in the array
    unsigned int tidx = threadIdx.x%d;
//  This is the index of the array from the arrays given to a specific block
    unsigned int Qt = (threadIdx.x-tidx)/d;
//  This is the "global" index of the array, in which the thread is going to work
    unsigned int gbx = Qt + blockIdx.x*(blockDim.x/d);

//  the thread works on M[gbx][tidx]

//  We pick specific sizes of arrays, so this is not necessary and result in a lost of performance
//  TODO : delete
    if ( gbx >= N ) {// excedent block
        return;
    }

    int K[2];
    int P[2];

    if (tidx > size_arrays) {
        K[0] = tidx - size_arrays;
        K[1] = size_arrays;

        P[0] = size_arrays;
        P[1] = tidx - size_arrays;
    } else {
        K[0] = 0;
        K[1] = tidx;

        P[0] = tidx;
        P[1] = 0;
    }

    while (true) {
        int offset = abs((K[1] - P[1]) / 2);
        int Q[] = {K[0] + offset, K[1] - offset};

        if (Q[1] >= 0 && Q[0] <= size_arrays && (Q[1] == size_arrays || Q[0] == 0 || A[gbx][Q[1]] > B[gbx][Q[0] - 1])) {
            if (Q[0] == size_arrays || Q[1] == 0 || A[gbx][Q[1] - 1] <= B[gbx][Q[0]]) {
                if (Q[1] < size_arrays && (Q[0] == size_arrays || A[gbx][Q[1]] <= B[gbx][Q[0]]))
                    M[gbx][tidx] = A[gbx][Q[1]];
                else
                    M[gbx][tidx] = B[gbx][Q[0]];
                break;
            } else {
                K[0] = Q[0] + 1;
                K[1] = Q[1] - 1;
            }
        } else {
            P[0] = Q[0] - 1;
            P[1] = Q[1] + 1;
        }
    }
}

//  =========================================================== Merging large arrays ================================================================

/*
    * @brief Merges two arrays of int
    * @param A the first arrays to merge
    * @param B the second arrays to merge
    * @param n_arrays the number of M_i arrays (also the number of A_i and B_i)
    * @param size_arrays the size of the A_i and B_i arrays, since they're all supposed to have the same
    * @param d = 2*size_arrays The sizes of M_i arrays
*/
__global__ void mergeBigBatch_k(const int** A, const int** B, int** M, const size_t n_arrays, const size_t size_arrays, const size_t d) {
//  Several blocks have to merge a single array

//  This is the thread's position in the merged array
    size_t thread_position = threadIdx.x + blockIdx.x * blockDim.x;
    size_t array_id = blockIdx.x / (d/blockDim.x);

//  We pick specific sizes of arrays, so this is not necessary and result in a lost of performance
//  TODO : delete
    if ( array_id >= n_arrays ) {
        return;
    }

    int K[2];
    int P[2];

    if (thread_position > size_arrays) {
        K[0] = thread_position - size_arrays;
        K[1] = size_arrays;

        P[0] = size_arrays;
        P[1] = thread_position - size_arrays;
    } else {
        K[0] = 0;
        K[1] = thread_position;

        P[0] = thread_position;
        P[1] = 0;
    }

    while (true) {
        int offset = abs((K[1] - P[1]) / 2);
        int Q[] = {K[0] + offset, K[1] - offset};

        if (Q[1] >= 0 && Q[0] <= size_arrays && (Q[1] == size_arrays || Q[0] == 0 || A[array_id][Q[1]] > B[array_id][Q[0] - 1])) {
            if (Q[0] == size_arrays || Q[1] == 0 || A[array_id][Q[1] - 1] <= B[array_id][Q[0]]) {
                if (Q[1] < size_arrays && (Q[0] == size_arrays || A[array_id][Q[1]] <= B[array_id][Q[0]]))
                    M[array_id][thread_position] = A[array_id][Q[1]];
                else
                    M[array_id][thread_position] = B[array_id][Q[0]];
                break;
            } else {
                K[0] = Q[0] + 1;
                K[1] = Q[1] - 1;
            }
        } else {
            P[0] = Q[0] - 1;
            P[1] = Q[1] + 1;
        }
    }
}
