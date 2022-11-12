#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#define N 1024
#define d 8

void testCUDA(cudaError_t error, const char *file, int line)  {
	if (error != cudaSuccess) {
	   printf("There is an error in file %s at line %d\n", file, line);
       exit(EXIT_FAILURE);
	} 
}

#define testCUDA(error) (testCUDA(error, __FILE__ , __LINE__))

void QuickSort( int* array , size_t end) {
    if (end > 1) {
        int pivot = array[end-1];

        size_t cursor = 0;
        int swap_variable;
        for (size_t i = 0; i < end - 1 ; i++) {
            if ( array[i] < pivot ) {
                swap_variable = array[i];
                array[i] = array[cursor];
                array[cursor] = swap_variable;

                cursor = cursor + 1;
            }
        }
        swap_variable = array[end-1];
        array[end-1] = array[cursor];
        array[cursor] = swap_variable;

        QuickSort(array , cursor );
        QuickSort(array + cursor + 1, end - cursor - 1);
    }
}

bool IsSortedAscending(int* array, size_t length) {
    for(size_t i=1; i<length; i++) {
        if (array[i-1]>array[i]) {
            return false;
        }
    }
    return true;
}

bool IsSortedDescending(int* array, size_t length) {
    for(size_t i=1; i<length; i++) {
        if (array[i-1]<array[i]) {
            return false;
        }
    }
    return true;
}

/*
    * @brief Generates a random array
    * @param arr the array that'll be malloced and filled
    * @param length The size of the array
    */
void GenerateRandomArray( int** arr, const size_t length ) {
    *arr = (int*) malloc(length*sizeof(int));

    for (size_t i=0 ; i<length ; i++) {
        (*arr)[i] = rand()%(5*length);
    }
    QuickSort(*arr, length);
}

void PrintList(int* A, size_t length){
    
    for (int i = 0; i < length; i++) {
        printf("%d ", A[i]);
    }
    printf("\n");
}



__global__ void mergeSmallBatch_k(const int** A, const int** B, int** M, const size_t* NA, const size_t* NB) {
//  Since the number of threads per block is a multiple of d, that means a single block can merge several arrays

//  This is the thread's position in the array
    int tidx = threadIdx.x%d;
//  This is the index of the array from the arrays given to a specific block
    int Qt = (threadIdx.x-tidx)/d;
//  This is the "global" index of the array, in which the thread is going to work
    int gbx = Qt + blockIdx.x*(blockDim.x/d);

//  the thread works on M[gbx][tidx]

    if ( gbx >= N ) {// excedent block
        return;
    }

/*    if (idx >= NA + NB)
        return;*/

    int K[2];
    int P[2];

    if (tidx > NA[gbx]) {
        K[0] = tidx - NA[gbx];
        K[1] = NA[gbx];

        P[0] = NA[gbx];
        P[1] = tidx - NA[gbx];
    } else {
        K[0] = 0;
        K[1] = tidx;

        P[0] = tidx;
        P[1] = 0;
    }

    while (true) {
        int offset = abs((K[1] - P[1]) / 2);
        int Q[] = {K[0] + offset, K[1] - offset};

        if (Q[1] >= 0 && Q[0] <= NB[gbx] && (Q[1] == NA[gbx] || Q[0] == 0 || A[gbx][Q[1]] > B[gbx][Q[0] - 1])) {
            if (Q[0] == NB[gbx] || Q[1] == 0 || A[gbx][Q[1] - 1] <= B[gbx][Q[0]]) {
                if (Q[1] < NA[gbx] && (Q[0] == NB[gbx] || A[gbx][Q[1]] <= B[gbx][Q[0]]))
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



int main()
{
    printf("Creating arrays...\t");
    srand(time(NULL));
    int a_sizes[] = {1, 4, 6};

    int** A = (int**) malloc(N*sizeof(int*));
    size_t* NA = (size_t*) malloc(N*sizeof(size_t));
    for (size_t i=0;i<N;i++) {
        NA[i] = a_sizes[i%3];
        GenerateRandomArray(A+i, NA[i]);
    }


    int** B = (int**) malloc(N*sizeof(int*));
    size_t* NB = (size_t*) malloc(N*sizeof(size_t));
    for (size_t i=0;i<N;i++) {
        NB[i] = d - NA[i];
        GenerateRandomArray(B+i, NB[i]);
    }
    printf(" Done!\n");


//  We could create contiguous arrays...
    printf("Creating GPU arrays...\t");
    int** A_GPU;
    int** B_GPU;
    int** M_GPU;

    size_t* NA_GPU;
    size_t* NB_GPU;

    testCUDA(cudaMalloc(&NA_GPU, N * sizeof(size_t)));
    testCUDA(cudaMalloc(&NB_GPU, N * sizeof(size_t)));


    int** tempo_array;
    tempo_array = (int**) malloc(N*sizeof(int*));

//  Creating M_GPU
    printf("Creating M_GPU...\t");
    testCUDA(cudaMalloc(&M_GPU, N * sizeof(int*)));
    for (size_t i=0;i<N;i++) {
        testCUDA(cudaMalloc(tempo_array+i, (NB[i]+NA[i]) * sizeof(int)));
    }
    testCUDA(cudaMemcpy(M_GPU, tempo_array, N * sizeof(int*), cudaMemcpyHostToDevice));
    printf("Done!\n");

//  Creating A_GPU
    printf("Creating A_GPU...\t");
    testCUDA(cudaMalloc(&A_GPU, N * sizeof(int*)));
    for (size_t i=0;i<N;i++) {
        testCUDA(cudaMalloc(tempo_array+i, NA[i] * sizeof(int)));
        testCUDA(cudaMemcpy(tempo_array[i], A[i], NA[i] * sizeof(int), cudaMemcpyHostToDevice));
    }
    testCUDA(cudaMemcpy(A_GPU, tempo_array, N * sizeof(int*), cudaMemcpyHostToDevice));
    printf("Done!\n");

//  Creating B_GPU
    printf("Creating B_GPU...\t");
    testCUDA(cudaMalloc(&B_GPU, N * sizeof(int*)));
    for (size_t i=0;i<N;i++) {
        testCUDA(cudaMalloc(tempo_array+i, NB[i] * sizeof(int)));
        testCUDA(cudaMemcpy(tempo_array[i], B[i], NB[i] * sizeof(int), cudaMemcpyHostToDevice));
    }
    testCUDA(cudaMemcpy(B_GPU, tempo_array, N * sizeof(int*), cudaMemcpyHostToDevice));
    printf("Done!\n");

//  Filling the arrays 
    printf("Copying the sizes...\t");
    testCUDA(cudaMemcpy(NA_GPU, NA, N * sizeof(size_t), cudaMemcpyHostToDevice));
    testCUDA(cudaMemcpy(NB_GPU, NB, N * sizeof(size_t), cudaMemcpyHostToDevice));
    printf(" Done!\n");



    int N_Blocks = N/d+10;
    int NTPB = 1024;
    
    printf("Merging...\t");
    mergeSmallBatch_k<<<N_Blocks, NTPB>>>((const int**)A_GPU, (const int**)B_GPU, M_GPU, NA_GPU, NB_GPU);
    printf(" Done!\n");
    

    int** M = (int**) malloc(N * sizeof(int*));
    int** tempo = (int**) malloc(N * sizeof(int*));
    testCUDA(cudaMemcpy(tempo, M_GPU, N * sizeof(int*), cudaMemcpyDeviceToHost));
    printf("M created\n");
    for (size_t i=0;i<N;i++) {
        M[i] = (int*) malloc(d * sizeof(int));
        testCUDA(cudaMemcpy(M[i], tempo[i], d * sizeof(int), cudaMemcpyDeviceToHost));
    }
    printf("M filled\n");



//    PrintList(M, (NA + NB));
    printf("Verifying result\n");
    for (size_t i=0;i<N;i++) {  
        if (! IsSortedAscending(M[i],d) ) {
            printf("The result isn't correct...\n");
            exit(EXIT_FAILURE);
        }
    }
    printf("The result is correct!\n");



    testCUDA(cudaFree(A_GPU));
    testCUDA(cudaFree(B_GPU));
    testCUDA(cudaFree(M_GPU));
    

    for (size_t i=0;i<N;i++) {
        free(A[i]);
        free(B[i]);
    }
    free(A);
    free(B);
    free(M);

    return 0;
}