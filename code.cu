#include <stdlib.h>
#include <stdio.h>
#include <time.h>

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
        QuickSort(array + cursor , end - cursor - 1);
    }
}

/*
    * @brief Generates a random array
    * @param arr the array that'll be malloced and filled
    * @param The size of the array
    */
void GenerateRandomArray( int** arr, const size_t size ) {
    srand(time(NULL));
    *array = (int*) malloc(size*sizeof(int));

    for (size_t i=0 ; i<size ; i++) {
        (*arr)[i] = rand()%(5*size);
    }
    QuickSort(arr, size);
}


__global__ void mergeSmall_k(const int* A, const int* B, int* M, const int NA, const int NB) {
    // Test with shared memory

    unsigned long idx = threadIdx.x;

    if (idx >= NA + NB)
        return;

    int K[2];
    int P[2];

    if (idx > NA) {
        K[0] = idx - NA;
        K[1] = NA;

        P[0] = NA;
        P[1] = idx - NA;
    } else {
        K[0] = 0;
        K[1] = idx;

        P[0] = idx;
        P[1] = 0;
    }

    while (true) {
        int offset = abs((K[1] - P[1]) / 2);
        int Q[] = {K[0] + offset, K[1] - offset};

        if (Q[1] >= 0 && Q[0] <= NB && (Q[1] == NA || Q[0] == 0 || A[Q[1]] > B[Q[0] - 1])) {
            if (Q[0] == NB || Q[1] == 0 || A[Q[1] - 1] <= B[Q[0]]) {
                if (Q[1] < NA && (Q[0] == NB || A[Q[1]] <= B[Q[0]]))
                    M[idx] = A[Q[1]];
                else
                    M[idx] = B[Q[0]];
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

void PrintList(int* A, int N){
    
    for (int i = 0; i < N; i++) {
        printf("%d ", A[i]);
    }
    printf("\n");
}

int main()
{

    int* A;
    int NA = 10;
    GenerateRandomArray(&A, NA);
    if ( NA < 50 ) {
        PrintList(A, NA);
    }


    int* B;
    int NB = 10;
    GenerateRandomArray(&B, NB);
    if ( NB < 50 ) {
        PrintList(B, NB);
    }

    int* M = (int*) malloc((NA + NB) * sizeof(int));

    int* A_GPU, *B_GPU, *M_GPU;

    testCUDA(cudaMalloc(&M_GPU, (NA + NB) * sizeof(int)));
    testCUDA(cudaMalloc(&A_GPU, NA * sizeof(int)));
    testCUDA(cudaMalloc(&B_GPU, NB * sizeof(int)));


    testCUDA(cudaMemcpy(A_GPU, A, NA * sizeof(int), cudaMemcpyHostToDevice));
    testCUDA(cudaMemcpy(B_GPU, B, NB * sizeof(int), cudaMemcpyHostToDevice));

    int NB = 1;
    int NTPB = 1024;

    mergeSmall_k<<<NB, NTPB>>>(A_GPU, B_GPU, M_GPU, NA, NB);

    testCUDA(cudaMemcpy(M, M_GPU, (NA + NB) * sizeof(int), cudaMemcpyDeviceToHost));

    PrintList(M, (NA + NB));

    testCUDA(cudaFree(A_GPU));
    testCUDA(cudaFree(B_GPU));
    testCUDA(cudaFree(M_GPU));
    free(M);

    return 0;
}