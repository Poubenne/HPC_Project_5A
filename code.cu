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
        QuickSort(array + cursor + 1, end - cursor - 1);
    }
}

/*
    * @brief Generates a random array
    * @param arr the array that'll be malloced and filled
    * @param The size of the array
    */
void GenerateRandomArray( int** arr, const size_t size ) {
    *arr = (int*) malloc(size*sizeof(int));

    for (size_t i=0 ; i<size ; i++) {
        (*arr)[i] = rand()%(5*size);
    }
    QuickSort(*arr, size);
}


__global__ void mergeSmall_k(const int* A, const int* B, int* M, const int NA, const int NB) {
    // Test with shared memory

    //unsigned long idx = threadIdx.x;
    unsigned long idx = blockIdx.x * blockDim.x + threadIdx.x;
    //printf("My ID : %lu\n", idx);

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

bool IsSorted(int* arr, int size){
    for (int i=1;i<size;i++){
        if (arr[i-1] > arr[i])
            return false;
    }
    return true;
}

int main()
{
    srand(time(NULL));
    int* A;
    int NA = 1024;
    GenerateRandomArray(&A, NA);
    if ( NA < 50 ) {
        PrintList(A, NA);
    }


    int* B;
    int NB = 1024;
    GenerateRandomArray(&B, NB);
    if ( NB < 50 ) {
        PrintList(B, NB);
    }

    int* M = (int*) malloc((NA + NB) * sizeof(int));

    int* A_GPU, *B_GPU, *M_GPU;

    float TimerV;
    cudaEvent_t start, stop;

    testCUDA(cudaMalloc(&M_GPU, (NA + NB) * sizeof(int)));
    testCUDA(cudaMalloc(&A_GPU, NA * sizeof(int)));
    testCUDA(cudaMalloc(&B_GPU, NB * sizeof(int)));


    testCUDA(cudaMemcpy(A_GPU, A, NA * sizeof(int), cudaMemcpyHostToDevice));
    testCUDA(cudaMemcpy(B_GPU, B, NB * sizeof(int), cudaMemcpyHostToDevice));

    int N_Blocks = 2;
    int NTPB = 1024;

    testCUDA(cudaEventCreate(&start));
    testCUDA(cudaEventCreate(&stop));
    testCUDA(cudaEventRecord(start, 0));

    mergeSmall_k<<<N_Blocks, NTPB>>>(A_GPU, B_GPU, M_GPU, NA, NB);

    testCUDA(cudaEventRecord(stop, 0));
    testCUDA(cudaEventSynchronize(stop));
    testCUDA(cudaEventElapsedTime(&TimerV, start, stop));


    testCUDA(cudaMemcpy(M, M_GPU, (NA + NB) * sizeof(int), cudaMemcpyDeviceToHost));

    if (NA + NB < 100)
        PrintList(M, (NA + NB));

    printf("M is sorted : %d\n", IsSorted(M, NA + NB));

    printf("Time taken to merge arrays : %f s\n", TimerV / 1000);
    
    testCUDA(cudaFree(A_GPU));
    testCUDA(cudaFree(B_GPU));
    testCUDA(cudaFree(M_GPU));
    free(M);

    return 0;
}