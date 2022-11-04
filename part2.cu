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


__global__ void mergeSmallBatch_k(int** A, int** B, int** M, int N, int d) {
    // Test with shared memory

    //unsigned long idx = threadIdx.x;
    unsigned long tidx = threadIdx.x % d;
    unsigned long Qt = (threadIdx.x - tidx) / d;
    unsigned long gbx = Qt + blockDim.x * blockIdx.x / d; //Index of the array M_gbx

    //printf("My ID : %lu\n", idx);

    if (tidx >= d)
        return;

    int K[2];
    int P[2];

    int NA = sizeof(A[gbx]) / sizeof(int);
    int NB = sizeof(B[gbx]) / sizeof(int);

    if (tidx > NA) {
        K[0] = tidx - NA;
        K[1] = NA;

        P[0] = NA;
        P[1] = tidx - NA;
    } else {
        K[0] = 0;
        K[1] = tidx;

        P[0] = tidx;
        P[1] = 0;
    }

    while (true) {
        int offset = abs((K[1] - P[1]) / 2);
        int Q[] = {K[0] + offset, K[1] - offset};

        if (Q[1] >= 0 && Q[0] <= NB && (Q[1] == NA || Q[0] == 0 || A[gbx][Q[1]] > B[gbx][Q[0] - 1])) {
            if (Q[0] == NB || Q[1] == 0 || A[gbx][Q[1] - 1] <= B[gbx][Q[0]]) {
                if (Q[1] < NA && (Q[0] == NB || A[gbx][Q[1]] <= B[gbx][Q[0]]))
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

void PrintList(int* A, int N){
    
    for (int i = 0 ; i < N ; i++) {
        printf("%d ", A[i]);
    }
    printf("\n");
}

bool IsSorted(int* arr, int size){
    for (int i = 1 ; i < size ; i++){
        if (arr[i-1] > arr[i])
            return false;
    }
    return true;
}

int main()
{
    srand(time(NULL));
    int d = 20;
    int N = 10;
    int** A = (int**) malloc(N * sizeof(int*));
    int NA = 10;
    for (int i = 0 ; i < N ; i++){
        GenerateRandomArray(&A[i], NA);
        if ( NA < 50 ) {
            PrintList(A[i], NA);
        }
    }

    printf("\n");

    int** B = (int**) malloc(N * sizeof(int*));
    int NB = 10;
    for (int i = 0 ; i < N ; i++){
        GenerateRandomArray(&B[i], NB);
        if ( NB < 50 ) {
            PrintList(B[i], NB);
        }
    }

    int** M = (int**) malloc(N * sizeof(int*));

    int** A_GPU, **B_GPU, **M_GPU;

    float TimerV;
    cudaEvent_t start, stop;

    testCUDA(cudaMalloc((void **)&M_GPU, N * sizeof(int*)));
    testCUDA(cudaMalloc((void **)&A_GPU, N * sizeof(int*)));
    testCUDA(cudaMalloc((void **)&B_GPU, N * sizeof(int*)));

    //testCUDA(cudaMemcpy(A_GPU, A, N * sizeof(int*), cudaMemcpyHostToDevice));
    //testCUDA(cudaMemcpy(B_GPU, B, N * sizeof(int*), cudaMemcpyHostToDevice));

    int* A_dummy;

    for (int i = 0; i < N; i++){
        printf("Line %d\n", i);
        //PrintList(A[i], NA);
        testCUDA(cudaMemcpy((void *)&A_GPU[i], (void *)&A[i], NA * sizeof(int), cudaMemcpyHostToDevice)); //PROBLEM, Won't work for i = 6

        testCUDA(cudaMemcpy(&A_dummy, &A_GPU[i], NA * sizeof(int), cudaMemcpyDeviceToHost));

        printf("Copied List : \n");
        PrintList(A_dummy, NA);

        testCUDA(cudaMemcpy((void *)&B_GPU[i], (void *)&B[i], NB * sizeof(int), cudaMemcpyHostToDevice));
    }

    puts("A and B copied");

    int N_Blocks = 1;
    int NTPB = 1024;

    testCUDA(cudaEventCreate(&start));
    testCUDA(cudaEventCreate(&stop));
    testCUDA(cudaEventRecord(start, 0));

    mergeSmallBatch_k<<<N_Blocks, NTPB>>>(A_GPU, B_GPU, M_GPU, N, d);

    testCUDA(cudaEventRecord(stop, 0));
    testCUDA(cudaEventSynchronize(stop));
    testCUDA(cudaEventElapsedTime(&TimerV, start, stop));

    for (int i = 0; i < N; i++){
        testCUDA(cudaMemcpy(&M[i], &M_GPU[i], (NA + NB) * sizeof(int), cudaMemcpyDeviceToHost));
    }

    for (int i = 0 ; i < N ; i++){
        if (NA + NB < 100)
            PrintList(M[i], (NA + NB));

        printf("M[%d] is sorted : %d\n", i, IsSorted(M[i], NA + NB));

    }

    

    printf("Time taken to merge arrays : %f s\n", TimerV / 1000);
    
    for (int i = 0 ; i < N ; i++){
        testCUDA(cudaFree(A_GPU[i]));
        testCUDA(cudaFree(B_GPU[i]));
        testCUDA(cudaFree(M_GPU[i]));
        free(A[i]);
        free(B[i]);
        free(M[i]);
    }


    testCUDA(cudaFree(A_GPU));
    testCUDA(cudaFree(B_GPU));
    testCUDA(cudaFree(M_GPU));
    free(A);
    free(B);
    free(M);

    return 0;
}