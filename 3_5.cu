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

/*
    * @brief Merges two arrays of int
    * @param A the first array to merge
    * @param B the second array to merge
    * @param NA the size of A
    * @param NB the size of B
    * @param d = NA + NB is the size of M
*/
__global__ void mergeAB(const int* A, const int* B, int* M, const size_t NA, const size_t NB, const size_t d) {
/*Here each thread will be in charge of one diagonal of the merge path of M, we can't split into blocks since M is too big*/


//  This is the thread's global position
    size_t thread_position = threadIdx.x + blockIdx.x * blockDim.x;

    if ( thread_position >= d ) {// thread, if d isn't a power of 2
        return;
    }

    int K[2];
    int P[2];

    if (thread_position > NA) {
        K[0] = thread_position - NA;
        K[1] = NA;

        P[0] = NA;
        P[1] = thread_position - NA;
    } else {
        K[0] = 0;
        K[1] = thread_position;

        P[0] = thread_position;
        P[1] = 0;
    }

    while (true) {
        int offset = abs((K[1] - P[1]) / 2);
        int Q[] = {K[0] + offset, K[1] - offset};

        if (Q[1] >= 0 && Q[0] <= NB && (Q[1] == NA || Q[0] == 0 || A[Q[1]] > B[Q[0] - 1])) {
            if (Q[0] == NB || Q[1] == 0 || A[Q[1] - 1] <= B[Q[0]]) {
                if (Q[1] < NA && (Q[0] == NB || A[Q[1]] <= B[Q[0]]))
                    M[thread_position] = A[Q[1]];
                else
                    M[thread_position] = B[Q[0]];
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
    srand(time(NULL));

    int N_Blocks = 5;
    int NTPB = 1024;
    /*
    N_Blocks*NTPB = d size of the large array
    Each available thread has to be assigned one index of the merged array
    */

    size_t d = N_Blocks * NTPB;
    size_t N = d;

    size_t NA = d/3;
    int* A = (int*) malloc(NA * sizeof(int));
    GenerateRandomArray(&A, NA);

    size_t NB = d-NA;
    int* B = (int*) malloc(N * sizeof(int));
    GenerateRandomArray(&B, NB);

    int* M = (int*) malloc(N * sizeof(int));

    int* A_GPU, *B_GPU, *M_GPU;

    int niter = 300000;
    float TimerV;
    cudaEvent_t start, stop;

    testCUDA(cudaMalloc(&M_GPU, N * sizeof(int)));
    testCUDA(cudaMalloc(&A_GPU, NA * sizeof(int)));
    testCUDA(cudaMalloc(&B_GPU, NB * sizeof(int)));

    testCUDA(cudaMemcpy(A_GPU, A, NA * sizeof(int), cudaMemcpyHostToDevice));
    testCUDA(cudaMemcpy(B_GPU, B, NB * sizeof(int), cudaMemcpyHostToDevice));

    testCUDA(cudaEventCreate(&start));
    testCUDA(cudaEventCreate(&stop));
    testCUDA(cudaEventRecord(start, 0));

    for(int i = 0; i < niter; i++)
        mergeAB<<<N_Blocks, NTPB>>>(A_GPU, B_GPU, M_GPU, NA, NB, d);

    testCUDA(cudaEventRecord(stop, 0));
    testCUDA(cudaEventSynchronize(stop));
    testCUDA(cudaEventElapsedTime(&TimerV, start, stop));


    testCUDA(cudaMemcpy(M, M_GPU, (NA + NB) * sizeof(int), cudaMemcpyDeviceToHost));


    if (IsSorted(M, N))
        printf("M is correctly sorted!\n");
    else
        printf("M is not correctly sorted!\n");

    

    printf("Time taken to merge arrays : %f s\n", (TimerV / 1000) / niter);

    testCUDA(cudaFree(A_GPU));
    testCUDA(cudaFree(B_GPU));
    testCUDA(cudaFree(M_GPU));
    free(A);
    free(B);
    free(M);

    return 0;
}