//#include "sorting_functions.cuh"
#include "utility_functions.cuh"


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

int main()
{
    srand(time(NULL));
    /*Creating Arrays*/
    int* A;
    int N = 512;
    int NA = N;
    GenerateRandomArray(&A, NA);
    if ( NA < 50 ) {
        PrintList(A, NA);
    }


    int* B;
    int NB = N;
    GenerateRandomArray(&B, NB);
    if ( NB < 50 ) {
        PrintList(B, NB);
    }

    int* M = (int*) malloc((NA + NB) * sizeof(int));

    int* A_GPU, *B_GPU, *M_GPU;

    float TimerV;
    int niter = 1000000;
    cudaEvent_t start, stop;
    
    /*Creating Timer*/
    
    testCUDA(cudaMalloc(&M_GPU, (NA + NB) * sizeof(int)));
    testCUDA(cudaMalloc(&A_GPU, NA * sizeof(int)));
    testCUDA(cudaMalloc(&B_GPU, NB * sizeof(int)));


    testCUDA(cudaMemcpy(A_GPU, A, NA * sizeof(int), cudaMemcpyHostToDevice));
    testCUDA(cudaMemcpy(B_GPU, B, NB * sizeof(int), cudaMemcpyHostToDevice));

    int N_Blocks = 1;
    int NTPB = 1024;

    testCUDA(cudaEventCreate(&start));
    testCUDA(cudaEventCreate(&stop));
    testCUDA(cudaEventRecord(start, 0));

    /*Benchmarking Kernel*/

    for(int i = 0; i < niter; i++)
        mergeSmall_k<<<N_Blocks, NTPB>>>(A_GPU, B_GPU, M_GPU, NA, NB);

    testCUDA(cudaEventRecord(stop, 0));
    testCUDA(cudaEventSynchronize(stop));
    testCUDA(cudaEventElapsedTime(&TimerV, start, stop));


    testCUDA(cudaMemcpy(M, M_GPU, (NA + NB) * sizeof(int), cudaMemcpyDeviceToHost));

    if (NA + NB < 100)
        PrintList(M, (NA + NB));

    /*Checking that merged array is sorted*/
    printf("M is sorted : %d\n", IsSorted(M, NA + NB));

    printf("Average time taken to merge arrays : %2f s\n", (TimerV / 1000) / niter);
    
    testCUDA(cudaFree(A_GPU));
    testCUDA(cudaFree(B_GPU));
    testCUDA(cudaFree(M_GPU));
    free(M);

    return 0;
}