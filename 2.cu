#include "utility_functions.cuh"
//#include "sorting_functions.cuh"

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
    if ( gbx >= n_arrays ) {// excedent block
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

int main(){

    int N = 512;
    int d = 1024;

    int N_Blocks = N/d+10;
    int NTPB = 1024;


    /*Creating arrays*/
    printf("Creating arrays...\t");
    srand(time(NULL));

    int** A = (int**) malloc(N*sizeof(int*));
    for (size_t i=0;i<N;i++) {
        GenerateRandomArray(A+i, d/2);
    }


    int** B = (int**) malloc(N*sizeof(int*));
    for (size_t i=0;i<N;i++) {
        GenerateRandomArray(B+i, d/2);
    }

/*
    int** S = (int**) malloc(N*sizeof(int*));
    size_t* NS = (size_t*) malloc(N*sizeof(size_t));
    for (size_t i=0;i<N;i++) {
        //NS[i] = d;
        GenerateUnsortedRandomArray(S+i, d);
    }
    printf(" Done!\n");
*/



//  We could create contiguous arrays...
    printf("Creating GPU arrays...\t");
    int** A_GPU;
    int** B_GPU;
    int** M_GPU;


    int** tempo_array;
    tempo_array = (int**) malloc(N*sizeof(int*));

//  Creating M_GPU
    printf("Creating M_GPU...\t");
    testCUDA(cudaMalloc(&M_GPU, N * sizeof(int*)));
    for (size_t i=0;i<N;i++) {
        testCUDA(cudaMalloc(tempo_array+i, d * sizeof(int)));
    }
    testCUDA(cudaMemcpy(M_GPU, tempo_array, N * sizeof(int*), cudaMemcpyHostToDevice));
    printf("Done!\n");

//  Creating A_GPU
    printf("Creating A_GPU...\t");
    testCUDA(cudaMalloc(&A_GPU, N * sizeof(int*)));
    for (size_t i=0;i<N;i++) {
        testCUDA(cudaMalloc(tempo_array+i, d/2 * sizeof(int)));
        testCUDA(cudaMemcpy(tempo_array[i], A[i], d/2 * sizeof(int), cudaMemcpyHostToDevice));
    }
    testCUDA(cudaMemcpy(A_GPU, tempo_array, N * sizeof(int*), cudaMemcpyHostToDevice));
    printf("Done!\n");

//  Creating B_GPU
    printf("Creating B_GPU...\t");
    testCUDA(cudaMalloc(&B_GPU, N * sizeof(int*)));
    for (size_t i=0;i<N;i++) {
        testCUDA(cudaMalloc(tempo_array+i, d/2 * sizeof(int)));
        testCUDA(cudaMemcpy(tempo_array[i], B[i], d/2 * sizeof(int), cudaMemcpyHostToDevice));
    }
    testCUDA(cudaMemcpy(B_GPU, tempo_array, N * sizeof(int*), cudaMemcpyHostToDevice));
    printf("Done!\n");

    /*Creating Timer*/

    float TimerV;
    int niter = 1000000;
    cudaEvent_t start, stop;
    

    /*Merging Arrays*/
    printf("Merging...\t");

    testCUDA(cudaEventCreate(&start));
    testCUDA(cudaEventCreate(&stop));
    testCUDA(cudaEventRecord(start, 0));
    
    for(int i = 0; i < niter; i++)
        mergeSmallBatch_k<<<N_Blocks, NTPB>>>((const int**)A_GPU, (const int**)B_GPU, M_GPU, N, d/2);
    
    testCUDA(cudaEventRecord(stop, 0));
    testCUDA(cudaEventSynchronize(stop));
    testCUDA(cudaEventElapsedTime(&TimerV, start, stop));

    printf("Done!\n");

    printf("Average time taken to merge arrays : %2f s\n", (TimerV / 1000) / niter);


    /*Gathering and verifying results on CPU*/


    int** M = (int**) malloc(N * sizeof(int*));
    testCUDA(cudaMemcpy(tempo_array, M_GPU, N * sizeof(int*), cudaMemcpyDeviceToHost));
    printf("M created\n");
    for (size_t i=0;i<N;i++) {
        M[i] = (int*) malloc(d * sizeof(int));
        testCUDA(cudaMemcpy(M[i], tempo_array[i], d * sizeof(int), cudaMemcpyDeviceToHost));
    }
    printf("M filled\n");

    printf("Verifying Merge result\n");
    for (size_t i=0;i<N;i++) {  
        if (! IsSortedAscending(M[i],d) ) {
            printf("The result isn't correct...\n");
            exit(EXIT_FAILURE);
        }
    }
    printf("The result is correct!\n");

    return 0;
}