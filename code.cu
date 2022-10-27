#include <stdlib.h>
#include <stdio.h>

void testCUDA(cudaError_t error, const char *file, int line)  {

	if (error != cudaSuccess) {
	   printf("There is an error in file %s at line %d\n", file, line);
       exit(EXIT_FAILURE);
	} 
}

#define testCUDA(error) (testCUDA(error, __FILE__ , __LINE__))

__global__ void mergeSmall_k(const int* A, const int* B, int* &M, int N1, int N2)
{

    // Test with shared memory

    unsigned long idx = threadIdx.x;

    int K[2];
    int P[2];

    if (idx > N1)
    {
        K[0] = idx - N1;
        K[1] = N1;

        P[0] = N1;
        P[1] = idx - N1;
    }
    else
    {
        K[0] = 0;
        K[1] = idx;

        P[0] = idx;
        P[1] = 0;
    }

    while (true)
    {

        int offset = abs((K[1] - P[1]) / 2);
        int Q[] = {K[0] + offset, K[1] - offset};

        if (Q[1] >= 0 && Q[0] <= N2 && (Q[1] == N1 || Q[0] == 0 || A[Q[1]] > B[Q[0] - 1]))
        {
            if (Q[0] == N2 || Q[1] == 0 || A[Q[1] - 1] <= B[Q[0]])
            {
                if (Q[1] < N1 && (Q[0] == N2 || A[Q[1]] <= B[Q[0]]))
                    M[idx] = A[Q[1]];
                else
                    M[idx] = B[Q[0]];
                break;
            }
            else
            {
                K[0] = Q[0] + 1;
                K[1] = Q[1] - 1;
            }
        }
        else
        {
            P[0] = Q[0] - 1;
            P[1] = Q[1] + 1;
        }
    }
}

void print_list(int* A, int N){
    
    for (int i = 0; i < N; i++){
        printf("%d ", A[i]);
    }
    printf("\n");
}

int main()
{

    int A[] = {1, 5, 7, 10, 15};
    int N1 = 5;

    int B[] = {2, 4, 8, 9, 13};
    int N2 = 5;

    int* M = (int*) malloc((N1 + N2) * sizeof(int));

    int* A_GPU, *B_GPU, *M_GPU;

    testCUDA(cudaMalloc(&M_GPU, (N1 + N2) * sizeof(int)));
    testCUDA(cudaMalloc(&A_GPU, N1 * sizeof(int)));
    testCUDA(cudaMalloc(&B_GPU, N2 * sizeof(int)));


    testCUDA(cudaMemcpy(A_GPU, A, N1 * sizeof(int), cudaMemcpyHostToDevice));
    testCUDA(cudaMemcpy(B_GPU, B, N2 * sizeof(int), cudaMemcpyHostToDevice));

    int NB = 1;
    int NTPB = 1024;

    mergeSmall_k<<<NB, NTPB>>>(A_GPU, B_GPU, M_GPU, N1, N2);

    testCUDA(cudaMemcpy(M, M_GPU, (N1 + N2) * sizeof(int), cudaMemcpyDeviceToHost));

    print_list(M, (N1 + N2));

    testCUDA(cudaFree(A_GPU));
    testCUDA(cudaFree(B_GPU));
    testCUDA(cudaFree(M_GPU));
    free(M);

    return 0;
}