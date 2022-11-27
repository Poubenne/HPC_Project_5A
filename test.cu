#include <stdlib.h>
#include <stdio.h>
#include <time.h>

__global__ void TestFunc(int** arrays, int* sizes, int n_sizes) {
    unsigned long idx = threadIdx.x;
    unsigned long bx = blockIdx.x;

    if ( bx  >= n_sizes || idx >= sizes[bx] ) {
        return ;
    }

    arrays[bx][idx] ++;
}

int main(void) {
    int** M;
    int* M_s;
    int** M_GPU;
    int* M_s_GPU;

    M = (int**)malloc(3*sizeof(int*));
    M_s = (int*) malloc(3*sizeof(int));
    M_s[0] = 2;
    M[0] = (int*) malloc(M_s[0]*sizeof(int));
    M[0][0] = 1;
    M[0][1] = 1;

    M_s[1] = 3;
    M[1] = (int*) malloc(M_s[1]*sizeof(int));
    M[1][0] = 1;
    M[1][1] = 1;
    M[1][2] = 1;

    M_s[2] = 2;
    M[2] = (int*) malloc(M_s[2]*sizeof(int));
    M[2][0] = 1;
    M[2][1] = 1;

    printf("sizes\n");
    cudaMalloc(&M_s_GPU, 3*sizeof(int));
    printf("gpu array created\n");
    cudaMemcpy(M_s_GPU, M_s, 3*sizeof(int), cudaMemcpyHostToDevice);
    printf("gpu array filled\n");



    printf("Actual array of arrays\n");
    int** tempo = (int**)malloc(3*sizeof(int*));
    for (size_t i=0;i<3;i++) {
        cudaMalloc(tempo+i, M_s[i]*sizeof(int));
        cudaMemcpy(tempo[i], M[i], M_s[i]*sizeof(int), cudaMemcpyHostToDevice);
    }
    printf("tempo done\n");
    cudaMalloc(&M_GPU, 3*sizeof(int*));
    cudaMemcpy(M_GPU, tempo, 3*sizeof(int*), cudaMemcpyHostToDevice);
    printf("done\n");


    printf("function\n");
    TestFunc<<<3, 3>>>(M_GPU, M_s_GPU, 3);
    printf("\tdone\n");

    printf("Preparing result\n");
    cudaMemcpy(tempo, M_GPU, 3*sizeof(int*), cudaMemcpyDeviceToHost);
    printf("Adresses copied\n");
    for (size_t i=0;i<3;i++) {
        cudaMemcpy( M[i], tempo[i], M_s[i]*sizeof(int), cudaMemcpyDeviceToHost);
    }
    printf("Result copied\n");


    for (size_t i=0;i<3;i++) {
        for (size_t j=0;j<M_s[i];j++) {
            printf("M[%lu][%lu] = %d\n",i,j,M[i][j]);
        }
    }
}