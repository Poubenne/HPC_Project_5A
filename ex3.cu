#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#define N 1024
#define d 1024

int N_Blocks = N/d+10;
int NTPB = 1024;

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

void GenerateUnsortedRandomArray( int** arr, const size_t length ) {
    *arr = (int*) malloc(length*sizeof(int));

    for (size_t i=0 ; i<length ; i++) {
        (*arr)[i] = rand()%(5*length);
    }
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

__global__ void SortSmall_k(int **M, const size_t *NM, int j, int k){
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


void SortSmall(int **M, const size_t* NM)
{
//Sorts a group of small arrays using GPU parallelized Bitonic Sort

    printf("Preparing Sort...\n");

    int **M_GPU;
    size_t *NM_GPU;
    //size_t size = NUM_VALS * sizeof(float);

    testCUDA(cudaMalloc(&NM_GPU, N * sizeof(size_t)));

    int** tempo_array;
    tempo_array = (int**) malloc(N*sizeof(int*));
    
    
    printf("Creating M_GPU...\n");

    for (int i = 0; i < N; i++){
        testCUDA(cudaMalloc(&tempo_array[i], NM[i] * sizeof(int)));
    }


    testCUDA(cudaMalloc(&M_GPU, N * sizeof(int*)));
    
    for (int i = 0; i < N; i++){
        testCUDA(cudaMemcpy(tempo_array[i], M[i], NM[i] * sizeof(int), cudaMemcpyHostToDevice));
    }
    
    testCUDA(cudaMemcpy(M_GPU, tempo_array, N*sizeof(int*), cudaMemcpyHostToDevice));

    printf("Done Preparing Sort!\n");


    printf("Begin Sorting Procedure...\t");

    int j, k;
    /* Major step */
    for (k = 2; k <= d; k <<= 1) {
        /* Minor step */
        for (j=k>>1; j>0; j=j>>1) {
            //printf("k : %d || j : %d\n", k, j);
            //SortSmall_k<<<N_Blocks, NTPB>>>(M_GPU, NM_GPU, j, k);
            SortSmall_k<<<N, NTPB>>>(M_GPU, NM_GPU, j, k);
            //printf("\n");
        }
    }

    printf("Done Sorting!\n");
    
    printf("Importing M from GPU...\t");

    for (size_t i=0;i<N;i++) {
        testCUDA(cudaMemcpy(M[i], tempo_array[i], d * sizeof(int), cudaMemcpyDeviceToHost));
    }

    printf("M successfully imported from GPU\n");

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


void MergeInto(int** M, size_t* NM){

    int** A = M;
    int** B = M + N / 2;
    size_t* NA = NM;
    size_t* NB = NM + N / 2;
    int size = N / 2;
    int elements_per_array = d;

    while(size != 0){

        int ** merged_list = (int**) malloc(size * sizeof(int*));

        int** tempo_array;
        tempo_array = (int**) malloc(size*sizeof(int*));

        int** merged_list_GPU;
        int** A_GPU;
        int** B_GPU;

        size_t* NA_GPU;
        size_t* NB_GPU;

        testCUDA(cudaMalloc(&NA_GPU, size * sizeof(size_t)));
        testCUDA(cudaMalloc(&NB_GPU, size * sizeof(size_t)));

        printf("Creating Merged Array on GPU...\t");
        testCUDA(cudaMalloc(&merged_list_GPU, size * sizeof(int*)));
        for (size_t i=0;i<size;i++) {
            testCUDA(cudaMalloc(tempo_array+i, elements_per_array * sizeof(int)));
        }
        testCUDA(cudaMemcpy(merged_list_GPU, tempo_array, size * sizeof(int*), cudaMemcpyHostToDevice));
        printf("Done!\r");

        printf("Creating A on GPU...\t");
        testCUDA(cudaMalloc(&A_GPU, size * sizeof(int*)));
        for (size_t i=0;i<size;i++) {
            testCUDA(cudaMalloc(tempo_array+i, elements_per_array * sizeof(int)));
        }
        testCUDA(cudaMemcpy(A_GPU, tempo_array, size * sizeof(int*), cudaMemcpyHostToDevice));
        printf("Done!\r");

        printf("Creating Merged Array on GPU...\t");
        testCUDA(cudaMalloc(&B_GPU, size * sizeof(int*)));
        for (size_t i=0;i<size;i++) {
            testCUDA(cudaMalloc(tempo_array+i, elements_per_array * sizeof(int)));
        }
        testCUDA(cudaMemcpy(B_GPU, tempo_array, size * sizeof(int*), cudaMemcpyHostToDevice));
        printf("Done!\r");

        
        printf("Copying the sizes...\t");
        testCUDA(cudaMemcpy(NA_GPU, NA, size * sizeof(size_t), cudaMemcpyHostToDevice));
        testCUDA(cudaMemcpy(NB_GPU, NB, size * sizeof(size_t), cudaMemcpyHostToDevice));
        printf(" Done!\n");
        
        
        mergeSmallBatch_k<<<size, elements_per_array>>>(A, B, merged_list_GPU, NA_GPU, NB_GPU);
        


        testCUDA(cudaMemcpy(tempo_array, merged_list_GPU, N * sizeof(int*), cudaMemcpyDeviceToHost));

        for (size_t i=0;i<N;i++) {
            testCUDA(cudaMemcpy(M[i], tempo_array[i], d * sizeof(int), cudaMemcpyDeviceToHost));
        }


        testCUDA(cudaFree(A_GPU));
        testCUDA(cudaFree(B_GPU));

        free(merged_list);
        free(tempo_array);

        elements_per_array *= 2;
        size /= 2;
    }


}

void GlobalMerge(int** A, int** B, const size_t* NA, const size_t* NB){
    size_t* NM = (size_t*) malloc(N * sizeof(size_t));
    for (int i = 0; i < N; i++){
        NM[i] = NA[i] + NB[i];
    }

    int** M = (int**) malloc(N * sizeof(int*));

    for (int i = 0; i < N; i++){
        M[i] = (int *) malloc(NM[i] * sizeof(int));
    }

    /*Step 1: Sort all sub-arrays of A and B*/

    SortSmall(A, NA);
    SortSmall(B, NB);

    /*Step 2: Merge all A_i and B_i into each M_i*/

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



    printf("Merging...\t");
    mergeSmallBatch_k(A_GPU, B_GPU, M_GPU, NA_GPU, NB_GPU);
    printf("Done!\n");



    testCUDA(cudaMemcpy(tempo_array, M_GPU, N * sizeof(int*), cudaMemcpyDeviceToHost));

    for (size_t i=0;i<N;i++) {
        testCUDA(cudaMemcpy(M[i], tempo_array[i], d * sizeof(int), cudaMemcpyDeviceToHost));
    }

    printf("M filled\n");


    /*Step 3: Merge all M_i together until only one array is left*/


}


int main(){
    printf("Creating arrays...\t");
    srand(time(NULL));
    int a_sizes[] = {1, 4, 6};

    int** A = (int**) malloc(N*sizeof(int*));
    size_t* NA = (size_t*) malloc(N*sizeof(size_t));
    for (size_t i=0;i<N;i++) {
        NA[i] = a_sizes[i%3];
        GenerateUnsortedRandomArray(A+i, NA[i]);
    }


    int** B = (int**) malloc(N*sizeof(int*));
    size_t* NB = (size_t*) malloc(N*sizeof(size_t));
    for (size_t i=0;i<N;i++) {
        NB[i] = d - NA[i];
        GenerateUnsortedRandomArray(B+i, NB[i]);
    }
    printf(" Done!\n");

    // Merge A and B Here
    GlobalMerge(A, B, NA, NB);
    
    printf("Freeing Memory...\n");
    for (size_t i=0;i<N;i++) {
        free(A[i]);
        free(B[i]);
    }
    free(A);
    free(B);

    return 0;
}