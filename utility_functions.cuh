#include <stdlib.h>
#include <stdio.h>
#include <time.h>

/*
    Header files shouldn't contain code, but this is a small project, so whatever
*/

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
    * @param The size of the array
    */
void GenerateRandomArray( int** arr, const size_t size ) {
    *arr = (int*) malloc(size*sizeof(int));

    for (size_t i=0 ; i<size ; i++) {
        (*arr)[i] = rand()%(5*size);
    }
    QuickSort(*arr, size);
}

void GenerateUnsortedRandomArray( int** arr, const size_t length ) {
    *arr = (int*) malloc(length*sizeof(int));

    for (size_t i=0 ; i<length ; i++) {
        (*arr)[i] = rand()%(5*length);
    }
}

void ReRandomizeArray( int* arr, const size_t length ) {
    for (size_t i=0 ; i<length ; i++) {
        arr[i] = rand()%(5*length);
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