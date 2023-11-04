#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>


// source: https://www.geeksforgeeks.org/merge-sort/
// source: https://mpitutorial.com/tutorials/mpi-scatter-gather-and-allgather/

void merge(int arr[], int l, int m, int r){
    int first =  m-l+1;
    int second = r-m;
    int array1[first], array2[second];
    for (int i = 0; i < first; i++){
        array1[i] = arr[l+i];
    }
    for(int j = 0; j < second; j++){
        array2[j] = arr[m+1+j];
    }
    int left = 0;
    int right = 0;
    int k = l;
    while(left < first && right < second){
        if(array1[left] <= array2[right]){
            arr[k] = array1[left];
            left++;
        }
        else{
            arr[k] = array2[right];
            right++;
        }
        k++;
    }

    while(left < first){
        arr[k] = array1[left];
        left++;
        k++;
    }
    while(right < second){
        arr[k] = array2[right];
        right++;
        k++;
    }
}

void mergesort(int array[], int l, int r){
    if(l<r){
        int m = l+(r-l)/2;
        mergesort(array, l, m);
        mergesort(array, m+1, r);

        merge(array,l,m,r);
    }
}

void parallel_mergesort(int arr[], int size, int taskid, int numtasks) {
    int local_size = size / numtasks;
    int local_arr[local_size];
    MPI_Scatter(arr, local_size, MPI_INT, local_arr, local_size, MPI_INT, 0, MPI_COMM_WORLD);
    mergesort(local_arr, 0, local_size - 1);
    MPI_Gather(local_arr, local_size, MPI_INT, arr, local_size, MPI_INT, 0, MPI_COMM_WORLD);
    if (taskid == 0) {
        mergesort(arr, 0, size - 1);
    }
}


int main(int argc, char** argv){
    int rc = 0;
    MPI_INIT(&argc, &argv);
    int taskid, numtasks;
    MPI_COMM_rank(MPI_COMM_WORLD, &taskid);
    MPI_COMM_size(MPI_COMM_WORLD, &numtasks);
    if(numtasks<2){
        printf("Need at least two MPI tasks. Quitting...\n");
        MPI_Abort(MPI_COMM_WORLD, rc);
        exit(1);
    }
    
    if(argc != 2){
        printf("Usage: %s <array size>\n", argv[0]);
        MPI_Abort(MPI_COMM_WORLD, rc);
        exit(1);
    }

    int array_size = atoi(argv[1]);
    int arr[array_size];
    if(taskid == 0){
        for(int i = 0; i < array_size; i++){
            arr[i] = rand()%100;
        }
    }
    

    parallel_mergesort(arr, 0, arr.size()-1);


    MPI_Finalize();


    return 0;
}