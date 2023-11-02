#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>


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


int main(int argc, char** argv){
    // MPI_INIT(&argc, &argv);
    // int taskid, numtasks;
    int arr[6] = {12, 11, 13, 5, 6, 7};
    // MPI_COMM_rank(MPI_COMM_WORLD, &taskid);
    // MPI_COMM_size(MPI_COMM_WORLD, &numtasks);
    // if(numtasks<2){
    //     printf("Need at least two MPI tasks. Quitting...\n");
    //     MPI_Abort(MPI_COMM_WORLD, rc);
    //     exit(1);
    // }
    
    // if(rank ==0){
    //     int arr[6] = {12, 11, 13, 5, 6, 7};
    // }
    

    mergesort(arr, 0, arr.size()-1);





    return 0;
}