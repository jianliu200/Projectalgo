// #include "mpi.h"
// #include <stdio.h>
// #include <stdlib.h>
// #include <limits.h>

// #include <caliper/cali.h>
// #include <caliper/cali-manager.h>
// #include <adiak.hpp>


// // source: https://www.geeksforgeeks.org/merge-sort/
// // source: https://mpitutorial.com/tutorials/mpi-scatter-gather-and-allgather/

// void merge(int arr[], int l, int m, int r){
//     int first =  m-l+1;
//     int second = r-m;
//     int array1[first], array2[second];
//     for (int i = 0; i < first; i++){
//         array1[i] = arr[l+i];
//     }
//     for(int j = 0; j < second; j++){
//         array2[j] = arr[m+1+j];
//     }
//     int left = 0;
//     int right = 0;
//     int k = l;
//     while(left < first && right < second){
//         if(array1[left] <= array2[right]){
//             arr[k] = array1[left];
//             left++;
//         }
//         else{
//             arr[k] = array2[right];
//             right++;
//         }
//         k++;
//     }

//     while(left < first){
//         arr[k] = array1[left];
//         left++;
//         k++;
//     }
//     while(right < second){
//         arr[k] = array2[right];
//         right++;
//         k++;
//     }
// }

// void mergesort(int array[], int l, int r){
//     if(l<r){
//         int m = l+(r-l)/2;
//         mergesort(array, l, m);
//         mergesort(array, m+1, r);

//         merge(array,l,m,r);
//     }
// }

// void parallel_mergesort(int arr[], int size, int taskid, int numtasks) {
//     int local_size = size / numtasks;
//     int local_arr[local_size];
//     MPI_Scatter(arr, local_size, MPI_INT, local_arr, local_size, MPI_INT, 0, MPI_COMM_WORLD);
//     mergesort(local_arr, 0, local_size - 1);
//     MPI_Gather(local_arr, local_size, MPI_INT, arr, local_size, MPI_INT, 0, MPI_COMM_WORLD);
//     if (taskid == 0) {
//         mergesort(arr, 0, size - 1);
//     }
// }


// int main(int argc, char** argv){
//     int rc = 0;
//     MPI_INIT(&argc, &argv);
//     int taskid, numtasks;
//     MPI_COMM_rank(MPI_COMM_WORLD, &taskid);
//     MPI_COMM_size(MPI_COMM_WORLD, &numtasks);
//     if(numtasks<2){
//         printf("Need at least two MPI tasks. Quitting...\n");
//         MPI_Abort(MPI_COMM_WORLD, rc);
//         exit(1);
//     }
    
//     if(argc != 2){
//         printf("Usage: %s <array size>\n", argv[0]);
//         MPI_Abort(MPI_COMM_WORLD, rc);
//         exit(1);
//     }

//     int array_size = atoi(argv[1]);
//     int arr[array_size];
//     if(taskid == 0){
//         for(int i = 0; i < array_size; i++){
//             arr[i] = rand()%100;
//         }
//     }
    

//     parallel_mergesort(arr, 0, arr.size()-1);


//     MPI_Finalize();


//     return 0;
// }

// source: https://github.com/racorretjer/Parallel-Merge-Sort-with-MPI/blob/master/merge-mpi.c
// author: racorretjer

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

void merge(int *, int *, int, int, int);
void mergeSort(int *, int *, int, int);

const char* main_loop = "main loop";
const char* comm =  "comm";
const char* comm_large = "comm_large";
const char* barrier = "barrier";
const char* scatter = "scatter";
const char* gather = "gather";
const char* data_init = "data_init";

int main(int argc, char** argv) {
	
	/********** Create and populate the array **********/
    CALI_MARK_BEGIN(main_loop);
	int n = atoi(argv[1]);
	int *original_array = malloc(n * sizeof(int));
	
	int c;
	srand(time(NULL));
	printf("This is the unsorted array: ");
    CALI_MARK_BEGIN(data_init);
	for(c = 0; c < n; c++) {
		
		original_array[c] = rand() % n;
		printf("%d ", original_array[c]);
		
		}
    CALI_MARK_END(data_init);
	printf("\n");
	printf("\n");
	
	/********** Initialize MPI **********/
	int world_rank;
	int world_size;
	
	MPI_INIT(&argc, &argv);
    CALI_MARK_BEGIN(comm);
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
		
	/********** Divide the array in equal-sized chunks **********/
	int size = n/world_size;
	
	/********** Send each subarray to each process **********/
	int *sub_array = malloc(size * sizeof(int));
    CALI_MARK_BEGIN(comm_large);
    CALI_MARK_BEGIN(scatter);
	MPI_Scatter(original_array, size, MPI_INT, sub_array, size, MPI_INT, 0, MPI_COMM_WORLD);
	CALI_MARK_END(scatter);
	/********** Perform the mergesort on each process **********/
	int *tmp_array = malloc(size * sizeof(int));
	mergeSort(sub_array, tmp_array, 0, (size - 1));
	
	/********** Gather the sorted subarrays into one **********/
	int *sorted = NULL;
	if(world_rank == 0) {
		
		sorted = malloc(n * sizeof(int));
		
		}
	CALI_MARK_BEGIN(gather);
	MPI_Gather(sub_array, size, MPI_INT, sorted, size, MPI_INT, 0, MPI_COMM_WORLD);
    CALI_MARK_END(gather);
    CALI_MARK_END(comm_large);
	
	/********** Make the final mergeSort call **********/
	if(world_rank == 0) {
		
		int *other_array = malloc(n * sizeof(int));
		mergeSort(sorted, other_array, 0, (n - 1));
		
		/********** Display the sorted array **********/
		printf("This is the sorted array: ");
		for(c = 0; c < n; c++) {
			
			printf("%d ", sorted[c]);
			
			}
			
		printf("\n");
		printf("\n");
			
		/********** Clean up root **********/
		free(sorted);
		free(other_array);
			
		}
	
	/********** Clean up rest **********/
	free(original_array);
	free(sub_array);
	free(tmp_array);
	
	/********** Finalize MPI **********/
    CALI_MARK_BEGIN(barrier);
	MPI_Barrier(MPI_COMM_WORLD);
    CALI_MARK_END(barrier);
	MPI_Finalize();
    CALI_MARK_END(comm);

    


    adiak::init(NULL);
    adiak::launchdate();    // launch date of the job
    adiak::libraries();     // Libraries used
    adiak::cmdline();       // Command line used to launch the job
    adiak::clustername();   // Name of the cluster
    adiak::value("Algorithm", "Merge sort"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", "MPI"); // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", "integer"); // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", "4 bytes"); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", n); // The number of elements in input dataset (1000)
    adiak::value("InputType", "Random"); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    adiak::value("num_procs", world_size); // The number of processors (MPI ranks)
    // adiak::value("num_threads", num_threads); // The number of CUDA or OpenMP threads
    // adiak::value("num_blocks", num_blocks); // The number of CUDA blocks 
    //adiak::value("group_num", group_number); // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "Online") // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").

	CALI_MARK_END(main_loop);
	
}

/********** Merge Function **********/
void merge(int *a, int *b, int l, int m, int r) {
	
	int h, i, j, k;
	h = l;
	i = l;
	j = m + 1;
	
	while((h <= m) && (j <= r)) {
		
		if(a[h] <= a[j]) {
			
			b[i] = a[h];
			h++;
			
			}
			
		else {
			
			b[i] = a[j];
			j++;
			
			}
			
		i++;
		
		}
		
	if(m < h) {
		
		for(k = j; k <= r; k++) {
			
			b[i] = a[k];
			i++;
			
			}
			
		}
		
	else {
		
		for(k = h; k <= m; k++) {
			
			b[i] = a[k];
			i++;
			
			}
			
		}
		
	for(k = l; k <= r; k++) {
		
		a[k] = b[k];
		
		}
		
	}

/********** Recursive Merge Function **********/
void mergeSort(int *a, int *b, int l, int r) {
	
	int m;
	
	if(l < r) {
		
		m = (l + r)/2;
		
		mergeSort(a, b, l, m);
		mergeSort(a, b, (m + 1), r);
		merge(a, b, l, m, r);
		
		}
		
	}