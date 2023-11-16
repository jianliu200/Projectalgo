// source: https://github.com/racorretjer/Parallel-Merge-Sort-with-MPI/blob/master/merge-mpi.c
// author: racorretjer
// I am using this source code for MPI implementation of merge sort. I have added caliper and adiak annotations to the code.

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include <algorithm> 

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

void merge(int *, int *, int, int, int);
void mergeSort(int *, int *, int, int);

// const char* main_loop = "main loop";
const char* comm =  "comm";
const char* comm_large = "comm_large";
const char* comp = "comp";
const char* comp_large = "comp_large";
const char* MPIbarrier = "MPI_Barrier";
const char* MPIscatter = "MPI_Scatter";
const char* MPIgather = "MPI_Gather"; 	
const char* data_init = "data_init";
const char* correctness_check = "correctness_check";
int main(int argc, char** argv) {
	CALI_CXX_MARK_FUNCTION;
	/********** Create and populate the array **********/
	cali::ConfigManager mgr;
	mgr.start();
    // CALI_MARK_BEGIN(main_loop);
	// int num_threads = atoi(argv[1]);
	int n = atoi(argv[1]);
	printf("The size of the array is: %d", n);
	printf("\n");
	// printf("The number of threads is: %d", num_threads);
	// printf("\n");
	int *original_array = (int*)malloc(n * sizeof(int));
	
	int c;
	srand(time(NULL));
	printf("This is the unsorted array: ");
    CALI_MARK_BEGIN(data_init);
	// For random
	// for(c = 0; c < n; c++) {
		
	// 	original_array[c] = rand() % n;
	// 	//printf("%d ", original_array[c]);
		
	// 	}

	// for sorted arrays
	// for(c = 0; c < n; c++) {
		
	// 	original_array[c] = c;
	// 	//printf("%d ", original_array[c]);
		
	// 	}

	// for reverse sorted arrays
	for(c = n -1 ; c >= 0; c--) {
		
		original_array[c] = c;
		//printf("%d ", original_array[c]);
		
		}
	
    CALI_MARK_END(data_init);
	printf("\n");
	printf("\n");
	
	/********** Initialize MPI **********/
	int world_rank;
	int world_size;
	
	MPI_Init(&argc, &argv);

	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	// if(num_threads > 0){
	// 	world_size = num_threads;
	// }
	printf("The world_size is: %d", world_size);
	printf("\n");
	/********** Divide the array in equal-sized chunks **********/
	int size = n/world_size;
	
	/********** Send each subarray to each process **********/
	int *sub_array = (int*)malloc(size * sizeof(int));
	CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    CALI_MARK_BEGIN(MPIscatter);
	MPI_Scatter(original_array, size, MPI_INT, sub_array, size, MPI_INT, 0, MPI_COMM_WORLD);
	CALI_MARK_END(MPIscatter);
	CALI_MARK_END(comm_large);
	CALI_MARK_END(comm);
	/********** Perform the mergesort on each process **********/
	int *tmp_array = (int*)malloc(size * sizeof(int));
	CALI_MARK_BEGIN(comp);
	CALI_MARK_BEGIN(comp_large);
	mergeSort(sub_array, tmp_array, 0, (size - 1));
	CALI_MARK_END(comp_large);
	CALI_MARK_END(comp);

	CALI_MARK_BEGIN(comm);
	CALI_MARK_BEGIN(comm_large);
	/********** Gather the sorted subarrays into one **********/
	int *sorted = NULL;
	if(world_rank == 0) {
		
		sorted = (int*)malloc(n * sizeof(int));
		
		}
	CALI_MARK_BEGIN(MPIgather);
	MPI_Gather(sub_array, size, MPI_INT, sorted, size, MPI_INT, 0, MPI_COMM_WORLD);
    CALI_MARK_END(MPIgather);
    CALI_MARK_END(comm_large);
	CALI_MARK_END(comm);
	
	/********** Make the final mergeSort call **********/
	if(world_rank == 0) {
		
		int *other_array = (int*)malloc(n * sizeof(int));
		mergeSort(sorted, other_array, 0, (n - 1));
		
		/********** Display the sorted array **********/
		printf("This is the sorted array: ");
		// for(c = 0; c < n; c++) {
			
		// 	printf("%d ", sorted[c]);
			
		// 	}
		
		CALI_MARK_BEGIN(correctness_check);
		bool sorting = std::is_sorted(sorted, sorted + n);
		CALI_MARK_END(correctness_check);
		if(sorting == true) {
			printf("\n");
			printf("\n");
			printf("The array is sorted");
			printf("\n");
			printf("\n");
		}
		else {
			printf("\n");
			printf("\n");
			printf("The array is not sorted");
			printf("\n");
			printf("\n");
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
    CALI_MARK_BEGIN(MPIbarrier);
	MPI_Barrier(MPI_COMM_WORLD);
    CALI_MARK_END(MPIbarrier);
	// CALI_MARK_END(main_loop);
	mgr.stop();
   	mgr.flush();

	MPI_Finalize();
    

    


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
    adiak::value("group_num", "7"); // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "Online: https://github.com/racorretjer/Parallel-Merge-Sort-with-MPI/blob/master/merge-mpi.c"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").

		
	
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