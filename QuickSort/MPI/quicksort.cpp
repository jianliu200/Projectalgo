// C program to implement the Quick Sort
// Algorithm using MPI
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>
using namespace std;

const char* comm = "comm";
const char* comm_large = "comm_large";
const char* comp = "comp";
const char* comp_large = "comp_large";
const char* barrier = "barrier";
const char* gather = "gather";
const char* scatter = "scatter";
const char* Bcast = "Bcast";
const char* data_init = "data_init";
const char* checkCorrect = "checkCorrect";

bool checkSorted(int* arr, int size) {
    CALI_MARK_BEGIN(checkCorrect);
    int prev = arr[0];
    for(int i = 0; i < size; i++) {
        if (arr[i] < prev){
            return false;
        }
        prev = arr[i];
    }
    CALI_MARK_END(checkCorrect);
    return true;
}

// Function to swap two numbers
void swap(int* arr, int i, int j)
{
	int t = arr[i];
	arr[i] = arr[j];
	arr[j] = t;
}

// Function that performs the Quick Sort
// for an array arr[] starting from the
// index start and ending at index end
void quicksort(int* arr, int start, int end)
{
	int pivot, index;

	// Base Case
	if (end <= 1)
		return;

	// Pick pivot and swap with first
	// element Pivot is middle element
	pivot = arr[start + end / 2];
	swap(arr, start, start + end / 2);

	// Partitioning Steps
	index = start;

	// Iterate over the range [start, end]
	for (int i = start + 1; i < start + end; i++) {

		// Swap if the element is less
		// than the pivot element
		if (arr[i] < pivot) {
			index++;
			swap(arr, i, index);
		}
	}

	// Swap the pivot into place
	swap(arr, start, index);

	// Recursive Call for sorting
	// of quick sort function
	quicksort(arr, start, index - start);
	quicksort(arr, index + 1, start + end - index - 1);
}

// Function that merges the two arrays
int* merge(int* arr1, int n1, int* arr2, int n2)
{
	int* result = (int*)malloc((n1 + n2) * sizeof(int));
	int i = 0;
	int j = 0;
	int k;

	for (k = 0; k < n1 + n2; k++) {
		if (i >= n1) {
			result[k] = arr2[j];
			j++;
		}
		else if (j >= n2) {
			result[k] = arr1[i];
			i++;
		}

		// Indices in bounds as i < n1
		// && j < n2
		else if (arr1[i] < arr2[j]) {
			result[k] = arr1[i];
			i++;
		}

		// v2[j] <= v1[i]
		else {
			result[k] = arr2[j];
			j++;
		}
	}
	return result;
}

// Driver Code
int main(int argc, char* argv[])
{
    CALI_CXX_MARK_FUNCTION;
    cali::ConfigManager mgr;
    mgr.start();

    int number_of_elements;
    int* data = NULL;
    int chunk_size, own_chunk_size;
    int* chunk;
    double time_taken;
    MPI_Status status;

    if (argc != 2) {
        printf("Desired number of arguments are not there in argv....\n");
        printf("1 argument required with the desired number of elements...\n");
        exit(-1);
    }

    int number_of_process, rank_of_process;
    int rc = MPI_Init(&argc, &argv);

    if (rc != MPI_SUCCESS) {
        printf("Error in creating MPI program.\n Terminating......\n");
        MPI_Abort(MPI_COMM_WORLD, rc);
    }

    MPI_Comm_size(MPI_COMM_WORLD, &number_of_process);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_of_process);

    if (rank_of_process == 0) {
        // Read the number of elements from the command line
        number_of_elements = atoi(argv[1]);

        // Allocate memory for the data array
        data = (int*)malloc(number_of_elements * sizeof(int));

        // Generate a random array
        // srand(time(NULL));
        // for (int i = 0; i < number_of_elements; i++) {
        //     data[i] = rand() % 100; // Adjust the range as needed
        // }

        // Generate a reverse-sorted array
        // for (int i = 0; i < number_of_elements; i++) {
        //     data[i] = number_of_elements - i; // Fill in reverse order
        // }

        // Generate a sorted array
        for (int i = 0; i < number_of_elements; i++) {
            data[i] = i + 1; // Fill in ascending order
        }

        // Print the generated array
        CALI_MARK_BEGIN(data_init);
        printf("Elements in the array are: \n");
        for (int i = 0; i < number_of_elements; i++) {
            printf("%d ", data[i]);
        }
        CALI_MARK_END(data_init);
        printf("\n");
    }

    CALI_MARK_BEGIN(comm);
    // Blocks all processes until reaching this point
    CALI_MARK_BEGIN(barrier);
    MPI_Barrier(MPI_COMM_WORLD);
    CALI_MARK_END(barrier);

    // Starts Timer
    time_taken -= MPI_Wtime();

    // Broadcast the size to all processes from the root process
    CALI_MARK_BEGIN(Bcast);
    MPI_Bcast(&number_of_elements, 1, MPI_INT, 0, MPI_COMM_WORLD);
    CALI_MARK_END(Bcast);

    // Compute chunk size
    chunk_size = (number_of_elements % number_of_process == 0)
                     ? (number_of_elements / number_of_process)
                     : number_of_elements / (number_of_process - 1);

    // Allocate memory for the local chunk
    chunk = (int*)malloc(chunk_size * sizeof(int));

    // Scatter the chunk size data to all processes
    CALI_MARK_BEGIN(scatter);
    MPI_Scatter(data, chunk_size, MPI_INT, chunk, chunk_size, MPI_INT, 0, MPI_COMM_WORLD);
    CALI_MARK_END(scatter);

    // Free memory for the root process as it's no longer needed
    if (rank_of_process == 0) {
        free(data);
    }

    // Compute size of own chunk and then sort them using quicksort
    own_chunk_size = (number_of_elements >= chunk_size * (rank_of_process + 1))
                         ? chunk_size
                         : (number_of_elements - chunk_size * rank_of_process);

    CALI_MARK_END(comm);
    // Sorting array with quicksort for every chunk as called by process
    CALI_MARK_BEGIN(comp);
    quicksort(chunk, 0, own_chunk_size);
    CALI_MARK_END(comp);

    CALI_MARK_BEGIN(comm);
    for (int step = 1; step < number_of_process;
         step = 2 * step) {
        if (rank_of_process % (2 * step) != 0) {
            MPI_Send(chunk, own_chunk_size, MPI_INT,
                     rank_of_process - step, 0,
                     MPI_COMM_WORLD);
            break;
        }

        if (rank_of_process + step < number_of_process) {
            int received_chunk_size
                = (number_of_elements
                   >= chunk_size
                      * (rank_of_process + 2 * step))
                      ? (chunk_size * step)
                      : (number_of_elements
                         - chunk_size
                         * (rank_of_process + step));
            int* chunk_received;
            chunk_received = (int*)malloc(
                received_chunk_size * sizeof(int));
            MPI_Recv(chunk_received, received_chunk_size,
                     MPI_INT, rank_of_process + step, 0,
                     MPI_COMM_WORLD, &status);

            data = merge(chunk, own_chunk_size,
                         chunk_received,
                         received_chunk_size);

            free(chunk);
            free(chunk_received);
            chunk = data;
            own_chunk_size
                = own_chunk_size + received_chunk_size;
        }
    }
    

    // Stop the timer
    time_taken += MPI_Wtime();

    // Print the sorted array
    printf("\n\n\n\nSorted array: \n");
    for (int i = 0; i < own_chunk_size; i++) {
        printf("%d ", chunk[i]);
    }
    printf("\n");

    // Check if the array is sorted
    cout << "Is the array sorted? " << checkSorted(chunk, own_chunk_size) << endl;
    CALI_MARK_END(comm);

    // Additional information for profiling
    adiak::init(NULL);
    adiak::launchdate();    // launch date of the job
    adiak::libraries();     // Libraries used
    adiak::cmdline();       // Command line used to launch the job
    adiak::clustername();   // Name of the cluster
    adiak::value("Algorithm", "Quick Sort"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", "MPI"); // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", "int"); // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", sizeof(int)); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", number_of_elements); // The number of elements in input dataset
    adiak::value("InputType", "Random"); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    adiak::value("num_procs", number_of_process); // The number of processors (MPI ranks)
    adiak::value("check correct", checkCorrect); // check the correctness
    // adiak::value("num_threads", num_threads); // The number of CUDA or OpenMP threads
    // adiak::value("num_blocks", num_blocks); // The number of CUDA blocks 
    adiak::value("group_num", "7"); // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "Online: https://www.geeksforgeeks.org/implementation-of-quick-sort-using-mpi-omp-and-posix-thread/"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").

    mgr.stop();
    mgr.flush();
    MPI_Finalize();
    return 0;
}