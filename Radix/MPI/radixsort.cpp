/**
 * Parallel implementation of radix sort. The list to be sorted is split
 * across multiple MPI processes and each sub-list is sorted during each
 * pass as in straight radix sort. 
 */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

// global constants definitions
#define b 32           // number of bits for integer
#define g 8            // group of bits for each scan
#define N b / g        // number of passes
#define B (1 << g)     // number of buckets, 2^g

// MPI tags constants, offset by max bucket to avoid collisions
#define COUNTS_TAG_NUM  B + 1 
#define PRINT_TAG_NUM  COUNTS_TAG_NUM + 1 
#define NUM_TAG PRINT_TAG_NUM + 1

// structure encapsulating buckets with arrays of elements
typedef struct list List;
struct list {
  int* array;
  size_t length;
  size_t capacity;
};

// add item to a dynamic array encapsulated in a structure
int add_item(List* list, int item) {
  if (list->length >= list->capacity) {
    size_t new_capacity = list->capacity*2;
    int* temp = (int*)realloc(list->array, new_capacity*sizeof(int));
    if (!temp) {
      printf("ERROR: Could not realloc for size %d!\n", (int) new_capacity); 
      return 0;
    }
    list->array = temp;
    list->capacity = new_capacity;
  }

  list->array[list->length++] = item;

  return 1;
}

// Compute j bits which appear k bits from the right in x
// Ex. to obtain rightmost bit of x call bits(x, 0, 1)
unsigned bits(unsigned x, int k, int j) {
  return (x >> k) & ~(~0 << j);
}

// Radix sort elements while communicating between other MPI processes
// a - array of elements to be sorted
// buckets - array of buckets, each bucket pointing to array of elements
// P - total number of MPI processes
// rank - rank of this MPI process
// n - number of elements to be sorted
int* radix_sort(int *a, List* buckets, const int P, const int rank, int * n) {
  int count[B][P];   // array of counts per bucket for all processes
  int l_count[B];    // array of local process counts per bucket
  int l_B = B / P;   // number of local buckets per process
  int p_sum[l_B][P]; // array of prefix sums

  // MPI request and status
  MPI_Request req;
  MPI_Status stat;

  for (int pass = 0; pass < N; pass++) {          // each pass

    // init counts arrays
    for (int j = 0; j < B; j++) {
      count[j][rank] = 0;
      l_count[j] = 0;
      buckets[j].length = 0;
    } 

    // count items per bucket
    for (int i = 0; i < *n; i++) {
      unsigned int idx = bits(a[i], pass*g, g);
      count[idx][rank]++; 
      l_count[idx]++;
      if (!add_item(&buckets[idx], a[i])) {
        return NULL;
      }
    }

    // do one-to-all transpose
    for (int p = 0; p < P; p++) {
      if (p != rank) {
        // send counts of this process to others
        MPI_Isend(
            l_count,
            B,
            MPI_INT,
            p,
            COUNTS_TAG_NUM,
            MPI_COMM_WORLD,
            &req);
      }
    }

    // receive counts from others
    for (int p = 0; p < P; p++) {
      if (p != rank) {
        MPI_Recv(
            l_count,
            B,
            MPI_INT,
            p,
            COUNTS_TAG_NUM,
            MPI_COMM_WORLD,
            &stat);

        // populate counts per bucket for other processes
        for (int i = 0; i < B; i++) {
          count[i][p] = l_count[i];
        }
      }
    }

    // calculate new size based on values received from all processes
    int new_size = 0;
    for (int j = 0; j < l_B; j++) {
      int idx = j + rank * l_B;
      for (int p = 0; p < P; p++) {
        p_sum[j][p] = new_size;
        new_size += count[idx][p];
      }
    }

    // reallocate array if newly calculated size is larger
    if (new_size > *n) {
      int* temp = (int*)realloc(a, new_size*sizeof(int));
      if (!a) {
        if (rank == 0) {
          printf("ERROR: Could not realloc for size %d!\n", new_size); 
        }
        return NULL;
      }
      // reassign pointer back to original
      a = temp;
    }

    // send keys of this process to others
    for (int j = 0; j < B; j++) {
      int p = j / l_B;   // determine which process this buckets belongs to
      int p_j = j % l_B; // transpose to that process local bucket index
      if (p != rank && buckets[j].length > 0) {
        MPI_Isend(
            buckets[j].array,
            buckets[j].length,
            MPI_INT,
            p,
            p_j,
            MPI_COMM_WORLD,
            &req);
      }
    }

    // receive keys from other processes
    for (int j = 0; j < l_B; j++) {
      // transpose from local to global index 
      int idx = j + rank * l_B; 
      for (int p = 0; p < P; p++) {

        // get bucket count
        int b_count = count[idx][p]; 
        if (b_count > 0) {

          // point to an index in array where to insert received keys
          int *dest = &a[p_sum[j][p]]; 
          if (rank != p) {
            MPI_Recv(
                dest,
                b_count,
                MPI_INT,
                p,
                j,
                MPI_COMM_WORLD,
                &stat);  

          } else {
            // is same process, copy from buckets to our array
            memcpy(dest, &buckets[idx].array[0], b_count*sizeof(int));
          }
        }
      }
    }

    // update new size
    *n = new_size;
  }

  return a;
}

void usage(char* message) {
  fprintf(stderr, "Incorrect usage! %s\n", message);
}

int main(int argc, char** argv) {
  CALI_CXX_MARK_FUNCTION;

  const char* data_init = "data_init";

  cali::ConfigManager mgr;
  mgr.start();

  int rank, size;

  // initialize MPI environment and obtain basic info
  MPI_Init(&argc, &argv);

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // initialize vars and allocate memory
  int n_total = atoi(argv[1]);
  int n = n_total/size;
  if (n < 1) {
    if (rank == 0) {
      usage("Number of elements must be >= number of processes!");
    }
    MPI_Finalize();
    return EXIT_FAILURE;
  }

  int remainder = B % size;   // in case number of buckets is not divisible
  if (remainder > 0) {
    if (rank == 0) {
      usage("Number of buckets must be divisible by number of processes\n");
    } 
    MPI_Finalize();
    return EXIT_FAILURE;
  }

  // allocate memory and initialize buckets
  // if n is not divisible by size, make the last process handle the reamainder
  if (rank == size-1) {
    int remainder = n_total % size;
    if (remainder > 0) {
      n += remainder;
    }
  }

  const int s = n * rank;
  int* a = (int*)malloc(sizeof(int) * n);

  int b_capacity = n / B;
  if (b_capacity < B) {
    b_capacity = B;
  }
  List* buckets = (List*)malloc(B*sizeof(List));
  for (int j = 0; j < B; j++) {
    buckets[j].array = (int*)malloc(b_capacity*sizeof(int));
    buckets[j].capacity = B;
  }

  // initialize local array
  CALI_MARK_BEGIN(data_init);
  for (int i = 0; i < n; ++i) {
    a[i] = rand() % 1000;
  }
  CALI_MARK_END(data_init);

  // let all processes get here
  MPI_Barrier(MPI_COMM_WORLD);

  // then run the sorting algorithm
  a = radix_sort(&a[0], buckets, size, rank, &n);

  if (a == NULL) {
    printf("ERROR: Sort failed, exiting ...\n");
    MPI_Finalize();
    return EXIT_FAILURE;
  }

  // wait for all processes to finish before printing results 
  MPI_Barrier(MPI_COMM_WORLD);

  adiak::init(NULL);
  adiak::launchdate();    // launch date of the job
  adiak::libraries();     // Libraries used
  adiak::cmdline();       // Command line used to launch the job
  adiak::clustername();   // Name of the cluster
  adiak::value("Algorithm", "RadixSort"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
  adiak::value("ProgrammingModel", "MPI"); // e.g., "MPI", "CUDA", "MPIwithCUDA"
  adiak::value("Datatype", "int"); // The datatype of input elements (e.g., double, int, float)
  adiak::value("SizeOfDatatype", sizeof(int)); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
  adiak::value("InputSize", n_total); // The number of elements in input dataset (1000)
  adiak::value("InputType", "Random"); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
  adiak::value("num_procs", size);
  adiak::value("implementation_source", "Online (https://github.com/ym720/p_radix_sort_mpi/blob/master)"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").

  
  // release MPI resources
  mgr.stop();
  mgr.flush();
  MPI_Finalize();

  // release memory allocated resources
  for (int j = 0; j < B; j++) {
    free(buckets[j].array);
  }
  free(buckets);
  free(a);

  return 0;
}