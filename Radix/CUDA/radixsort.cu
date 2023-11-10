#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

#include <cuda_runtime.h>
#include <cuda.h>

#define WSIZE 100
#define LOOPS 1
#define UPPER_BIT 10
#define LOWER_BIT 0

const char* data_init = "data_init";
const char* correctness_check = "correctness_check";
const char* comp = "comp";
const char* comp_large = "comp_large";
const char* comm = "comm";
const char* comm_large = "comm_large";


__device__ unsigned int ddata[WSIZE];

__global__ void parallelRadix() {
 
  // This data in shared memory
	__shared__ volatile unsigned int sdata[WSIZE * 2];

	// Load from global into shared variable
	sdata[threadIdx.x] = ddata[threadIdx.x];

	unsigned int bitmask = 1 << LOWER_BIT;
	unsigned int offset = 0;
	// -1, -2, -4, -8, -16, -32, -64, -128, -256,...
	unsigned int thrmask = 0xFFFFFFFFU << threadIdx.x;
	unsigned int mypos;

	// For each LSB to MSB
	for (int i = LOWER_BIT; i <= UPPER_BIT; i++)
	{
		unsigned int mydata = sdata[((WSIZE - 1) - threadIdx.x) + offset];
		unsigned int mybit = mydata&bitmask;
		// Get population of ones and zeroes
		unsigned int ones = __ballot(mybit);
		unsigned int zeroes = ~ones;
		// Switch ping-pong buffers
		offset ^= WSIZE;

		// Do zeroes, then ones
		if (!mybit)
		{
			mypos = __popc(zeroes&thrmask);
		}
		else  {      // Threads with a one bit
			// Get my position in ping-pong buffer
			mypos = __popc(zeroes) + __popc(ones&thrmask);
		}

		// Move to buffer  (or use shfl for cc 3.0)
		sdata[mypos - 1 + offset] = mydata;
		// Repeat for next bit
		bitmask <<= 1;
	}
	// Put results to global
	ddata[threadIdx.x] = sdata[threadIdx.x + offset];
}

int main() {
  CALI_CXX_MARK_FUNCTION;
  
  cali::ConfigManager mgr;
  mgr.start();

	unsigned int hdata[WSIZE];

	for (int lcount = 0; lcount < LOOPS; lcount++) {
		srand(time(NULL));

		// Fill array with random elements
		// Range = 1024
    CALI_MARK_BEGIN(data_init);
		for (int i = 0; i < WSIZE; i++) {
			hdata[i] = i;
		}
    CALI_MARK_END(data_init);

    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
		// Copy data from host to device
		cudaMemcpyToSymbol(ddata, hdata, WSIZE * sizeof(unsigned int));
    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);

    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_large);
		parallelRadix <<< 1, WSIZE >>>();
    CALI_MARK_END(comp_large);
    
		// Make kernel function synchronous
		cudaDeviceSynchronize();
    CALI_MARK_END(comp);

    CALI_MARK_BEGIN(comm);
		// Copy data from device to host
    CALI_MARK_BEGIN(comm_large);
		cudaMemcpyFromSymbol(hdata, ddata, WSIZE * sizeof(unsigned int));
    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);
	}

  CALI_MARK_BEGIN(correctness_check);
  for (int i = 0; i < WSIZE - 1; ++i) {
    if (hdata[i] > hdata[i + 1]) {
      printf("Incorrectly sorted...\n");
      return 0;
    }
  }
  CALI_MARK_END(correctness_check);

  printf("Final List Sorted Correctly!\n");

  adiak::init(NULL);
  adiak::launchdate();    // launch date of the job
  adiak::libraries();     // Libraries used
  adiak::cmdline();       // Command line used to launch the job
  adiak::clustername();   // Name of the cluster
  adiak::value("Algorithm", "RadixSort"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
  adiak::value("ProgrammingModel", "CUDA"); // e.g., "MPI", "CUDA", "MPIwithCUDA"
  adiak::value("Datatype", "int"); // The datatype of input elements (e.g., double, int, float)
  adiak::value("SizeOfDatatype", sizeof(int)); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
  adiak::value("InputSize", "100"); // The number of elements in input dataset (1000)
  adiak::value("InputType", "Random"); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
  adiak::value("num_threads", "4"); // The number of CUDA or OpenMP threads  
  adiak::value("num_blocks", "25"); // The number of CUDA blocks 
  adiak::value("group_num", "7"); // The number of your group (integer, e.g., 1, 10)
  adiak::value("implementation_source", "Online (https://github.com/ym720/p_radix_sort_mpi/blob/master)"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").


  return 0;
}