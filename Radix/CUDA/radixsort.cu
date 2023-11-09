#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

#include <cuda_runtime.h>
#include <cuda.h>

#define WSIZE 32
#define LOOPS 1
#define UPPER_BIT 10
#define LOWER_BIT 0

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

	unsigned int hdata[WSIZE];
	float totalTime = 0;

	for (int lcount = 0; lcount < LOOPS; lcount++) {
		srand(time(NULL));
		// Array elements have value in range of 1024
		unsigned int range = 1U << UPPER_BIT;

		// Fill array with random elements
		// Range = 1024
		for (int i = 0; i < WSIZE; i++) {
			hdata[i] = i;
		}

		// Copy data from host to device
		cudaMemcpyToSymbol(ddata, hdata, WSIZE * sizeof(unsigned int));

		parallelRadix <<< 1, WSIZE >>>();
    
		// Make kernel function synchronous
		cudaDeviceSynchronize();

		// Copy data from device to host
		cudaMemcpyFromSymbol(hdata, ddata, WSIZE * sizeof(unsigned int));
	}

	printf("Parallel Radix Sort:\n");
	printf("Array size = %d\n", WSIZE * LOOPS);
	printf("Time elapsed = %fseconds\n", totalTime);

  return 0;
}