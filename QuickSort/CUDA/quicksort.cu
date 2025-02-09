#include <time.h>
#include <stdio.h>
#include <stdlib.h>
// #include <cutil_inline.h>
// #include <cuda.h>
#include <cuda_runtime.h>
#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>
using namespace std;


#define MAX_THREADS 512 
#define N 67108864

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

 int*	r_values;
 int*	d_values;

 // initialize data set
 void Init(int* values, int i) {
	srand( time(NULL) );
        printf("\n------------------------------\n");

        // randomly generated input array
        for (int x = 0; x < N; ++x) {
                values[x] = rand() % 100;
                // printf("%d ", values[x]);
        }

        // Reverse sorted input array
        // for (int x = 0; x < N; ++x) {
        //         values[x] = N - x;
        //         // printf("%d ", values[x]);
        // }

        // Sorted input array
        // for (int x = 0; x < N; ++x) {
        //         values[x] = x + 1;
        //         // printf("%d ", values[x]);
        // }

        // if (i == 0) {
        // // Uniform distribution
        //         printf("Data set distribution: Uniform\n");
        //         for (int x = 0; x < N; ++x) {
        //                 values[x] = rand() % 100;
        //                 //printf("%d ", values[x]);
        //         }
        // }
        // else if (i == 1) {
        // // Gaussian distribution
        // #define MEAN    100
        // #define STD_DEV 5
        //         printf("Data set distribution: Gaussian\n");
        //         float r;
        //         for (int x = 0; x < N; ++x) {
        //                 r  = (rand()%3 - 1) + (rand()%3 - 1) + (rand()%3 - 1);
        //                 values[x] = int( round(r * STD_DEV + MEAN) );
        //                 //printf("%d ", values[x]);
        //         }
        // }
        // else if (i == 2) {
        // // Bucket distribution
        //         printf("Data set distribution: Bucket\n");
        //         int j = 0;
        //         for (int x = 0; x < N; ++x, ++j) {
        //                 if (j / 20 < 1)
        //                         values[x] = rand() % 20;
        //                 else if (j / 20 < 2)
        //                         values[x] = rand() % 20 + 20;
        //                 else if (j / 20 < 3)
        //                         values[x] = rand() % 20 + 40;
        //                 else if (j / 20 < 4)
        //                         values[x] = rand() % 20 + 60;
        //                 else if (j / 20 < 5)
        //                         values[x] = rand() % 20 + 80;
        //                 if (j == 100)
        //                         j = 0;
        //                 //printf("%d ", values[x]);
        //         }
        // }
        // else if (i == 3) {
        // // Sorted distribution
        //         printf("Data set distribution: Sorted\n");
        //         /*for (int x = 0; x < N; ++x)
        //                 printf("%d ", values[x]);
	// 	*/
        // }
	// else if (i == 4) {
        // // Zero distribution
        //         printf("Data set distribution: Zero\n");
        //         int r = rand() % 100;
        //         for (int x = 0; x < N; ++x) {
        //                 values[x] = r;
        //                 //printf("%d ", values[x]);
        //         }
        // }
        printf("\n");
}

 // Kernel function
 __global__ static void quicksort(int* values) {
 #define MAX_LEVELS	300

	int pivot, L, R;
	int idx =  threadIdx.x + blockIdx.x * blockDim.x;
	int start[MAX_LEVELS];
	int end[MAX_LEVELS];

	start[idx] = idx;
	end[idx] = N - 1;
	while (idx >= 0) {
		L = start[idx];
		R = end[idx];
		if (L < R) {
			pivot = values[L];
			while (L < R) {
				while (values[R] >= pivot && L < R)
					R--;
				if(L < R)
					values[L++] = values[R];
				while (values[L] < pivot && L < R)
					L++;
				if (L < R)
					values[R--] = values[L];
			}
			values[L] = pivot;
			start[idx + 1] = L + 1;
			end[idx + 1] = end[idx];
			end[idx++] = L;
			if (end[idx] - start[idx] > end[idx - 1] - start[idx - 1]) {
	                        // swap start[idx] and start[idx-1]
        	                int tmp = start[idx];
                	        start[idx] = start[idx - 1];
                        	start[idx - 1] = tmp;

	                        // swap end[idx] and end[idx-1]
        	                tmp = end[idx];
                	        end[idx] = end[idx - 1];
                        	end[idx - 1] = tmp;
	                }

		}
		else
			idx--;
	}
}
 
 // program main
 int main(int argc, char **argv) {
        CALI_CXX_MARK_FUNCTION;
        cali::ConfigManager mgr;
        CALI_MARK_BEGIN(comm);
        mgr.start();


	printf("./quicksort starting with %d numbers...\n", N);
 	// unsigned int hTimer;
 	size_t size = N * sizeof(int);
 	
 	// allocate host memory
 	r_values = (int*)malloc(size);
 	
	// allocate device memory
        cudaMalloc((void**)&d_values, size);

	// allocate threads per block
        const unsigned int cThreadsPerBlock = 128;
                
	/* Types of data sets to be sorted:
         *      1. Normal distribution
         *      2. Gaussian distribution
         *      3. Bucket distribution
         *      4. Sorted Distribution
         *      5. Zero Distribution
         */

        CALI_MARK_END(comm);
        
	for (int i = 0; i < 5; ++i) {
                // initialize data set
                CALI_MARK_BEGIN(data_init);
                Init(r_values, i);
                CALI_MARK_END(data_init);

         CALI_MARK_BEGIN(comm);

	 	// copy data to device	
		cudaMemcpy(d_values, r_values, size, cudaMemcpyHostToDevice);

		printf("Beginning kernel execution...\n");

		// cutCreateTimer(&hTimer);
 		cudaThreadSynchronize();
		// cutResetTimer(hTimer);
	 	// cutStartTimer(hTimer);
	
		// execute kernel
        CALI_MARK_END(comm);

        CALI_MARK_BEGIN(comp);
 		quicksort <<< MAX_THREADS / cThreadsPerBlock, MAX_THREADS / cThreadsPerBlock, cThreadsPerBlock >>> (d_values);
	 	cudaDeviceSynchronize();
        CALI_MARK_END(comp);

        CALI_MARK_BEGIN(comm);

 		// cutilSafeCall( cudaThreadSynchronize() );
	 	// cutStopTimer(hTimer);
	 	// double gpuTime = cutGetTimerValue(hTimer);

 		// printf( "\nKernel execution completed in %f ms\n", gpuTime );
 	
	 	// copy data back to host
		cudaMemcpy(r_values, d_values, size, cudaMemcpyDeviceToHost);
 	
	 	// test print
 		/*for (int i = 0; i < N; i++) {
 			printf("%d ", r_values[i]);
 		}
 		printf("\n");
		*/

        CALI_MARK_END(comm);

		// test
        CALI_MARK_BEGIN(checkCorrect);
                printf("\nTesting results...\n");
                for (int x = 0; x < N - 1; x++) {
                        if (r_values[x] > r_values[x + 1]) {
                                printf("Sorting failed.\n");
                                break;
                        }
                        else
                                if (x == N - 2)
                                        printf("SORTING SUCCESSFUL\n");
                }
        CALI_MARK_END(checkCorrect);

	}
 	
 	// free memory
	cudaFree(d_values);
 	free(r_values);
 	
 	// cutilExit(argc, argv);
 	cudaThreadExit();

        int blockSize = N / MAX_THREADS;
        adiak::init(NULL);
        adiak::launchdate();    // launch date of the job
        adiak::libraries();     // Libraries used
        adiak::cmdline();       // Command line used to launch the job
        adiak::clustername();   // Name of the cluster
        adiak::value("Algorithm", "Quick Sort"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
        adiak::value("ProgrammingModel", "Cuda"); // e.g., "MPI", "CUDA", "MPIwithCUDA"
        adiak::value("Datatype", "int"); // The datatype of input elements (e.g., double, int, float)
        adiak::value("SizeOfDatatype", sizeof(int)); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
        adiak::value("InputSize", N); // The number of elements in input dataset (1000)
        adiak::value("InputType", "Sorted"); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
        //     adiak::value("num_procs", num_procs); // The number of processors (MPI ranks)
        adiak::value("check correct", checkCorrect); // check the correctness
        adiak::value("num_threads", MAX_THREADS); // The number of CUDA or OpenMP threads
        adiak::value("num_blocks", blockSize); // The number of CUDA blocks 
        adiak::value("group_num", "7"); // The number of your group (integer, e.g., 1, 10)
        adiak::value("implementation_source", "Online: https://github.com/saigowri/CUDA/blob/master/quicksort.cu"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").

        mgr.stop();
        mgr.flush();
                
        return 0;
}