#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

#define OK 1
#define EXPECTATION_ERROR 1
#define MALLOC_ERROR 2
#define CUDA_ERROR 3

// source: https://github.com/chrishadi/cuda-sort/tree/main
// author: Chris Hadi
// I am using this source code for CUDA implementation of merge sort. I have added caliper and adiak annotations to the code.

const char* main_loop = "main loop";
const char* comm =  "comm";
const char* comm_large = "comm_large";
const char* small_comm1 = "small_comm1";
const char* small_comm2 = "small_comm2";
const char* comp = "comp";
const char* comp_large = "comp_large";
const char* data_init = "data_init";
const char* correctness = "correctness";


cudaError_t mergeSortWithCuda(int* arr, unsigned int count);
bool assertArrEq(int* expected, int* actual, size_t size);
int testMergeSortWithCuda(int* actual, int* expected, const unsigned int count);

typedef struct mergeSortResult {
    cudaError_t cudaStatus;
    char* msg;
} mergeSortResult_t;

__global__ void mergeSortKernel(int* arr, int* aux, unsigned int blockSize, const unsigned int last)
{
    int x = threadIdx.x;
    int start = blockSize * x;
    int end = start + blockSize - 1;
    int mid = start + (blockSize / 2) - 1;
    int l = start, r = mid + 1, i = start;

    if (end > last) { end = last; }
    if (start == end || end <= mid) { return; }

    while (l <= mid && r <= end) {
        if (arr[l] <= arr[r]) {
            aux[i++] = arr[l++];
        }
        else {
            aux[i++] = arr[r++];
        }
    }

    while (l <= mid) { aux[i++] = arr[l++]; }
    while (r <= end) { aux[i++] = arr[r++]; }

    for (i = start; i <= end; i++) {
        arr[i] = aux[i];
    }
}

inline mergeSortResult_t mergeSortError(cudaError_t cudaStatus, char* msg) {
    mergeSortResult_t error;
    error.cudaStatus = cudaStatus;
    error.msg = msg;
    return error;
}

inline mergeSortResult_t mergeSortSuccess() {
    mergeSortResult_t success;
    success.cudaStatus = cudaSuccess;
    return success;
}

inline mergeSortResult_t doMergeSortWithCuda(int* arr, unsigned int count, int* dev_arr, int* dev_aux) {
    const unsigned int last = count - 1;
    const unsigned size = count * sizeof(int);
    unsigned int threadCount;
    cudaError_t cudaStatus;
    char msg[1024];

    // Copy input vectors from host memory to GPU buffers.
    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);

    CALI_MARK_BEGIN(small_comm1);
    cudaStatus = cudaMemcpy(dev_arr, arr, size, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        return mergeSortError(cudaStatus, "cudaMemcpy failed!");
    }
    CALI_MARK_END(small_comm1);

    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_large);
    for (unsigned int blockSize = 2; blockSize < 2 * count; blockSize *= 2) {
        threadCount = count / blockSize;
        if (count % blockSize > 0) { threadCount++; }

        // Launch a kernel on the GPU with one thread for each block.
        mergeSortKernel<<<1, threadCount>>>(dev_arr, dev_aux, blockSize, last);

        // Check for any errors launching the kernel
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            sprintf(msg, "mergeSortKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            return mergeSortError(cudaStatus, msg);
        }

        // cudaDeviceSynchronize waits for the kernel to finish, and returns
        // any errors encountered during the launch.
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            sprintf(msg, "cudaDeviceSynchronize returned error code %d after launching mergeSortKernel!\n", cudaStatus);
            return mergeSortError(cudaStatus, msg);
        }
    }
    CALI_MARK_END(comp_large);
    CALI_MARK_END(comp);
    // Copy output vector from GPU buffer to host memory.
    CALI_MARK_BEGIN(small_comm2);
    cudaStatus = cudaMemcpy(arr, dev_arr, size, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        return mergeSortError(cudaStatus, "cudaMemcpy failed!");
    }
    CALI_MARK_END(small_comm2);
    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);

    return mergeSortSuccess();
}

cudaError_t mergeSortWithCuda(int* arr, unsigned int count)
{
    const unsigned int size = count * sizeof(int);
    int* dev_arr = 0;
    int* dev_aux = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed! Do you have a CUDA-capable GPU installed?");
        return cudaStatus;
    }

    // Allocate GPU buffers for two vectors (main and aux array).
    cudaStatus = cudaMalloc((void**)&dev_arr, size);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        return cudaStatus;
    }

    cudaStatus = cudaMalloc((void**)&dev_aux, size);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        cudaFree(dev_arr);
        return cudaStatus;
    }

    mergeSortResult_t result = doMergeSortWithCuda(arr, count, dev_arr, dev_aux);

    if (result.cudaStatus != cudaSuccess) {
        fprintf(stderr, result.msg);
    }

    cudaFree(dev_arr);
    cudaFree(dev_aux);

    return cudaStatus;
}

int main()
{
    CALI_CXX_MARK_FUNCTION;
    cali::ConfigManager mgr;
	mgr.start();
    CALI_MARK_BEGIN(main_loop);
    const unsigned int count = 64;
    const unsigned int size = count * sizeof(int);
    int status = MALLOC_ERROR;
    int* actual = (int*) malloc(size);
    int* expected = (int*) malloc(size);

    if (actual != NULL && expected != NULL) {
        status = testMergeSortWithCuda(actual, expected, count);
    }
    else {
        fprintf(stderr, "malloc failed!");
    }

    free(actual);
    free(expected);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    int cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return CUDA_ERROR;
    }
    CALI_MARK_END(main_loop);
    mgr.stop();
   	mgr.flush();
    adiak::init(NULL);
    adiak::launchdate();    // launch date of the job
    adiak::libraries();     // Libraries used
    adiak::cmdline();       // Command line used to launch the job
    adiak::clustername();   // Name of the cluster
    adiak::value("Algorithm", "Merge sort"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", "CUDA"); // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", "int"); // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", "4 bytes"); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", count); // The number of elements in input dataset (1000)
    adiak::value("InputType", "Random"); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    //adiak::value("num_procs", world_size); // The number of processors (MPI ranks)
    adiak::value("num_threads", count); // The number of CUDA or OpenMP threads
    adiak::value("num_blocks", count); // The number of CUDA blocks 
    adiak::value("group_num", "7"); // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "Online: https://github.com/chrishadi/cuda-sort/tree/main"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").

    return status;
}

int cmpInt(const void* a, const void* b) {
    return *(int*)a - *(int*)b;
}

int testMergeSortWithCuda(int* actual, int* expected, const unsigned int count) {
    CALI_MARK_BEGIN(data_init);
    for (unsigned int i = 0; i < count; i++) {
        expected[i] = actual[i] = rand();
    }
    CALI_MARK_END(data_init);

    qsort(expected, count, sizeof(int), cmpInt);

    cudaError_t cudaStatus = mergeSortWithCuda(actual, count);
    CALI_MARK_BEGIN(correctness);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "mergeSortWithCuda failed!");
        return CUDA_ERROR;
    }

    if (!assertArrEq(expected, actual, count * sizeof(int))) {
        puts("cuda sorted array is not equal to the qsorted array!");
        return EXPECTATION_ERROR;
    }
    puts("This is sorted by mergesort correctly");
    puts("test ok.");
    CALI_MARK_END(correctness);
    return OK;
    
}

bool assertArrEq(int* expected, int* actual, size_t size) {
    return memcmp(expected, actual, size) == 0;
}