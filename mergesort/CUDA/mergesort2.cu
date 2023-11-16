// source: https://github.com/Liadrinz/HelloCUDA/blob/master/cuda/MergeSort.cu
// author: Liadrinz
// I am using this source code for CUDA implementation of merge sort. I have added caliper and adiak annotations to the code.

// 归并排序
#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdlib.h>
#include <time.h>
// #include <Windows.h>
#include <algorithm> 
#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

//const char* main_loop = "main loop";
const char* comm =  "comm";
const char* comm_large = "comm_large";
const char* comp = "comp";
const char* comp_large = "comp_large";
const char* data_init = "data_init";
const char* correctness_check = "correctness_check";

__global__ void MergeSort(int *nums, int *temp, int n) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    for (int i = 2; i < 2 * n; i *= 2) {
        int len = i;
        if (n - tid < len) len = n - tid;
        if (tid % i == 0) {
            int *seqA = &nums[tid], lenA = i / 2, j = 0;
            int *seqB = &nums[tid + lenA], lenB = len - lenA, k = 0;
            int p = tid;
            while (j < lenA && k < lenB) {
                if (seqA[j] < seqB[k]) {
                    temp[p++] = seqA[j++];
                } else {
                    temp[p++] = seqB[k++];
                }
            }
            while (j < lenA)
                temp[p++] = seqA[j++];
            while (k < lenB)
                temp[p++] = seqB[k++];
            for (int j = tid; j < tid + len; j++)
                nums[j] = temp[j];
        }
        __syncthreads();
    }
}

int main(int argc, char** argv) {
    CALI_CXX_MARK_FUNCTION;
    cali::ConfigManager mgr;
	mgr.start();
    //float s = GetTickCount();

    // 初始化数列
    int size = atoi(argv[1]);//1024 * 1024;
    int *nums = (int*)malloc(sizeof(int) * size);
    srand(time(0));

    CALI_MARK_BEGIN(data_init);
    // for random input
    // for (int i = 0; i < size; ++i) {
    //     nums[i] = rand() % size;
    // }

    // for sorted input
    // for (int i = 0; i < size; ++i) {
    //     nums[i] = i;
    //     // printf("%d ", nums[i]);
    // }

    // for reverse sorted input
    for (int i = 0; i < size; ++i) {
        nums[i] = size - i;
    }
    CALI_MARK_END(data_init);

    // 拷贝到设备
    int *dNums;
    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    cudaMalloc((void**)&dNums, sizeof(int) * size);
    cudaMemcpy(dNums, nums, sizeof(int) * size, cudaMemcpyHostToDevice);

    // 临时存储
    int *dTemp;
    cudaMalloc((void**)&dTemp, sizeof(int) * size);
    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);
    int threadsize = atoi(argv[2]);
    printf("The size of the input is %d\n", size);
    printf("The number of threads is %d\n", threadsize);

    dim3 threadPerBlock(atoi(argv[2]));//it is 1024
    dim3 blockNum((size + threadPerBlock.x - 1) / threadPerBlock.x, 1, 1);
    
    float blocknum = (size + threadsize - 1) / threadsize;

    

    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_large);
    MergeSort<<<blockNum, threadPerBlock>>>(dNums, dTemp, size);
    CALI_MARK_END(comp_large);
    CALI_MARK_END(comp);

    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    cudaMemcpy(nums, dNums, sizeof(int) * size, cudaMemcpyDeviceToHost);
    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);

    // 打印结果
    // for (int i = 0; i < size; ++i) {
    //     printf("%d ", nums[i]);
    // }
    // printf("\n");
    CALI_MARK_BEGIN(correctness_check);
    bool sorting = std::is_sorted(nums, nums + size);
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

    free(nums);
    cudaFree(dNums);
    cudaFree(dTemp);

    printf("Number of numbers: %d\n", size);
    // printf("Sorting time: %fms\n", GetTickCount() - s);

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
    adiak::value("InputSize", size); // The number of elements in input dataset (1000)
    adiak::value("InputType", "Random"); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    //adiak::value("num_procs", world_size); // The number of processors (MPI ranks)
    adiak::value("num_threads", threadsize); // The number of CUDA or OpenMP threads
    adiak::value("num_blocks", blocknum); // The number of CUDA blocks 
    adiak::value("group_num", "7"); // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "Online: https://github.com/Liadrinz/HelloCUDA/blob/master/cuda/MergeSort.cu"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").


}


