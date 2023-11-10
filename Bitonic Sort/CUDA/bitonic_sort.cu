/*
 * Parallel bitonic sort using CUDA.
 * Compile with
 * nvcc bitonic_sort.cu
 * Based on http://www.tools-of-computing.com/tc/CS/Sorts/bitonic_sort.htm
 * License: BSD 3
 */

#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

#include <cuda.h>
#include <cuda_runtime.h>

int THREADS;
int BLOCKS;
int NUM_VALS;

const char* bitonic_sort_step_region = "bitonic_sort_step";
const char* cudaMemcpy_host_to_device = "cudaMemcpy_host_to_device";
const char* cudaMemcpy_device_to_host = "cudaMemcpy_device_to_host";
const char* main = "main";
const char* correctness_check = "correctness_check";
const char* comm = "comm";
const char* comm_large = "comm_large";
const char* comp = "comp";
const char* comp_large = "comp_large";
 

CALI_CXX_MARK_FUNCTION;

cudaEvent_t host_to_device_start, host_to_device_stop;
cudaEvent_t device_to_host_start, device_to_host_stop;
cudaEvent_t bitonic_sort_step_start, bitonic_sort_step_stop;

float kernel_counter = 0;

void print_elapsed(clock_t start, clock_t stop)
{
  double elapsed = ((double) (stop - start)) / CLOCKS_PER_SEC;
  printf("Elapsed time: %.3fs\n", elapsed);
}

float random_float()
{
  return (float)rand()/(float)RAND_MAX;
}

void array_print(float *arr, int length)
{
  int i;
  for (i = 0; i < length; ++i) {
    printf("%1.3f ",  arr[i]);
  }
  printf("\n");
}

void array_fill(float *arr, int length)
{
  srand(time(NULL));
  int i;
  for (i = 0; i < length; ++i) {
    arr[i] = random_float();
  }
}

__global__ void bitonic_sort_step(float *dev_values, int j, int k)
{
  unsigned int i, ixj; /* Sorting partners: i and ixj */
  i = threadIdx.x + blockDim.x * blockIdx.x;
  ixj = i^j;

  /* The threads with the lowest ids sort the array. */
  if ((ixj)>i) {
    if ((i&k)==0) {
      /* Sort ascending */
      if (dev_values[i]>dev_values[ixj]) {
        /* exchange(i,ixj); */
        float temp = dev_values[i];
        dev_values[i] = dev_values[ixj];
        dev_values[ixj] = temp;
      }
    }
    if ((i&k)!=0) {
      /* Sort descending */
      if (dev_values[i]<dev_values[ixj]) {
        /* exchange(i,ixj); */
        float temp = dev_values[i];
        dev_values[i] = dev_values[ixj];
        dev_values[ixj] = temp;
      }
    }
  }
}

/**
 * Inplace bitonic sort using CUDA.
 */
void bitonic_sort(float *values)
{
  float *dev_values;
  size_t size = NUM_VALS * sizeof(float);

  cudaMalloc((void**) &dev_values, size);

  //MEM COPY FROM HOST TO DEVICE
  CALI_MARK_BEGIN("comm");
  CALI_MARK_BEGIN("comm_large");
  CALI_MARK_BEGIN("cudaMemcpy_host_to_device");
  cudaEventRecord(host_to_device_start);
  cudaMemcpy(dev_values, values, size, cudaMemcpyHostToDevice);
  cudaEventSynchronize(host_to_device_stop);
  cudaEventRecord(host_to_device_stop);
  CALI_MARK_END("cudaMemcpy_host_to_device");
  CALI_MARK_END("comm_large");
  CALI_MARK_END("comm")

  dim3 blocks(BLOCKS,1);    /* Number of blocks   */
  dim3 threads(THREADS,1);  /* Number of threads  */

  int j, k;
  CALI_MARK_BEGIN("comp");
  CALI_MARK_BEGIN("comp_large");
  CALI_MARK_BEGIN("bitonic_sort_step");
  cudaEventRecord(bitonic_sort_step_start);

  /* Major step */
  for (k = 2; k <= NUM_VALS; k <<= 1) {
    /* Minor step */
    for (j=k>>1; j>0; j=j>>1) {
      kernel_counter++;
      bitonic_sort_step<<<blocks, threads>>>(dev_values, j, k);
    }
  }
  cudaDeviceSynchronize();
  cudaEventRecord(bitonic_sort_step_stop);
  CALI_MARK_END("bitonic_sort_step");
  CALI_MARK_END("comp_large");
  CALI_MARK_END("comp");

  CALI_MARK_BEGIN("comm");
  CALI_MARK_BEGIN("comm_large");
  //MEM COPY FROM DEVICE TO HOST
  CALI_MARK_BEGIN("cudaMemcpy_device_to_host");
  cudaEventRecord(device_to_host_start);
  cudaMemcpy(values, dev_values, size, cudaMemcpyDeviceToHost);
  cudaEventSynchronize(device_to_host_stop);
  cudaEventRecord(device_to_host_stop);
  CALI_MARK_END("cudaMemcpy_device_to_host");
  CALI_MARK_END("comm_large");
  CALI_MARK_END("comm");

  cudaFree(dev_values);
}

int main(int argc, char *argv[])
{
  CALI_MARK_BEGIN("main");
  THREADS = atoi(argv[1]);
  NUM_VALS = atoi(argv[2]);
  BLOCKS = NUM_VALS / THREADS;

  printf("Number of threads: %d\n", THREADS);
  printf("Number of values: %d\n", NUM_VALS);
  printf("Number of blocks: %d\n", BLOCKS);

  // Create caliper ConfigManager object
  cali::ConfigManager mgr;
  mgr.start();

  cudaEventCreate(&host_to_device_start);
  cudaEventCreate(&host_to_device_stop);
  cudaEventCreate(&device_to_host_start);
  cudaEventCreate(&device_to_host_stop);
  cudaEventCreate(&bitonic_sort_step_start);
  cudaEventCreate(&bitonic_sort_step_stop);

  clock_t start, stop;

  float *values = (float*) malloc( NUM_VALS * sizeof(float));
  array_fill(values, NUM_VALS);

  start = clock();
  bitonic_sort(values); /* Inplace */
  stop = clock();
  cudaEventSynchronize(bitonic_sort_step_stop);

  print_elapsed(start, stop);

  // Store results in these variables.
  float effective_bandwidth_gb_s;
  float bitonic_sort_step_time;
  float cudaMemcpy_host_to_device_time;
  float cudaMemcpy_device_to_host_time;

  // Get the elapsed time for each event.
  cudaEventElapsedTime(&bitonic_sort_step_time, bitonic_sort_step_start, bitonic_sort_step_stop);
  cudaEventElapsedTime(&cudaMemcpy_host_to_device_time, host_to_device_start, host_to_device_stop);
  cudaEventElapsedTime(&cudaMemcpy_device_to_host_time, device_to_host_start, device_to_host_stop);

  // Calculate the effective bandwidth in GB/s
  float total_data_size = NUM_VALS * 4 * 4 * kernel_counter / (1024 * 1024 * 1024);
  effective_bandwidth_gb_s = total_data_size / (bitonic_sort_step_time / 1000);

  printf("Effective bandwidth: %f GB/s\n", effective_bandwidth_gb_s);
  printf("Elapsed time (cudamemcpy host to device): %f\n", cudaMemcpy_host_to_device_time/1000);
  printf("Elapsed time (cudamemcpy device to host): %f\n", cudaMemcpy_device_to_host_time/1000);
  printf("Elapsed time (bitonic sort step time): %f\n", bitonic_sort_step_time/1000);

  CALI_MARK_BEGIN("correctness_check");
  // Check that the array is sorted.
  int i = 0;
  int j = i + 1;
  for(; j < NUM_VALS; i++, j++) {
    if(values[i] > values[j]) {
      printf("Array not sorted correctly.\n");
      break;
    }
  }
  CALI_MARK_END("correctness_check");
  CALI_MARK_END("main");

  adiak::init(NULL);
  adiak::user();
  adiak::launchdate();
  adiak::libraries();
  adiak::cmdline();
  adiak::clustername();
  adiak::value("num_threads", THREADS);
  adiak::value("num_blocks", BLOCKS);
  adiak::value("num_vals", NUM_VALS);
  adiak::value("program_name", "cuda_bitonic_sort");
  adiak::value("datatype_size", sizeof(float));
  adiak::value("effective_bandwidth (GB/s)", effective_bandwidth_gb_s);
  adiak::value("bitonic_sort_step_time", bitonic_sort_step_time);
  adiak::value("cudaMemcpy_host_to_device_time", cudaMemcpy_host_to_device_time);
  adiak::value("cudaMemcpy_device_to_host_time", cudaMemcpy_device_to_host_time);

  // Flush Caliper output before finalizing MPI
  mgr.stop();
  mgr.flush();
}
