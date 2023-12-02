# CSCE 435 Group project

## 0. Group number:

## 1. Group members:

1. Jiangyuan Liu
2. Akhil Mathew
3. Jacob Thomas
4. Ashwin Kundeti

## The way our team is communicating is by using Discord and iMessages

## 2. _due 10/25_ Project topic

For our project topic, we are going to be exploring parellel algoithm for sorting.

### 2a. Brief project description (what algorithms will you be comparing and on what architectures)

Merge Sort (MPI)
Merge Sort (CUDA)

Radix Sort (MPI)
Radix Sort (CUDA)

Quick Sort (MPI)
Quick Sort (CUDA)

Bitonic Sort (MPI)
Bitonic Sort (CUDA)

### 2b. Pseudocode for each parallel algorithm

1. Merge sort

```
For MPI implimentation:

function parallelMergeSort(array):
  if MPI parent:
    localArray = array
    MPI_Scatter(array, localArray, size, MPI_DATATYPE, MPI_ROOT, MPI_COMM_WORLD)
    for each MPI process:
      if MPI process rank != MPI parent rank:
        MPI_Send(localArray, MPI process rank)
      else:
        MPI_Receive(newArray, MPI process rank)
        merge(localArray, newArray)
    MPI_Gather(localArray, array, size, MPI_DATATYPE, MPI_ROOT, MPI_COMM_WORLD)
    return array
  else:
    localArray = array
    MPI_Scatter(array, localArray, size, MPI_DATATYPE, MPI_ROOT, MPI_COMM_WORLD)
    for each MPI process:
      if MPI process rank != MPI parent rank:
        MPI_Receive(newArray, MPI process rank)
        merge(localArray, newArray)
      else:
        MPI_Send(localArray, MPI process rank)
    MPI_Gather(localArray, array, size, MPI_DATATYPE, MPI_ROOT, MPI_COMM_WORLD)

For the CUDA implimentation:

# Host code
function parallelMergeSort(array):
    # Allocate memory on GPU
    cudaMalloc(deviceArray, size)

    # Copy data from host to device
    cudaMemcpy(deviceArray, array, size, cudaMemcpyHostToDevice)

    # Launch kernel with specified number of blocks and threads
    parallelMergeSortKernel<<<numBlocks, numThreads>>>(deviceArray, size)

    # Copy sorted data from device to host
    cudaMemcpy(array, deviceArray, size, cudaMemcpyDeviceToHost)

    # Free allocated memory on GPU
    cudaFree(deviceArray)

# Device code
function parallelMergeSortKernel(deviceArray, size):
    localArray = deviceArray  # Each block has its own copy of localArray

    # Perform merge sort on localArray
    mergeSort(localArray, size)

    # Synchronize threads within the block before returning
    __syncthreads()

# Function to perform merge sort on a given array
function mergeSort(array, size):
    # Base case: If the array is of size 1 or empty, it's already sorted
    if size <= 1:
        return

    # Split the array into two halves
    mid = size / 2
    leftArray = array[:mid]
    rightArray = array[mid:]

    # Recursively sort the two halves
    mergeSort(leftArray, mid)
    mergeSort(rightArray, size - mid)

    # Merge the sorted halves
    merge(array, leftArray, mid, rightArray, size - mid)

# Function to merge two sorted arrays
function merge(array, leftArray, leftSize, rightArray, rightSize):
    i = 0
    j = 0
    k = 0

    # Compare elements of left and right arrays and merge them in sorted order
    while i < leftSize and j < rightSize:
        if leftArray[i] <= rightArray[j]:
            array[k] = leftArray[i]
            i += 1
        else:
            array[k] = rightArray[j]
            j += 1
        k += 1

    # Copy the remaining elements of leftArray, if any
    while i < leftSize:
        array[k] = leftArray[i]
        i += 1
        k += 1

    # Copy the remaining elements of rightArray, if any
    while j < rightSize:
        array[k] = rightArray[j]
        j += 1
        k += 1


```

Source:
https://rachitvasudeva.medium.com/parallel-merge-sort-algorithm-e8175ab60e7
https://www.sjsu.edu/people/robert.chun/courses/cs159/s3/T.pdf
ChatGPT: https://chat.openai.com/

2. Bitonic sort

```
Pseudocode:
Bitonic sort (MPI):
bitonic_sort(A, direction):
    n = A_size
    if n > 1:
        // Split data among processes
        local_A = split_data(A, A_size)

        Bitonic_sort(local_A, direction) // Top half
        Bitonic_sort(local_A, direction) // Bottom half

        // Synchronize before merging
        MPI_Barrier(MPI_COMM_WORLD)

        Merge(local_A, direction) // Merge the first and second half

        // Synchronize after merging
        MPI_Barrier(MPI_COMM_WORLD)
    end
end

Merge (MPI, A, direction):
    n = A_size
    if n > 1:
        for i in range(0, n/2):
            if A[i] > A[i + n/2]:
                swap(A[i], A[i + n/2])
            end
        end
    end
end

Main (MPI):
    Initialize MPI
    Get numTasks and rank
    If rank == 0:
        // Initialize 'A' with data
        A = initialize_data(A_size)
    MPI_Bcast(A, A_size, MPI_INT, 0, MPI_COMM_WORLD) // Broadcast 'A' to all processes
    bitonic_sort(A, direction) // Perform the bitonic sort

    // Gather sorted data to rank 0
    All_A = gather_data(A, A_size)

    If rank == 0:
        Merge (MPI, All_A, direction) // Merge sorted data on rank 0
        Print sorted result
    MPI_Finalize() // Finalize MPI
End
```

source 1: https://www.baeldung.com/cs/bitonic-sort

source 2: OpenAI. (2023). ChatGPT [Large language model]. https://chat.openai.com

CUDA

```
// Kernel function to perform bitonic sort
function bitonicSortKernel(values, j, k):
  // same as before

// Function to perform bitonic sort on GPU
function cudaBitonicSort(values, size):

  // Allocate memory on GPU
  dev_values = cudaMalloc(size * sizeof(int))

  // Copy values to GPU
  cudaMemcpy(dev_values, values, size * sizeof(int), cudaMemcpyHostToDevice)

  // Define block size and grid size
  // same as before

  // Loop over stages
  for k = 2 to size by powers of 2:
    for j = k/2 down to 1:
      launch bitonicSortKernel with grid_size blocks and block_size threads per block,
        passing dev_values, j, and k

      cudaDeviceSynchronize() // wait for kernel to finish

  // Copy sorted data back to CPU
  cudaMemcpy(values, dev_values, size * sizeof(int), cudaMemcpyDeviceToHost)

  // Free GPU memory
  cudaFree(dev_values)
```

source 1: https://codepal.ai/code-generator/query/15oCYvGw/bitonic-sort-cuda

source 2: https://claude.ai/

3. Quicksort
   MPI

```
procedure parallel_quicksort(A[1...n])
  begin
    initialize MPI
    rank := MPI_Comm_rank(MPI_COMM_WORLD)
    size := MPI_Comm_size(MPI_COMM_WORLD)

    local_n := n / size
    local_A[1...local_n]

    MPI_Scatter(A, local_A, local_n, MPI_INT, 0, MPI_COMM_WORLD)

    quicksort(local_A, local_n)

    MPI_Gather(local_A, local_n, MPI_INT, A, local_n, MPI_INT, 0, MPI_COMM_WORLD)

    if rank == 0 then
      perform_final_merge_or_postprocessing(A)

    MPI_Finalize()
  end parallel_quicksort

procedure quicksort(A[1...n], n)
  begin
    if n <= 1 then
      return

    pivot := choose_pivot(A, n)
    pivot_idx := partition(A, pivot, n)
    left_size := pivot_idx
    right_size := n - pivot_idx - 1

    split_data(A, left_A, right_A, pivot_idx, n)

    parallel_quicksort(left_A)
    parallel_quicksort(right_A)

    merge_sorted_arrays(A, left_A, right_A)
  end quicksort

```

CUDA

```
procedure cuda_quicksort(A[1...n])
  begin
    // Copy data from host to device
    cudaMemcpy(d_A, A, n * sizeof(int), cudaMemcpyHostToDevice)

    // Launch the CUDA quicksort kernel
    dim3 block_size(512)
    dim3 grid_size((n + block_size.x - 1) / block_size.x)
    cuda_quicksort_kernel<<<grid_size, block_size>>>(d_A, n)

    // Wait for the kernel to finish
    cudaDeviceSynchronize()

    // Copy sorted data back from device to host
    cudaMemcpy(A, d_A, n * sizeof(int), cudaMemcpyDeviceToHost)

    // Free device memory
    cudaFree(d_A)
  end cuda_quicksort

__global__ void cuda_quicksort_kernel(int* A, int n)
  begin
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int size = 2; size < n; size *= 2)
      begin
        int half_size = size / 2;

        for (int sub_size = half_size; sub_size > 0; sub_size /= 2)
          begin
            int step = half_size / sub_size;
            int middle = thread_id * step;
            int left = middle - half_size / 2;
            int right = middle + half_size / 2;

            if (right < n && A[right] < A[left])
              swap(A[left], A[right]);

            __syncthreads();
          end

        __syncthreads();
      end
  end cuda_quicksort_kernel
```

https://chat.openai.com

4. Radix Sort

```
Radix-Sort (MPI, A, d):
    // It works similarly to Counting Sort for d number of passes.
    // Each key in A[1..n] is a d-digit integer.
    // Digits are numbered 1 to d from right to left.

    Initialize MPI
    Get numTasks and rank

    for j = 1 to d do
        // Local counts for each process
        int local_count[10] = {0}

        // Count the number of keys at each digit place (pass j)
        for i = 0 to n do
            local_count[key_of(A[i], j)]++

        // Gather local counts to rank 0
        MPI_Gather(local_count, 10, MPI_INT, global_count, 10, MPI_INT, 0, MPI_COMM_WORLD)

        if rank == 0:
            // Calculate cumulative counts
            for k = 1 to 10 do
                global_count[k] = global_count[k] + global_count[k-1]

        // Broadcast global counts to all processes
        MPI_Bcast(global_count, 10, MPI_INT, 0, MPI_COMM_WORLD)

        // Initialize result array
        int result[n]

        // Build the resulting array by checking the new position of A[i] using count
        for i = n-1 downto 0 do
            result[global_count[key_of(A[i], j)]] = A[i]
            global_count[key_of(A[i], j)]--

        // Update A with the sorted result
        for i = 0 to n do
            A[i] = result[i]

    end for (j)

    MPI_Finalize() // Finalize MPI
End
```

CUDA

```
Define constants:
    WSIZE = 32
    LOOPS = 1
    UPPER_BIT = 10
    LOWER_BIT = 0

Declare global device array ddata[WSIZE]

Define kernel function parallelRadix():
    Declare shared volatile array sdata[WSIZE * 2]
    Declare unsigned integer bitmask, offset, thrmask, mypos

    Load ddata[threadIdx.x] into sdata[threadIdx.x]

    For each bit position from LOWER_BIT to UPPER_BIT:
        Get mydata from sdata[((WSIZE - 1) - threadIdx.x) + offset]
        Extract mybit using bitmask from mydata

        Get ones and zeroes count using __ballot()

        Switch ping-pong buffers
        Do zeroes and ones:
            If mybit is zero:
                Calculate my position in the ping-pong buffer for zeroes
            Else:
                Calculate my position in the ping-pong buffer for ones

        Move mydata to the appropriate position in the buffer
        Update bitmask for the next bit

    Put the sorted results back to global ddata[threadIdx.x]

Define main function:
    Declare unsigned integer array hdata[WSIZE]
    Declare float totalTime = 0

    For each loop iteration from 0 to LOOPS:
        Seed random number generator
        Initialize range as 2 to the power of UPPER_BIT
        Fill hdata array with values from 0 to WSIZE - 1
        Copy hdata to ddata on the device

        Call parallelRadix kernel with 1 block and WSIZE threads per block

        Synchronize device to ensure kernel completion

        Copy sorted data from ddata to hdata

    Print results:
        Print "Parallel Radix Sort:"
        Print "Array size = WSIZE * LOOPS"
        Print "Time elapsed = totalTime seconds"

    Return 0
```

Source 1: https://www.codingeek.com/algorithms/radix-sort-explanation-pseudocode-and-implementation/
Source 2: https://github.com/ufukomer/cuda-radix-sort/blob/master/docs/Radix%20Sort%20Analyses%20in%20Parallel%20and%20Serial%20Way.pdf
Source 3: OpenAI. (2023). ChatGPT [Large language model]. https://chat.openai.com

### 2c. Evaluation plan - what and how will you measure and compare

The way we want to compare the different versions of the code is by using CPU-only (MPI) and GPU-only (CUDA) and time it to see how long it takes for the cases to run. We are also going to be comparing them with the same task and see how only it takes for each one of them to run.

## 3. Project implementation

We were unable to gather the required .cali files due to maintenance being performed on Grace. We had scheduled to finish this assignment on Wednesday but due to unforeseen circumstances, we were unable to finish.

## 4. Performance evaluation

Include detailed analysis of computation performance, communication performance. Include figures and explanation of your analysis.

## Mergesort

#### CUDA with strong scaling:

For 655336:

![65536 main](https://i.imgur.com/NNj6YKU.png)
![65536 comp large](https://i.imgur.com/vQ6cHCJ.png)
![65536 comm](https://i.imgur.com/ij2RlBl.png)

For 262144:

![262144 main](https://i.imgur.com/NpRF5vD.png)
![262144 comp_large](https://i.imgur.com/BqNV04O.png)
![262144 comm](https://i.imgur.com/YP56LQZ.png)

For 1048576:

![1048576 main](https://i.imgur.com/LMW12lq.png)
![1048576 comp large](https://i.imgur.com/JRTcjxI.png)
![1048576 comm](https://i.imgur.com/muqCp4p.png)

For 4194304:

![4194304 main](https://i.imgur.com/OPCFXxN.png)
![4194304 comp large](https://i.imgur.com/Q4yQOeA.png)
![4194304 comm](https://i.imgur.com/Ia8oGoF.png)

For 16777216:

![16777216 main](https://i.imgur.com/yydChOq.png)
![16777216 comp large](https://i.imgur.com/tMUw1ca.png)
![16777216 comm](https://i.imgur.com/LY9o3WP.png)

For 67108864:

![67108864 main](https://i.imgur.com/7dgyTqd.png)
![67108864 comp large](https://i.imgur.com/xq36iOM.png)
![67108864 comm](https://i.imgur.com/Pprjclx.png)

#### CUDA with weak scaling:

For Random:

![Imgur](https://i.imgur.com/CHNm72S.png)
![Random comp large](https://i.imgur.com/dFNIE2E.png)
![Imgur](https://i.imgur.com/gwKP41I.png)

For Reverse:

![Imgur](https://i.imgur.com/ROhwwRF.png)
![Reverse comp large](https://i.imgur.com/JiHheXc.png)
![Reverse comm](https://i.imgur.com/Z82ekcj.png)

For Sorted:

![Imgur](https://i.imgur.com/6gex3EN.png)
![sorted comp_lage](https://i.imgur.com/pmLuvLH.png)
![sorted comm](https://i.imgur.com/fUo3A9I.png)

#### CUDA with speedup:

For Random:

![Random main](https://i.imgur.com/SXtarsG.png)
![Random comp large](https://i.imgur.com/r2W8p1G.png)
![Random comm](https://i.imgur.com/zAS0B8S.png)

For Reverse:

![Reverse](https://i.imgur.com/jUcEcAf.png)
![Reverse comp large](https://i.imgur.com/1IlStzq.png)
![Reverse comm](https://i.imgur.com/rl5reSz.png)

For Sorted:

![Sorted main](https://i.imgur.com/dSsAwDR.png)
![Sorted comp large](https://i.imgur.com/sP9Cbom.png)
![Sorted comm](https://i.imgur.com/1TtxX7w.png)

`
`

#### MPI with strong scaling:

For 65536:

![65536 main](https://i.imgur.com/gcYvOLq.png)
![65536 comp large](https://i.imgur.com/mmKRdYP.png)
![65536 comm](https://i.imgur.com/gdq2CVP.png)

For 262144:

![262144 main](https://i.imgur.com/wyyAVRw.png)
![262144 comp large](https://i.imgur.com/ajG8TzY.png)
![262144 comm](https://i.imgur.com/G2VC4ed.png)

For 1048576:

![1048576 main](https://i.imgur.com/MNmzLW4.png)
![1048576 comp large](https://i.imgur.com/Mgt06Xc.png)
![1048576 comm](https://i.imgur.com/mEE2k5k.png)

For 4194304:

![4194304 main](https://i.imgur.com/BGv8igX.png)
![4194304 comp large](https://i.imgur.com/3OZQebK.png)
![4194304 comm](https://i.imgur.com/uYf27ES.png)

For 16777216:

![16777216 main](https://i.imgur.com/Jj0V9Hg.png)
![16777216 comp large](https://i.imgur.com/XKOxj6m.png)
![16777216 comm](https://i.imgur.com/3VzIzPl.png)

For 67108864:

![67108864 main](https://i.imgur.com/hM2I9p4.png)
![67108864 comp large](https://i.imgur.com/nRylWOg.png)
![67108864 comm](https://i.imgur.com/bfTIrSn.png)

For 268435456:

![268435456 main](https://i.imgur.com/MV5F080.png)
![268435456 comp large](https://i.imgur.com/SpeF8Fo.png)
![268435456 comm](https://i.imgur.com/IRfYNVk.png)

#### MPI with weak scaling:

For Random:

![Random main](https://i.imgur.com/6bbbS2u.png)
![Random comp large](https://i.imgur.com/zYSkfLe.png)
![Random comm](https://i.imgur.com/KHiCCtf.png)

For Reverse:

![Reverse main](https://i.imgur.com/cXipYBr.png)
![Reverse comp large](https://i.imgur.com/IOe5j37.png)
![Reverse comm](https://i.imgur.com/UZa9NEQ.png)

For Sorted:

![Sorted main](https://i.imgur.com/yCR7QuO.png)
![Sorted comp large](https://i.imgur.com/AjNOutz.png)
![Sorted comm](https://i.imgur.com/7870HUz.png)

#### MPI with speedup:

For Random:

![Random main](https://i.imgur.com/8YCJHIj.png)
![Random comp large](https://i.imgur.com/DgcuYeo.png)
![Random comm](https://i.imgur.com/2jOEITc.png)

For Reverse:

![Reverse main](https://i.imgur.com/hVkZH7K.png)
![Reverse comp large](https://i.imgur.com/s2CrRd7.png)
![Reverse comm](https://i.imgur.com/vaKgkVP.png)

For Sorted:

![Sorted main](https://i.imgur.com/eIazqj2.png)
![Sorted comp large](https://i.imgur.com/6r6Z3DP.png)
![Sorted comm](https://i.imgur.com/Q7WBcRm.png)

## Radix Sort

### MPI:

#### Strong Scaling:

For 655336:

![65536 main](/Radix/MPI%20Pics/strong/radix-mpi-strong-mainRegion-RandomInput-65536Size.png)
![65536 comm](/Radix/MPI%20Pics/strong/radix-mpi-strong-commRegion-RandomInput-65536Size.png)
![65536 comp](/Radix/MPI%20Pics/strong/radix-mpi-strong-comp_largeRegion-RandomInput-65536Size.png)

For 262144:

![262144 main](/Radix/MPI%20Pics/strong/radix-mpi-strong-mainRegion-RandomInput-262144Size.png)
![262144 comm](/Radix/MPI%20Pics/strong/radix-mpi-strong-commRegion-RandomInput-262144Size.png)
![262144 comp](/Radix/MPI%20Pics/strong/radix-mpi-strong-comp_largeRegion-RandomInput-262144Size.png)

For 1048576:

![1048576 main](/Radix/MPI%20Pics/strong/radix-mpi-strong-mainRegion-SortedInput-1048576Size.png)
![1048576 comm](/Radix/MPI%20Pics/strong/radix-mpi-strong-commRegion-SortedInput-1048576Size.png)
![1048576 comp](/Radix/MPI%20Pics/strong/radix-mpi-strong-comp_largeRegion-SortedInput-1048576Size.png)

For 4194304:

![4194304 main](/Radix/MPI%20Pics/strong/radix-mpi-strong-mainRegion-RandomInput-4194304Size.png)
![4194304 comm](/Radix/MPI%20Pics/strong/radix-mpi-strong-commRegion-RandomInput-4194304Size.png)
![4194304 comp](/Radix/MPI%20Pics/strong/radix-mpi-strong-comp_largeRegion-RandomInput-4194304Size.png)

For 16777216:

![16777216 main](/Radix/MPI%20Pics/strong/radix-mpi-strong-mainRegion-ReverseInput-67108864Size.png)
![16777216 comm](/Radix/MPI%20Pics/strong/radix-mpi-strong-commRegion-ReverseInput-67108864Size.png)
![16777216 comp](/Radix/MPI%20Pics/strong/radix-mpi-strong-comp_largeRegion-ReverseInput-67108864Size.png)

#### Weak Scaling:

For Random Input:

![Random main](/Radix/MPI%20Pics/weak/radix-mpi-weak-mainRegion-RandomInput.png)
![Random comm](/Radix/MPI%20Pics/weak/radix-mpi-weak-commRegion-RandomInput.png)
![Random comp](/Radix/MPI%20Pics/weak/radix-mpi-weak-comp_largeRegion-RandomInput.png)

For Sorted Input:

![Sorted main](/Radix/MPI%20Pics/weak/radix-mpi-weak-mainRegion-SortedInput.png)
![Sorted comm](/Radix/MPI%20Pics/weak/radix-mpi-weak-commRegion-SortedInput.png)
![Sorted comp](/Radix/MPI%20Pics/weak/radix-mpi-weak-comp_largeRegion-SortedInput.png)

For Reverse Input:

![Reverse main](/Radix/MPI%20Pics/weak/radix-mpi-weak-mainRegion-ReverseInput.png)
![Reverse comm](/Radix/MPI%20Pics/weak/radix-mpi-weak-commRegion-ReverseInput.png)
![Reverse comp](/Radix/MPI%20Pics/weak/radix-mpi-weak-comp_largeRegion-ReverseInput.png)

#### Speedup:

For Random Input:

![Random main](/Radix/MPI%20Pics/speedup/radix-mpi-speedup-mainRegion-RandomInput.png)
![Random comm](/Radix/MPI%20Pics/speedup/radix-mpi-speedup-commRegion-RandomInput.png)
![Random comp](/Radix/MPI%20Pics/speedup/radix-mpi-speedup-comp_largeRegion-RandomInput.png)

For Sorted Input:

![Sorted main](/Radix/MPI%20Pics/speedup/radix-mpi-speedup-mainRegion-SortedInput.png)
![Sorted comm](/Radix/MPI%20Pics/speedup/radix-mpi-speedup-commRegion-SortedInput.png)
![Sorted comp](/Radix/MPI%20Pics/speedup/radix-mpi-speedup-comp_largeRegion-SortedInput.png)

For Reverse Input:

![Reverse main](/Radix/MPI%20Pics/speedup/radix-mpi-speedup-mainRegion-ReverseInput.png)
![Reverse comm](/Radix/MPI%20Pics/speedup/radix-mpi-speedup-commRegion-ReverseInput.png)
![Reverse comp](/Radix/MPI%20Pics/speedup/radix-mpi-speedup-comp_largeRegion-ReverseInput.png)

### CUDA:

#### Strong Scaling:

For 655336:

![65536 main](/Radix/CUDA%20Pics/strong/radix-cuda-strong-mainRegion-SortedInput-65536Size.png)
![65536 comm](/Radix/CUDA%20Pics/strong/radix-cuda-strong-commRegion-SortedInput-65536Size.png)
![65536 comp](/Radix/CUDA%20Pics/strong/radix-cuda-strong-comp_largeRegion-SortedInput-65536Size.png)

For 262144:

![262144 main](/Radix/CUDA%20Pics/strong/radix-cuda-strong-mainRegion-ReverseInput-262144Size.png)
![262144 comm](/Radix/CUDA%20Pics/strong/radix-cuda-strong-commRegion-ReverseInput-262144Size.png)
![262144 comp](/Radix/CUDA%20Pics/strong/radix-cuda-strong-comp_largeRegion-ReverseInput-262144Size.png)

For 1048576:

![1048576 main](/Radix/CUDA%20Pics/strong/radix-cuda-strong-mainRegion-RandomInput-1048576Size.png)
![1048576 comm](/Radix/CUDA%20Pics/strong/radix-cuda-strong-commRegion-RandomInput-1048576Size.png)
![1048576 comp](/Radix/CUDA%20Pics/strong/radix-cuda-strong-comp_largeRegion-RandomInput-1048576Size.png)

For 4194304:

![4194304 main](/Radix/CUDA%20Pics/strong/radix-cuda-strong-mainRegion-SortedInput-4194304Size.png)
![4194304 comm](/Radix/CUDA%20Pics/strong/radix-cuda-strong-commRegion-SortedInput-4194304Size.png)
![4194304 comp](/Radix/CUDA%20Pics/strong/radix-cuda-strong-comp_largeRegion-SortedInput-4194304Size.png)

For 16777216:

![16777216 main](/Radix/CUDA%20Pics/strong/radix-cuda-strong-mainRegion-ReverseInput-16777216Size.png)
![16777216 comm](/Radix/CUDA%20Pics/strong/radix-cuda-strong-commRegion-ReverseInput-16777216Size.png)
![16777216 comp](/Radix/CUDA%20Pics/strong/radix-cuda-strong-comp_largeRegion-ReverseInput-16777216Size.png)

for 67108864:

![67108864 main](/Radix/CUDA%20Pics/strong/radix-cuda-strong-mainRegion-ReverseInput-67108864Size.png)
![67108864 comm](/Radix/CUDA%20Pics/strong/radix-cuda-strong-commRegion-ReverseInput-67108864Size.png)
![67108864 comp](/Radix/CUDA%20Pics/strong/radix-cuda-strong-comp_largeRegion-ReverseInput-67108864Size.png)

#### Weak Scaling:

For Random Input:

![Random main](/Radix/CUDA%20Pics/weak/radix-cuda-weak-mainRegion-RandomInput.png)
![Random comm](/Radix/CUDA%20Pics/weak/radix-cuda-weak-commRegion-RandomInput.png)
![Random comp](/Radix/CUDA%20Pics/weak/radix-cuda-weak-comp_largeRegion-RandomInput.png)

For Sorted Input:

![Sorted main](/Radix/CUDA%20Pics/weak/radix-cuda-weak-mainRegion-SortedInput.png)
![Sorted comm](/Radix/CUDA%20Pics/weak/radix-cuda-weak-commRegion-SortedInput.png)
![Sorted comp](/Radix/CUDA%20Pics/weak/radix-cuda-weak-comp_largeRegion-SortedInput.png)

For Reverse Input:

![Reverse main](/Radix/CUDA%20Pics/weak/radix-cuda-weak-mainRegion-ReverseInput.png)
![Reverse comm](/Radix/CUDA%20Pics/weak/radix-cuda-weak-commRegion-ReverseInput.png)
![Reverse comp](/Radix/CUDA%20Pics/weak/radix-cuda-weak-comp_largeRegion-ReverseInput.png)

#### Speedup:

For Random Input:

![Random main](/Radix/CUDA%20Pics/speedup/radix-cuda-speedup-mainRegion-RandomInput.png)
![Random comm](/Radix/CUDA%20Pics/speedup/radix-cuda-speedup-commRegion-RandomInput.png)
![Random comp](/Radix/CUDA%20Pics/speedup/radix-cuda-speedup-comp_largeRegion-RandomInput.png)

For Sorted Input:

![Sorted main](/Radix/CUDA%20Pics/speedup/radix-cuda-speedup-mainRegion-SortedInput.png)
![Sorted comm](/Radix/CUDA%20Pics/speedup/radix-cuda-speedup-commRegion-SortedInput.png)
![Sorted comp](/Radix/CUDA%20Pics/speedup/radix-cuda-speedup-comp_largeRegion-SortedInput.png)

For Reverse Input:

![Reverse main](/Radix/CUDA%20Pics/speedup/radix-cuda-speedup-mainRegion-ReverseInput.png)
![Reverse comm](/Radix/CUDA%20Pics/speedup/radix-cuda-speedup-commRegion-ReverseInput.png)
![Reverse comp](/Radix/CUDA%20Pics/speedup/radix-cuda-speedup-comp_largeRegion-ReverseInput.png)

### Quick Sort

### MPI:

#### Strong Scaling:

For 655336:

![65536 main](QuickSort/quickgraphs/MPI/strong/quick-mpi-strong-scaling-mainRegion-65536Input.png)
![65536 comm](QuickSort/quickgraphs/MPI/strong/quick-mpi-strong-scaling-commRegion-65536Input.png)

For 262144:

![262144 main](QuickSort/quickgraphs/MPI/strong/quick-mpi-strong-scaling-mainRegion-262144Input.png)
![262144 comm](QuickSort/quickgraphs/MPI/strong/quick-mpi-strong-scaling-commRegion-262144Input.png)

For 1048576:

![1048576 main](QuickSort/quickgraphs/MPI/strong/quick-mpi-strong-scaling-mainRegion-1048576Input.png)
![1048576 comm](QuickSort/quickgraphs/MPI/strong/quick-mpi-strong-scaling-commRegion-1048576Input.png)

For 4194304:

![4194304 main](QuickSort/quickgraphs/MPI/strong/quick-mpi-strong-scaling-mainRegion-4194304Input.png)
![4194304 comm](QuickSort/quickgraphs/MPI/strong/quick-mpi-strong-scaling-commRegion-4194304Input.png)

For 16777216:

![16777216 main](QuickSort/quickgraphs/MPI/strong/quick-mpi-strong-scaling-mainRegion-16777216Input.png)
![16777216 comm](QuickSort/quickgraphs/MPI/strong/quick-mpi-strong-scaling-commRegion-16777216Input.png)

#### Weak Scaling:

For Random Input:

![Random main](QuickSort/quickgraphs/MPI/weak/quick-mpi-weak-scaling-mainRegion-RandomInput.png)
![Random comm](QuickSort/quickgraphs/MPI/weak/quick-mpi-weak-scaling-commRegion-RandomInput.png)

For Sorted Input:

![Sorted main](QuickSort/quickgraphs/MPI/weak/quick-mpi-weak-scaling-mainRegion-SortedInput.png)
![Sorted comm](QuickSort/quickgraphs/MPI/weak/quick-mpi-weak-scaling-commRegion-SortedInput.png)

For Reverse Input:

![Reverse main](QuickSort/quickgraphs/MPI/weak/quick-mpi-weak-scaling-mainRegion-ReverseInput.png)
![Reverse comm](QuickSort/quickgraphs/MPI/weak/quick-mpi-weak-scaling-commRegion-ReverseInput.png)

#### Speedup:

For Random Input:

![Random main](QuickSort/quickgraphs/MPI/speedup/quick-mpi-speedup-mainRegion-RandomInput.png)
![Random comm](QuickSort/quickgraphs/MPI/speedup/quick-mpi-speedup-commRegion-RandomInput.png)

For Sorted Input:

![Sorted main](QuickSort/quickgraphs/MPI/speedup/quick-mpi-speedup-mainRegion-SortedInput.png)
![Sorted comm](QuickSort/quickgraphs/MPI/speedup/quick-mpi-speedup-commRegion-SortedInput.png)

For Reverse Input:

![Reverse main](QuickSort/quickgraphs/MPI/speedup/quick-mpi-speedup-mainRegion-ReverseInput.png)
![Reverse comm](QuickSort/quickgraphs/MPI/speedup/quick-mpi-speedup-commRegion-ReverseInput.png)

### CUDA:

#### Strong Scaling:

For 4194304:

![4194304 main](QuickSort/quickgraphs/CUDA/strong/quick-CUDA-strong-scaling-mainRegion-4194304Input.png)
![4194304 comm](QuickSort/quickgraphs/CUDA/strong/quick-CUDA-strong-scaling-commRegion-4194304Input.png)
![4194304 comp](QuickSort/quickgraphs/CUDA/strong/quick-CUDA-strong-scaling-comp_largeRegion-4194304Input.png)

#### Weak Scaling:

For Random Input:

![Random main](QuickSort/quickgraphs/CUDA/weak/quick-CUDA-weak-scaling-mainRegion-RandomInput.png)
![Random comm](QuickSort/quickgraphs/CUDA/weak/quick-CUDA-weak-scaling-commRegion-RandomInput.png)
![Random comp](QuickSort/quickgraphs/CUDA/weak/quick-CUDA-weak-scaling-comp_largeRegion-RandomInput.png)

#### Speedup:

For Random Input:

![Random main](QuickSort/quickgraphs/CUDA/speedup/quick-CUDA-speedup-mainRegion-RandomInput.png)
![Random comm](QuickSort/quickgraphs/CUDA/speedup/quick-CUDA-speedup-commRegion-RandomInput.png)
![Random comp](QuickSort/quickgraphs/CUDA/speedup/quick-CUDA-speedup-comp_largeRegion-RandomInput.png)

## Bitonic Sort

### MPI:

#### Strong Scaling:

For 655336:

![65536 main](/Bitonic%20Sort/MPI/plots_mpi/strong/bitonic-mpi-strong-mainRegion-RandomInput-2^16_numvals.png)
![65536 comm](/Bitonic%20Sort/MPI/plots_mpi/strong/bitonic-mpi-strong-commRegion-RandomInput-2^16_numvals.png)
![65536 comp](/Bitonic%20Sort/MPI/plots_mpi/strong/bitonic-mpi-strong-comp_largeRegion-RandomInput-2^16_numvals.png)

For 262144:

![262144 main](/Bitonic%20Sort/MPI/plots_mpi/strong/bitonic-mpi-strong-mainRegion-SortedInput-2^18_numvals.png)
![262144 comm](/Bitonic%20Sort/MPI/plots_mpi/strong/bitonic-mpi-strong-commRegion-SortedInput-2^18_numvals.png)
![262144 comp](/Bitonic%20Sort/MPI/plots_mpi/strong/bitonic-mpi-strong-comp_largeRegion-SortedInput-2^18_numvals.png)

For 1048576:

![1048576 main](/Bitonic%20Sort/MPI/plots_mpi/strong/bitonic-mpi-strong-mainRegion-ReverseSortedInput-2^20_numvals.png)
![1048576 comm](/Bitonic%20Sort/MPI/plots_mpi/strong/bitonic-mpi-strong-commRegion-ReverseSortedInput-2^20_numvals.png)
![1048576 comp](/Bitonic%20Sort/MPI/plots_mpi/strong/bitonic-mpi-strong-comp_largeRegion-ReverseSortedInput-2^20_numvals.png)

For 4194304:

![4194304 main](/Bitonic%20Sort/MPI/plots_mpi/strong/bitonic-mpi-strong-mainRegion-RandomInput-2^22_numvals.png)
![4194304 comm](/Bitonic%20Sort/MPI/plots_mpi/strong/bitonic-mpi-strong-commRegion-RandomInput-2^22_numvals.png)
![4194304 comp](/Bitonic%20Sort/MPI/plots_mpi/strong/bitonic-mpi-strong-comp_largeRegion-RandomInput-2^22_numvals.png)

For 16777216:

![16777216 main](/Bitonic%20Sort/MPI/plots_mpi/strong/bitonic-mpi-strong-mainRegion-RandomInput-2^26_numvals.png)
![16777216 comm](/Bitonic%20Sort/MPI/plots_mpi/strong/bitonic-mpi-strong-commRegion-RandomInput-2^26_numvals.png)
![16777216 comp](/Bitonic%20Sort/MPI/plots_mpi/strong/bitonic-mpi-strong-comp_largeRegion-RandomInput-2^26_numvals.png)

#### Weak Scaling:

For Random Input:

![Random main](/Bitonic%20Sort/MPI/plots_mpi/weak/bitonic-mpi-weak-mainRegion-RandomInput.png)
![Random comm](/Bitonic%20Sort/MPI/plots_mpi/weak/bitonic-mpi-weak-commRegion-RandomInput.png)
![Random comp](/Bitonic%20Sort/MPI/plots_mpi/weak/bitonic-mpi-weak-comp_largeRegion-RandomInput.png)

For Sorted Input:

![Sorted main](/Bitonic%20Sort/MPI/plots_mpi/weak/bitonic-mpi-weak-mainRegion-SortedInput.png)
![Sorted comm](/Bitonic%20Sort/MPI/plots_mpi/weak/bitonic-mpi-weak-commRegion-SortedInput.png)
![Sorted comp](/Bitonic%20Sort/MPI/plots_mpi/weak/bitonic-mpi-weak-comp_largeRegion-SortedInput.png)

For Reverse Input:

![Reverse main](/Bitonic%20Sort/MPI/plots_mpi/weak/bitonic-mpi-weak-mainRegion-ReverseSortedInput.png)
![Reverse comm](/Bitonic%20Sort/MPI/plots_mpi/weak/bitonic-mpi-weak-commRegion-ReverseSortedInput.png)
![Reverse comp](/Bitonic%20Sort/MPI/plots_mpi/weak/bitonic-mpi-weak-comp_largeRegion-ReverseSortedInput.png)

#### Speedup:

For Random Input:

![Random main](/Bitonic%20Sort/MPI/plots_mpi/speedup/bitonic-mpi-speedup-mainRegion-RandomInput.png)
![Random comm](/Bitonic%20Sort/MPI/plots_mpi/speedup/bitonic-mpi-speedup-commRegion-RandomInput.png)
![Random comp](/Bitonic%20Sort/MPI/plots_mpi/speedup/bitonic-mpi-speedup-comp_largeRegion-RandomInput.png)

For Sorted Input:

![Sorted main](/Bitonic%20Sort/MPI/plots_mpi/speedup/bitonic-mpi-speedup-mainRegion-SortedInput.png)
![Sorted comm](/Bitonic%20Sort/MPI/plots_mpi/speedup/bitonic-mpi-speedup-commRegion-SortedInput.png)
![Sorted comp](/Bitonic%20Sort/MPI/plots_mpi/speedup/bitonic-mpi-speedup-comp_largeRegion-SortedInput.png)

For Reverse Input:

![Reverse main](/Bitonic%20Sort/MPI/plots_mpi/speedup/bitonic-mpi-speedup-mainRegion-ReverseSortedInput.png)
![Reverse comm](/Bitonic%20Sort/MPI/plots_mpi/speedup/bitonic-mpi-speedup-commRegion-ReverseSortedInput.png)
![Reverse comp](/Bitonic%20Sort/MPI/plots_mpi/speedup/bitonic-mpi-speedup-comp_largeRegion-ReverseSortedInput.png)

### CUDA:

#### Strong Scaling:

For 655336:

![65536 main](/Bitonic%20Sort/CUDA/plots_cuda/strong/bitonic-cuda-strong-mainRegion-SortedInput-2^16_numvals.png)
![65536 comm](/Bitonic%20Sort/CUDA/plots_cuda/strong/bitonic-cuda-strong-commRegion-SortedInput-2^16_numvals.png)
![65536 comp](/Bitonic%20Sort/CUDA/plots_cuda/strong/bitonic-cuda-strong-comp_largeRegion-SortedInput-2^16_numvals.png)

For 262144:

![262144 main](/Bitonic%20Sort/CUDA/plots_cuda/strong/bitonic-cuda-strong-mainRegion-ReverseSortedInput-2^18_numvals.png)
![262144 comm](/Bitonic%20Sort/CUDA/plots_cuda/strong/bitonic-cuda-strong-commRegion-ReverseSortedInput-2^18_numvals.png)
![262144 comp](/Bitonic%20Sort/CUDA/plots_cuda/strong/bitonic-cuda-strong-comp_largeRegion-ReverseSortedInput-2^18_numvals.png)

For 1048576:

![1048576 main](/Bitonic%20Sort/CUDA/plots_cuda/strong/bitonic-cuda-strong-mainRegion-RandomInput-2^20_numvals.png)
![1048576 comm](/Bitonic%20Sort/CUDA/plots_cuda/strong/bitonic-cuda-strong-commRegion-RandomInput-2^20_numvals.png)
![1048576 comp](/Bitonic%20Sort/CUDA/plots_cuda/strong/bitonic-cuda-strong-comp_largeRegion-RandomInput-2^20_numvals.png)

For 4194304:

![4194304 main](/Bitonic%20Sort/CUDA/plots_cuda/strong/bitonic-cuda-strong-mainRegion-SortedInput-2^22_numvals.png)
![4194304 comm](/Bitonic%20Sort/CUDA/plots_cuda/strong/bitonic-cuda-strong-commRegion-SortedInput-2^22_numvals.png)
![4194304 comp](/Bitonic%20Sort/CUDA/plots_cuda/strong/bitonic-cuda-strong-comp_largeRegion-SortedInput-2^22_numvals.png)

For 16777216:

![16777216 main](/Bitonic%20Sort/CUDA/plots_cuda/strong/bitonic-cuda-strong-mainRegion-ReverseSortedInput-2^24_numvals.png)
![16777216 comm](/Bitonic%20Sort/CUDA/plots_cuda/strong/bitonic-cuda-strong-commRegion-ReverseSortedInput-2^24_numvals.png)
![16777216 comp](/Bitonic%20Sort/CUDA/plots_cuda/strong/bitonic-cuda-strong-comp_largeRegion-ReverseSortedInput-2^24_numvals.png)

for 67108864:

![67108864 main](/Bitonic%20Sort/CUDA/plots_cuda/strong/bitonic-cuda-strong-mainRegion-ReverseSortedInput-2^26_numvals.png)
![67108864 comm](/Bitonic%20Sort/CUDA/plots_cuda/strong/bitonic-cuda-strong-commRegion-ReverseSortedInput-2^26_numvals.png)
![67108864 comp](/Bitonic%20Sort/CUDA/plots_cuda/strong/bitonic-cuda-strong-comp_largeRegion-ReverseSortedInput-2^26_numvals.png)

#### Weak Scaling:

For Random Input:

![Random main](/Bitonic%20Sort/CUDA/plots_cuda/weak/bitonic-cuda-weak-mainRegion-RandomInput.png)
![Random comm](/Bitonic%20Sort/CUDA/plots_cuda/weak/bitonic-cuda-weak-commRegion-RandomInput.png)
![Random comp](/Bitonic%20Sort/CUDA/plots_cuda/weak/bitonic-cuda-weak-comp_largeRegion-RandomInput.png)

For Sorted Input:

![Sorted main](/Bitonic%20Sort/CUDA/plots_cuda/weak/bitonic-cuda-weak-mainRegion-SortedInput.png)
![Sorted comm](/Bitonic%20Sort/CUDA/plots_cuda/weak/bitonic-cuda-weak-commRegion-SortedInput.png)
![Sorted comp](/Bitonic%20Sort/CUDA/plots_cuda/weak/bitonic-cuda-weak-comp_largeRegion-SortedInput.png)

For Reverse Input:

![Reverse main](/Bitonic%20Sort/CUDA/plots_cuda/weak/bitonic-cuda-weak-mainRegion-ReverseSortedInput.png)
![Reverse comm](/Bitonic%20Sort/CUDA/plots_cuda/weak/bitonic-cuda-weak-commRegion-ReverseSortedInput.png)
![Reverse comp](/Bitonic%20Sort/CUDA/plots_cuda/weak/bitonic-cuda-weak-comp_largeRegion-ReverseSortedInput.png)

#### Speedup:

For Random Input:

![Random main](/Bitonic%20Sort/CUDA/plots_cuda/speedup/bitonic-cuda-speedup-mainRegion-RandomInput.png)
![Random comm](/Bitonic%20Sort/CUDA/plots_cuda/speedup/bitonic-cuda-speedup-commRegion-RandomInput.png)
![Random comp](/Bitonic%20Sort/CUDA/plots_cuda/speedup/bitonic-cuda-speedup-comp_largeRegion-RandomInput.png)

For Sorted Input:

![Sorted main](/Bitonic%20Sort/CUDA/plots_cuda/speedup/bitonic-cuda-speedup-mainRegion-SortedInput.png)
![Sorted comm](/Bitonic%20Sort/CUDA/plots_cuda/speedup/bitonic-cuda-speedup-commRegion-SortedInput.png)
![Sorted comp](/Bitonic%20Sort/CUDA/plots_cuda/speedup/bitonic-cuda-speedup-comp_largeRegion-SortedInput.png)

For Reverse Input:

![Reverse main](/Bitonic%20Sort/CUDA/plots_cuda/speedup/bitonic-cuda-speedup-mainRegion-ReverseSortedInput.png)
![Reverse comm](/Bitonic%20Sort/CUDA/plots_cuda/speedup/bitonic-cuda-speedup-commRegion-ReverseSortedInput.png)
![Reverse comp](/Bitonic%20Sort/CUDA/plots_cuda/speedup/bitonic-cuda-speedup-comp_largeRegion-ReverseSortedInput.png)

## Comparison

### MPI:

![MPI main](/comparison/mpi.png)

### CUDA:

![CUDA main](/comparison/cuda.png)

## 4a and 4b. Vary the following parameters

When running our tests, these were our parameters for all algorithms.
For MPI, we ran the following script:

```
for num_proc in 2 4 8 16 32 64; do
    for size in 65536 262144 1048576 4194304 16777216 67108864; do
        sbatch mpi.grace_job "$size" "$num_proc"
        echo "ran for process $num_proc at size $size"
    done
done
```

For CUDA, we ran the following script:

```
for num_proc in 64 128 256 512; do
    for size in 65536 262144 1048576 4194304 16777216 67108864; do
        sbatch bitonic.grace_job "$size" "$num_proc"
    done
done
```

## 4c.

![Image Alt Text](https://media.discordapp.net/attachments/1166861153872384011/1174867022673362944/Screenshot_2023-11-16_at_6.21.59_PM.png?ex=6569272e&is=6556b22e&hm=c7bb920115f750f0bea058ae3fc8eaa4df1d0121b40801ae2ad56ba371886a20&=&width=1832&height=652)

Above is an example our performance metrics. This is similar for all implementations with the same structure as we used the average time for each function.

## Analysis

### Merge Sort

For the MPI implementation of merge sort, when it comes to the actual computation part of sorting of the code, the time goes down exponentially as you increase the number of processes. This makes sense as that increases the parallelism in the merge sort algorithm and so each of the processes are able to get a part of the data and work on it independently and with more of the processes, more data is getting worked on and it causes the execution of the data to be faster. This can be seen in the strong scaling, but weak scaling really shows the trend as it compares the number of processes and the number of inputs. On one number of inputs, as you increase the amount of threads, the time it takes to compute drops exponentially. In the speed up, it increases as the number of processes increases, further supposing the fact that number of processes increases the speed of sorting the array. The graph for the main and communication varies. In general, the main should also exponentially decrease like the part previously mentioned and the communication should be relatively the same. However that is not necessarily the case. The potential reason for having the communication having this is because there is overhead that is here for the middle section from the synchronization. For the one in main, there is the problem of overhead as well as there were cases where there was an error that was given for one of the cases. Although it ran and had the right output, because of the initial error given and then running, that could have increased the time. For the speed up, both of them spiders out for each of the input but then all of them level out.

For CUDA, the strong scaling was all over the place, the weak scaling was even all the way through and it was the same for the speed up. Unfortunately, this is not what we wanted as the graphs for strong scaling, weakening scaling, and speed up should be like the ones from MPI, where the actual sorting part should decrease time as the number of threads increase and same with weak scaling, and the speedup should increase. Some of the reason behind what could be the problem is that the algorithm implemented is not totally paralyzed and therefore, it creates a problem as it will make the time decrease as the number of threads increase.

### Quick Sort

The sequential implementation for quick sort works by taking in the input array and splitting it up into smaller buckets. These buckets are then sorted around a pivot point and merged back up until the whole array is sorted.

Quick sort is a sequential algorithm but it is still possible to make this algorithm parallel. The way this was done was similar for both the CUDA implementation and the MPI implementation. For the CUDA, the input array is partitioned into different portions based on their index. These different portions are assigned a thread that will sort the portion and combine it back with the main input array. The algorithm dynamically divides the input array into smaller subarrays and coordinates the sorting process among threads. The way quick sort MPI is parallelized is similar to how CUDA parallelized. The input array is partitioned into different portions and these portions are assigned different processes that will run and sort the portion. Once the portion is sorted, it is combined back with the main input array and this cycle repeats until the array is sorted. This is parallelized because all these processes run concurrently and sort each portion at the same time.

When graphing the implemented code, we realized there were some implementation errors as the graphs didn’t really look parallel. For CUDA, it was hanging when running the jobs and this could be because of a synchronization error. Another reason could be because of an incorrect Kernel Configuration. One of these reasons causes some of the grace jobs to take a long time or run infinitely, which is why the graphs look like straight lines since there isn’t much data to work with. For MPI, we also ran into some errors as the plots don’t look like one for an accurate parallel algorithm. We suspect that there may be some communication overhead occuring, which is why overall there was an increase in time as processes increased.

Let’s start by looking at the MPI graphs. When analyzing the strong scaling graphs for quick sort, we see a general trend of an increase in time as the processes are added. This is the case for all input sizes. It seems overall that random takes the most time out of the three input arrays we had in general. Random is closely followed by sorted for the most time and it seems that in most cases, reverse takes the least amount of time. For the weak scaling graphs, we see this same trend where more processes results in more time taken. We also see that the arrays with the more elements have the highest average time and are always taking more time to sort than ones with less elements. This is the case for random, sorted, and reverse inputs. This pattern is reflected in the speedup graphs as we see an overall decrease in the ratio for the speedup as the number of processes increases. We also see that the arrays with more elements end with more speed up than the ones with less elements.

Now let's take a look at the CUDA graphs. Starting with the strong scaling graphs, we see an increase in time for the algorithm as the number of processes increases. This is the case for all the input types. For the weak scaling, we see the same pattern of an increase in time as the number of processes increases. This is for the random input arrays. For our speedup graphs for quick sort, we see that the ratio for speedup decreases with more processes for random input arrays. As mentioned earlier, there was an implementation error for this code, which is why it doesn’t accurately represent a quick sort CUDA algorithm.

### Bitonic Sort

Bitonic sort is a sorting algorithm that works by converting random sequences of numbers into a bitonic sequence. A bitonic sequence is where the first and last element are smaller than the one in the middle and similarly the second and third of each end. This leaves the middle number as the largest number in the bitonic sequence. Then, this bitonic sequence is split up in a similar fashion to merge sort. Each sequence split up from the bitonic sequence is a bitonic sequence itself. These sequences become smaller and smaller and become sorted at the end. Then, all the sequences are joined together which results in the sorted sequence. This is a good algorithm to parallelize due to the predictable nature of the bitonic sequences in comparison to other sorting algorithms.

For the MPI implementation of this algorithm, it seems that the trends are not as expected. For the 3 regions we are observing, which are main, comm, and comp_large, the time per rank goes up as the number of processes increases. For the main region and comm region, there seems to be some trends with the time decreasing again past 32 processes. However, this may not be correct. Overall, as the values go up the time goes up even more and it can also be seen the communication plots are going up as the number of values for input increases. This shows that there is probably an issue with the communication between processes for this implementation. This is potentially the main reason for the unusual trends in the plots for the MPI implementation of Bitonic Sort. Similarly, for the weak scaling plots, the trend for 2^26 elements goes out of the scope of the plot which is incorrect. There are much smaller differences for the smaller number of elements. Regardless of the input type, all the trends are the same for the weak scaling plots. The speedup plots show a downward trend as the number of elements increases for all the input types. The lines somewhat overlap each other and as the number of processes increases, the speedup approaches one. I have a feeling the issues with this implementation of Bitonic Sort on MPI is a result of the communication between processes which is increasing the computation time greatly and causing these incorrect/unexpected trends in the plots.

For the CUDA implementation of this algorithm, it does perform as expected and the trends are correct. It can be seen that as the number of threads increases, the time decreases and starts higher for a lower number of threads. Across the strong scaling plots, for 2^16 values as input, the comp_large region shows a somewhat unusual trend and this is due to the relatively small problem size which results in communication across threads bottlenecking the performance and slowing down the sorting. This trend disappears more as the number of values go up. It can still be somewhat seen that as the number of threads increases for 2^18 number of values, the time consistently decreases for the main and comp_large regions. This trend continues all the way to 2^26 values. It seems the optimal number of threads for this set of input values is around 256 threads as seen in the last plot. That number of threads produces the shortest time. All the comm region plots are all over the place and produce very random plots. This is potentially because it is run on a GPU with many number of threads which causes a lot of communication and this is also somewhat random as well. The plots are unpredictable for communication but it can be seen the time does go up according to the axis which shows it is working correctly. The speed up plots also show an upward trend with the max speedup of around 4.25 with the number of values as 2^22.

### Comparison

When it comes to the comparison to all the sorting, radix sort came out as the fastest algorithm, with merge sort being second, and bitonic being third. In the graph, you can not see quick sort as the time took so long that it was really hard to see the other 3 sorting algorithms, therefore, we had to change the scale so we can see the other ones. The reason why is that quick sort is not parallelized correctly and runs more sequentially. Radix sorting being the fastest can potentially be because of the fact that the radix sort has a linear time complexity as well as the fact that it does not need to be broken down into subarrays. The reason why merge is faster than bitonic is that merge is O(nlogn) while for bitonic sort, it is O(log^2n) with a large number of elements.
