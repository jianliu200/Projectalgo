# CSCE 435 Group project

## 0. Group number: 

## 1. Group members:
1. Jiangyuan Liu
2. Akhil Mathew
3. Jacob Thomas
4. Ashwin Kundeti

The way our team is communicating is by using Discord and iMessages
---

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

3. Quicksort MPI
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

http://users.atw.hu/parallelcomp/ch09lev1sec4.html

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
Implement your proposed algorithms, and test them starting on a small scale.
Instrument your code, and turn in at least one Caliper file per algorithm;
if you have implemented an MPI and a CUDA version of your algorithm,
turn in a Caliper file for each.





<!--
For example:
- Algorithm 1a (MPI + CUDA)
- Algorithm 1b (MPI on each core)
- Algorithm 2a (MPI + CUDA)
- Algorithm 2b (MPI on each core)

### 2b. Pseudocode for each parallel algorithm
- For MPI programs, include MPI calls you will use to coordinate between processes
- For CUDA programs, indicate which computation will be performed in a CUDA kernel,
  and where you will transfer data to/from GPU

### 2c. Evaluation plan - what and how will you measure and compare
- Input sizes, Input types
- Strong scaling (same problem size, increase number of processors/nodes)
- Weak scaling (increase problem size, increase number of processors)
- Number of threads in a block on the GPU 


## 3. Project implementation
Implement your proposed algorithms, and test them starting on a small scale.
Instrument your code, and turn in at least one Caliper file per algorithm;
if you have implemented an MPI and a CUDA version of your algorithm,
turn in a Caliper file for each.

### 3a. Caliper instrumentation
Please use the caliper build `/scratch/group/csce435-f23/Caliper/caliper/share/cmake/caliper` 
(same as lab1 build.sh) to collect caliper files for each experiment you run.

Your Caliper regions should resemble the following calltree
(use `Thicket.tree()` to see the calltree collected on your runs):
```
main
|_ data_init
|_ comm
|    |_ MPI_Barrier
|    |_ comm_small  // When you broadcast just a few elements, such as splitters in Sample sort
|    |   |_ MPI_Bcast
|    |   |_ MPI_Send
|    |   |_ cudaMemcpy
|    |_ comm_large  // When you send all of the data the process has
|        |_ MPI_Send
|        |_ MPI_Bcast
|        |_ cudaMemcpy
|_ comp
|    |_ comp_small  // When you perform the computation on a small number of elements, such as sorting the splitters in Sample sort
|    |_ comp_large  // When you perform the computation on all of the data the process has, such as sorting all local elements
|_ correctness_check
```

Required code regions:
- `main` - top-level main function.
    - `data_init` - the function where input data is generated or read in from file.
    - `correctness_check` - function for checking the correctness of the algorithm output (e.g., checking if the resulting data is sorted).
    - `comm` - All communication-related functions in your algorithm should be nested under the `comm` region.
      - Inside the `comm` region, you should create regions to indicate how much data you are communicating (i.e., `comm_small` if you are sending or broadcasting a few values, `comm_large` if you are sending all of your local values).
      - Notice that auxillary functions like MPI_init are not under here.
    - `comp` - All computation functions within your algorithm should be nested under the `comp` region.
      - Inside the `comp` region, you should create regions to indicate how much data you are computing on (i.e., `comp_small` if you are sorting a few values like the splitters, `comp_large` if you are sorting values in the array).
      - Notice that auxillary functions like data_init are not under here.

All functions will be called from `main` and most will be grouped under either `comm` or `comp` regions, representing communication and computation, respectively. You should be timing as many significant functions in your code as possible. **Do not** time print statements or other insignificant operations that may skew the performance measurements.

**Nesting Code Regions** - all computation code regions should be nested in the "comp" parent code region as following:
```
CALI_MARK_BEGIN("comp");
CALI_MARK_BEGIN("comp_large");
mergesort();
CALI_MARK_END("comp_large");
CALI_MARK_END("comp");
```

**Looped GPU kernels** - to time GPU kernels in a loop:
```
### Bitonic sort example.
int count = 1;
CALI_MARK_BEGIN("comp");
CALI_MARK_BEGIN("comp_large");
int j, k;
/* Major step */
for (k = 2; k <= NUM_VALS; k <<= 1) {
    /* Minor step */
    for (j=k>>1; j>0; j=j>>1) {
        bitonic_sort_step<<<blocks, threads>>>(dev_values, j, k);
        count++;
    }
}
CALI_MARK_END("comp_large");
CALI_MARK_END("comp");
```

**Calltree Examples**:

```
# Bitonic sort tree - CUDA looped kernel
1.000 main
├─ 1.000 comm
│  └─ 1.000 comm_large
│     └─ 1.000 cudaMemcpy
├─ 1.000 comp
│  └─ 1.000 comp_large
└─ 1.000 data_init
```

```
# Matrix multiplication example - MPI
1.000 main
├─ 1.000 comm
│  ├─ 1.000 MPI_Barrier
│  ├─ 1.000 comm_large
│  │  ├─ 1.000 MPI_Recv
│  │  └─ 1.000 MPI_Send
│  └─ 1.000 comm_small
│     ├─ 1.000 MPI_Recv
│     └─ 1.000 MPI_Send
├─ 1.000 comp
│  └─ 1.000 comp_large
└─ 1.000 data_init
```

```
# Mergesort - MPI
1.000 main
├─ 1.000 comm
│  ├─ 1.000 MPI_Barrier
│  └─ 1.000 comm_large
│     ├─ 1.000 MPI_Gather
│     └─ 1.000 MPI_Scatter
├─ 1.000 comp
│  └─ 1.000 comp_large
└─ 1.000 data_init
```

#### 3b. Collect Metadata

Have the following `adiak` code in your programs to collect metadata:
```
adiak::init(NULL);
adiak::launchdate();    // launch date of the job
adiak::libraries();     // Libraries used
adiak::cmdline();       // Command line used to launch the job
adiak::clustername();   // Name of the cluster
adiak::value("Algorithm", algorithm); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
adiak::value("ProgrammingModel", programmingModel); // e.g., "MPI", "CUDA", "MPIwithCUDA"
adiak::value("Datatype", datatype); // The datatype of input elements (e.g., double, int, float)
adiak::value("SizeOfDatatype", sizeOfDatatype); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
adiak::value("InputSize", inputSize); // The number of elements in input dataset (1000)
adiak::value("InputType", inputType); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
adiak::value("num_procs", num_procs); // The number of processors (MPI ranks)
adiak::value("num_threads", num_threads); // The number of CUDA or OpenMP threads
adiak::value("num_blocks", num_blocks); // The number of CUDA blocks 
adiak::value("group_num", group_number); // The number of your group (integer, e.g., 1, 10)
adiak::value("implementation_source", implementation_source) // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").
```

They will show up in the `Thicket.metadata` if the caliper file is read into Thicket.

**See the `Builds/` directory to find the correct Caliper configurations to get the above metrics for CUDA, MPI, or OpenMP programs.** They will show up in the `Thicket.dataframe` when the Caliper file is read into Thicket.
