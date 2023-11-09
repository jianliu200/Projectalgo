
/**
 * mergesort.cu
 * a one-file c++ / cuda program for performing mergesort on the GPU
 * While the program execution is fairly slow, most of its runnning time
 *  is spent allocating memory on the GPU.
 * For a more complex program that performs many calculations,
 *  running on the GPU may provide a significant boost in performance
 */

// helper for main()

// source: https://github.com/kevin-albert/cuda-mergesort/blob/master/mergesort.cu
// author: kevin-albert
// I am using this source code for CUDA implementation of merge sort. I have added caliper and adiak annotations to the code.


#include <iostream>
#include <helper_cuda.h>
#include <sys/time.h>

long readList(long**);

// data[], size, threads, blocks, 
void mergesort(long*, long, dim3, dim3);
// A[]. B[], size, width, slices, nThreads
__global__ void gpu_mergesort(long*, long*, long, long, long, dim3*, dim3*);
__device__ void gpu_bottomUpMerge(long*, long*, long, long, long);

// const char* main_loop = "main loop";
// const char* comm =  "comm";
// const char* comm_large = "comm_large";
// const char* comp = "comp";
// const char* barrier = "barrier";
// const char* scatter = "scatter";
// const char* gather = "gather";
// const char* data_init = "data_init";
// const char* correctness = "correctness";

// profiling
int tm();

#define min(a, b) (a < b ? a : b)
const char* main_loop = "main loop";
const char* comm =  "comm";
const char* comm_large = "comm_large";
const char* comp = "comp";
const char* comp_large = "comp_large";
const char* data_init = "data_init";
const char* correctness = "correctness";

bool verbose = true;
int main(int argc, char** argv) {
    CALI_CXX_MARK_FUNCTION;
    cali::ConfigManager mgr;
	mgr.start();
    CALI_MARK_BEGIN(main_loop);
    

    //
    // Parse argv
    //
    // tm();
    // for (int i = 1; i < argc; i++) {
    //     if (argv[i][0] == '-' && argv[i][1] && !argv[i][2]) {
    //         char arg = argv[i][1];
    //         unsigned int* toSet = 0;
    //         switch(arg) {
    //             case 'x':
    //                 toSet = &threadsPerBlock.x;
    //                 break;
    //             case 'y':
    //                 toSet = &threadsPerBlock.y;
    //                 break;
    //             case 'z':
    //                 toSet = &threadsPerBlock.z;
    //                 break;
    //             case 'X':
    //                 toSet = &blocksPerGrid.x;
    //                 break;
    //             case 'Y':
    //                 toSet = &blocksPerGrid.y;
    //                 break;
    //             case 'Z':
    //                 toSet = &blocksPerGrid.z;
    //                 break;
    //             case 'v':
    //                 verbose = true;
    //                 break;
    //             default:
    //                 std::cout << "unknown argument: " << arg << '\n';
    //                 printHelp(argv[0]);
    //                 return -1;
    //         }

    //         if (toSet) {
    //             i++;
    //             *toSet = (unsigned int) strtol(argv[i], 0, 10);
    //         }
    //     }
    //     else {
    //         if (argv[i][0] == '?' && !argv[i][1])
    //             std::cout << "help:\n";
    //         else
    //             std::cout << "invalid argument: " << argv[i] << '\n';
    //         printHelp(argv[0]);
    //         return -1;
    //     }
    // }

    // if (verbose) {
    //     std::cout << "parse argv " << tm() << " microseconds\n";
    //     std::cout << "\nthreadsPerBlock:"
    //               << "\n  x: " << threadsPerBlock.x
    //               << "\n  y: " << threadsPerBlock.y
    //               << "\n  z: " << threadsPerBlock.z
    //               << "\n\nblocksPerGrid:"
    //               << "\n  x:" << blocksPerGrid.x
    //               << "\n  y:" << blocksPerGrid.y
    //               << "\n  z:" << blocksPerGrid.z
    //               << "\n\n total threads: " 
    //               << threadsPerBlock.x * threadsPerBlock.y * threadsPerBlock.z *
    //                  blocksPerGrid.x * blocksPerGrid.y * blocksPerGrid.z
    //               << "\n\n";
    // }

    //
    // Read numbers from stdin
    //
    long* data;
    long size = atoi(argv[2]);
    CALI_MARK_BEGIN(data_init);
    for(int i = 0; i < size; i++)
    {
        data[i] = rand() % size;
    }
    CALI_MARK_END(data_init);
    if (!size) return -1;
    dim3 threadsPerBlock;
    dim3 blocksPerGrid;

    threadsPerBlock.x = atoi(argv[1]);
    threadsPerBlock.y = 1;
    threadsPerBlock.z = 1;

    blocksPerGrid.x = atoi(argv[1])/size;
    blocksPerGrid.y = 1;
    blocksPerGrid.z = 1;

    // if (verbose)
    //     std::cout << "sorting " << size << " numbers\n\n";

    // merge-sort the data
    
    
    mergesort(data, size, threadsPerBlock, blocksPerGrid);
    
    
    tm();

    //
    // Print out the list
    //
    for (int i = 0; i < size; i++) {
        std::cout << data[i] << '\n';
    } 
    
    CALI_MARK_BEGIN(correctness);
    bool sorting = std::is_sorted(data, data + size);
    CALI_MARK_END(correctness);
    if(sorting == true)
    {
        std::cout << "The array is sorted\n";
    }
    else
    {
        std::cout << "The array is not sorted\n";
    }

    if (verbose) {
        std::cout << "print list to stdout: " << tm() << " microseconds\n";
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
    adiak::value("Datatype", "long"); // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", "4 bytes"); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", size); // The number of elements in input dataset (1000)
    adiak::value("InputType", "Random"); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    //adiak::value("num_procs", world_size); // The number of processors (MPI ranks)
    adiak::value("num_threads", threadsPerBlock.x); // The number of CUDA or OpenMP threads
    adiak::value("num_blocks", blocksPerGrid.x); // The number of CUDA blocks 
    adiak::value("group_num", "7"); // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "Online: https://github.com/kevin-albert/cuda-mergesort/blob/master/mergesort.cu"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").

}

void mergesort(long* data, long size, dim3 threadsPerBlock, dim3 blocksPerGrid) {

    //
    // Allocate two arrays on the GPU
    // we switch back and forth between them during the sort
    //
    long* D_data;
    long* D_swp;
    dim3* D_threads;
    dim3* D_blocks;
    
    // Actually allocate the two arrays
    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    tm();
    checkCudaErrors(cudaMalloc((void**) &D_data, size * sizeof(long)));
    checkCudaErrors(cudaMalloc((void**) &D_swp, size * sizeof(long)));
    if (verbose)
        std::cout << "cudaMalloc device lists: " << tm() << " microseconds\n";

    // Copy from our input list into the first array
    checkCudaErrors(cudaMemcpy(D_data, data, size * sizeof(long), cudaMemcpyHostToDevice));
    if (verbose) 
        std::cout << "cudaMemcpy list to device: " << tm() << " microseconds\n";
 
    //
    // Copy the thread / block info to the GPU as well
    //
    checkCudaErrors(cudaMalloc((void**) &D_threads, sizeof(dim3)));
    checkCudaErrors(cudaMalloc((void**) &D_blocks, sizeof(dim3)));

    if (verbose)
        std::cout << "cudaMalloc device thread data: " << tm() << " microseconds\n";
    checkCudaErrors(cudaMemcpy(D_threads, &threadsPerBlock, sizeof(dim3), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(D_blocks, &blocksPerGrid, sizeof(dim3), cudaMemcpyHostToDevice));

    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);
    
    if (verbose)
        std::cout << "cudaMemcpy thread data to device: " << tm() << " microseconds\n";

    long* A = D_data;
    long* B = D_swp;

    long nThreads = threadsPerBlock.x * threadsPerBlock.y * threadsPerBlock.z *
                    blocksPerGrid.x * blocksPerGrid.y * blocksPerGrid.z;

    //
    // Slice up the list and give pieces of it to each thread, letting the pieces grow
    // bigger and bigger until the whole list is sorted
    //
    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_large);
    for (int width = 2; width < (size << 1); width <<= 1) {
        long slices = size / ((nThreads) * width) + 1;

        if (verbose) {
            std::cout << "mergeSort - width: " << width 
                      << ", slices: " << slices 
                      << ", nThreads: " << nThreads << '\n';
            tm();
        }

        // Actually call the kernel
        gpu_mergesort<<<blocksPerGrid, threadsPerBlock>>>(A, B, size, width, slices, D_threads, D_blocks);

        if (verbose)
            std::cout << "call mergesort kernel: " << tm() << " microseconds\n";

        // Switch the input / output arrays instead of copying them around
        A = A == D_data ? D_swp : D_data;
        B = B == D_data ? D_swp : D_data;
    }
    CALI_MARK_END(comp_large);
    CALI_MARK_END(comp);

    //
    // Get the list back from the GPU
    //
    tm();
    checkCudaErrors(cudaMemcpy(data, A, size * sizeof(long), cudaMemcpyDeviceToHost));
    if (verbose)
        std::cout << "cudaMemcpy list back to host: " << tm() << " microseconds\n";
    
    
    // Free the GPU memory
    checkCudaErrors(cudaFree(A));
    checkCudaErrors(cudaFree(B));
    if (verbose)
        std::cout << "cudaFree: " << tm() << " microseconds\n";
}

// GPU helper function
// calculate the id of the current thread
__device__ unsigned int getIdx(dim3* threads, dim3* blocks) {
    int x;
    return threadIdx.x +
           threadIdx.y * (x  = threads->x) +
           threadIdx.z * (x *= threads->y) +
           blockIdx.x  * (x *= threads->z) +
           blockIdx.y  * (x *= blocks->z) +
           blockIdx.z  * (x *= blocks->y);
}

//
// Perform a full mergesort on our section of the data.
//
__global__ void gpu_mergesort(long* source, long* dest, long size, long width, long slices, dim3* threads, dim3* blocks) {
    unsigned int idx = getIdx(threads, blocks);
    long start = width*idx*slices, 
         middle, 
         end;

    for (long slice = 0; slice < slices; slice++) {
        if (start >= size)
            break;

        middle = min(start + (width >> 1), size);
        end = min(start + width, size);
        gpu_bottomUpMerge(source, dest, start, middle, end);
        start += width;
    }
}

//
// Finally, sort something
// gets called by gpu_mergesort() for each slice
//
__device__ void gpu_bottomUpMerge(long* source, long* dest, long start, long middle, long end) {
    long i = start;
    long j = middle;
    for (long k = start; k < end; k++) {
        if (i < middle && (j >= end || source[i] < source[j])) {
            dest[k] = source[i];
            i++;
        } else {
            dest[k] = source[j];
            j++;
        }
    }
}

// read data into a minimal linked list
typedef struct {
    int v;
    void* next;
} LinkNode;

// helper function for reading numbers from stdin
// it's 'optimized' not to check validity of the characters it reads in..
long readList(long** list) {
    tm();
    long v, size = 0;
    LinkNode* node = 0;
    LinkNode* first = 0;
    while (std::cin >> v) {
        LinkNode* next = new LinkNode();
        next->v = v;
        if (node)
            node->next = next;
        else 
            first = next;
        node = next;
        size++;
    }


    if (size) {
        *list = new long[size]; 
        LinkNode* node = first;
        long i = 0;
        while (node) {
            (*list)[i++] = node->v;
            node = (LinkNode*) node->next;
        }

    }

    if (verbose)
        std::cout << "read stdin: " << tm() << " microseconds\n";

    return size;
}


// 
// Get the time (in microseconds) since the last call to tm();
// the first value returned by this must not be trusted
//
timeval tStart;
int tm() {
    timeval tEnd;
    gettimeofday(&tEnd, 0);
    int t = (tEnd.tv_sec - tStart.tv_sec) * 1000000 + tEnd.tv_usec - tStart.tv_usec;
    tStart = tEnd;
    return t;
}
