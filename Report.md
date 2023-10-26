# CSCE 435 Group project

## 1. Group members:
1. Jiangyuan Liu
2. Akhil Mathew
3. Jacob Thomas
4. Ashwin Kundeti

The way our team is communicating is by using Discord and iMessages
---

## 2. _due 10/25_ Project topic
For our project topic, we are going to be exploring parellel algoithm for sorting.
## 2. _due 10/25_ Brief project description (what algorithms will you be comparing and on what architectures)

Sources for the code and algorithms

The algorithm that we are going to use:

1. Merge sort

2. Bitonic sort
```
Pseudocode:
Bitonic sort:
bitonic_sort(A, direction):
    n = A_size
    if n > 1:
        Bitonic_sort(arr[1 to n/2]) //top half
        Bitonic_sort(arr[(n/2 + 1) to n])
        Merge(first and second half)
    end
end

Merge(A, direction):
    n = A_size
    if n > 1:
        for i:n/2:
            if A[i] > A[i + n/2]:
                swap(A[i] A[i + n/2])
            end
        end
    end
end
```
source: https://www.baeldung.com/cs/bitonic-sort

3. Quicksort

The way we want to compare the different versions of the code is by using CPU-only (MPI) and GPU-only (CUDA) and time it to see how long it takes for the cases to run. We are also going to be comparing them with the same task and see how only it takes for each one of them to run.
<!--
For example:
- Algorithm 1a (MPI + CUDA)
- Algorithm 1b (MPI on each core)
- Algorithm 2a (MPI + CUDA)
- Algorithm 2b (MPI on each core) -->
