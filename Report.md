# CSCE 435 Group project

## 1. Group members:
1. Jiangyuan Liu
2. Akhil Mathew
3. Jacob Thomas
4. Ashwin Kundenti

The way our team is communicating is by using Discord and iMessages
---

## 2. _due 10/25_ Project topic
For our project topic, we are going to be exploring parellel algoithm for sorting.
## 2. _due 10/25_ Brief project description (what algorithms will you be comparing and on what architectures)

Sources for the code and algorithm: 

The algorithm that we are going to use:

1. Merge sort 

```
function parallelMergeSort(array):
  if parent:
    localArray = array
    for num processes:
      recieve(newArray)
      merge(localArray, newArray)
      return localArray
  else:
    localArray = array
    send(localArray)
```
https://rachitvasudeva.medium.com/parallel-merge-sort-algorithm-e8175ab60e7
https://www.sjsu.edu/people/robert.chun/courses/cs159/s3/T.pdf

2. Bitonic sort

3. Quicksort
```
function quicksort(a, low, high):
    if low < high:
        pivotIndex = partition(a, low, high)
        quicksort(a, low, pivotIndex - 1)
        quicksort(a, pivotIndex + 1, high)

function partition(a, low, high):
    pivot = a[(low + high) / 2]
    i = low - 1
    j = high + 1
    while true:
        do:
            i = i + 1
        while a[i] < pivot
        do:
            j = j - 1
        while a[j] > pivot
        if i >= j:
            return j
        swap(a[i], a[j])
```
The way we want to compare the different versions of the code is by using CPU-only (MPI) and GPU-only (CUDA) and time it to see how long it takes for the cases to run. We are also going to be comparing them with the same task and see how only it takes for each one of them to run.
<!-- 
For example:
- Algorithm 1a (MPI + CUDA)
- Algorithm 1b (MPI on each core)
- Algorithm 2a (MPI + CUDA)
- Algorithm 2b (MPI on each core) -->
