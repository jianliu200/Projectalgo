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
procedure BUILD TREE (A[1...n]) 
  begin 
    for each process i do 
    begin 
        root := i; 
        parenti := root; 
        leftchild[i] := rightchild[i] := n + 1; 
    end for 
    repeat for each process i  r oot do 
    begin 
      if (A[i] < A[parenti]) or (A[i]= A[parenti] and i <parenti) then 
      begin 
          leftchild[parenti] :=i ; 
          if i = leftchild[parenti] then exit 
          else parenti := leftchild[parenti]; 
      end for 
      else 
      begin 
          rightchild[parenti] :=i; 
          if i = rightchild[parenti] then exit 
          else parenti := rightchild[parenti]; 
      end else 
    end repeat 
end BUILD_TREE 
```
The way we want to compare the different versions of the code is by using CPU-only (MPI) and GPU-only (CUDA) and time it to see how long it takes for the cases to run. We are also going to be comparing them with the same task and see how only it takes for each one of them to run.
<!-- 
For example:
- Algorithm 1a (MPI + CUDA)
- Algorithm 1b (MPI on each core)
- Algorithm 2a (MPI + CUDA)
- Algorithm 2b (MPI on each core) -->
