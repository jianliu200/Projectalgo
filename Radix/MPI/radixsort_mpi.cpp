#include <stdio.h>
#include <mpi.h>

int getMax(int arr[], int n) {
  int mx = arr[0];
  for (int i = 1; i < n; i++)
    if (arr[i] > mx)
      mx = arr[i];
  return mx;
}

void countSort(int arr[], int n, int exp) {
  int output[n];
  int i, count[10] = { 0 };

  for (i = 0; i < n; i++)
    count[(arr[i] / exp) % 10]++;

  for (i = 1; i < 10; i++)
    count[i] += count[i - 1];

  for (i = n - 1; i >= 0; i--) {
    output[count[(arr[i] / exp) % 10] - 1] = arr[i];
    count[(arr[i] / exp) % 10]--;
  }

  for (i = 0; i < n; i++)
    arr[i] = output[i];
}

void radixsort(int arr[], int n, int rank, int size) {
  int m = getMax(arr, n);
  int local_n = n / size;
  int local_arr[local_n];
  MPI_Scatter(arr, local_n, MPI_INT, local_arr, local_n, MPI_INT, 0, MPI_COMM_WORLD);

  for (int exp = 1; m / exp > 0; exp *= 10) {
    int local_count[10] = { 0 };
    int local_output[local_n];
    for (int i = 0; i < local_n; i++)
      local_count[(local_arr[i] / exp) % 10]++;
    int global_count[10];
    MPI_Allreduce(local_count, global_count, 10, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    for (int i = 1; i < 10; i++)
      global_count[i] += global_count[i - 1];
    for (int i = local_n - 1; i >= 0; i--) {
      local_output[global_count[(local_arr[i] / exp) % 10] - 1] = local_arr[i];
      global_count[(local_arr[i] / exp) % 10]--;
    }
    MPI_Allgather(local_output, local_n, MPI_INT, arr, local_n, MPI_INT, MPI_COMM_WORLD);
  }
}

void print(int arr[], int n) {
  for (int i = 0; i < n; i++)
    printf("%d ", arr[i]);
}

int main(int argc, char** argv) {
  int arr[] = { 170, 45, 75, 90, 802, 24, 2, 66 };
  int n = sizeof(arr) / sizeof(arr[0]);

  MPI_Init(&argc, &argv);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  radixsort(arr, n, rank, size);

  if (rank == 0) {
    print(arr, n);
  }

  MPI_Finalize();
  return 0;
}
