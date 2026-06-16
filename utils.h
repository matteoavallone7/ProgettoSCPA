#ifndef UTILS_H
#define UTILS_H

#include <mpi.h>


typedef struct {
    MPI_Comm comm;
    MPI_Comm row_comm;
    MPI_Comm col_comm;
    int P, Q;
    int my_row, my_col;
    int rank, size;
} ProcessGrid;

typedef struct {
    int M_global;
    int N_global;
    int M_loc;
    int N_loc;
    int row_offset;
    int col_offset;
    float *data;
} LocalMatrix;

typedef struct {
    int len_loc;
    int local_size;
    int k;
    int offset;
    float *data;
} LocalMultiVector;

void setup_local_A(LocalMatrix *A_local, int M, int N, ProcessGrid *grid);
void setup_local_X(LocalMultiVector *X_local, ProcessGrid *grid, int N, int k);
void setup_local_Y(LocalMultiVector *Y_local, ProcessGrid *grid, int M, int k, int m_local, int row_offset);
void setup_process_grid(ProcessGrid *grid, int P, int Q);
void matmul_cuda(LocalMatrix *A_local, LocalMultiVector *X_local, LocalMultiVector *Y_local, ProcessGrid *grid, int k);
void gather_vector_gatherv(float *Y_global, const LocalMultiVector *Y_local, ProcessGrid *grid, int n, int k);
void free_process_grid(ProcessGrid *grid);
void matmul_cuda_profiled(const LocalMatrix *A, const LocalMultiVector *X, LocalMultiVector *Y, ProcessGrid *grid, int k, double *time_h2d, double *time_compute, double *time_d2h, double *time_mpi);

#endif
