#include "utils.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

void setup_local_A(LocalMatrix *A_local, int M, int N, ProcessGrid *grid)
{
    A_local->M_global = M;
    A_local->N_global = N;

    int base_rows = M / grid->P;
    int remainder_rows = M % grid->P;

    A_local->M_loc = base_rows + (grid->my_row < remainder_rows ? 1 : 0);

    int base_cols = N / grid->Q;
    int remainder_cols = N % grid->Q;

    A_local->N_loc = base_cols + (grid->my_col < remainder_cols ? 1 : 0);

    A_local->row_offset = grid->my_row * base_rows + (grid->my_row < remainder_rows ? grid->my_row : remainder_rows);
    A_local->col_offset = grid->my_col * base_cols + (grid->my_col < remainder_cols ? grid->my_col : remainder_cols);

    A_local->data = (float*)malloc(A_local->M_loc * A_local->N_loc * sizeof(float));

}


void setup_local_X(LocalMultiVector *X_local, ProcessGrid *grid, int N, int k)
{
    X_local->len_loc = N;
    X_local->k = k;

    // X distribuito lungo le colonne dei processi (Q)
    int base = N / grid->Q;
    int extra = N % grid->Q;

    X_local->local_size = base + (grid->my_col < extra ? 1 : 0);

    X_local->offset =
        grid->my_col * base +
        (grid->my_col < extra ? grid->my_col : extra);

    X_local->data = malloc(X_local->local_size * k * sizeof(float));

    if (grid->my_col == 0) {
        if (grid->rank == 0) {
            printf("Process (%d,%d): X[%d:%d, :] (%dx%d)\n",
                   grid->my_row, grid->my_col,
                   X_local->offset,
                   X_local->offset + X_local->local_size,
                   X_local->local_size, k);
        }
    } else {
        memset(X_local->data, 0,
               X_local->local_size * k * sizeof(float));
    }
}

void setup_local_Y(LocalMultiVector *Y_local, ProcessGrid *grid, int M, int k,
                   int m_local, int row_offset) {
    Y_local->len_loc = M;
    Y_local->k = k;
    Y_local->local_size = m_local;
    Y_local->offset = row_offset;
    Y_local->data = (float*)malloc(Y_local->local_size * k * sizeof(float));
    memset(Y_local->data, 0, Y_local->local_size * k * sizeof(float));
}

void gather_vector_gatherv(float *Y_global, const LocalMultiVector *Y_local, ProcessGrid *grid, int n, int k) {

    // 1. Prepariamo gli array per la ricezione (solo sul Rank 0)
    int *recvcounts = NULL;
    int *displs = NULL;

    if (grid->rank == 0) {
        recvcounts = (int*)malloc(grid->P * sizeof(int));
        displs = (int*)malloc(grid->P * sizeof(int));

        int base_size = n / grid->P;
        int extra = n % grid->P;

        for (int p = 0; p < grid->P; p++) {
            int size_p = base_size + (p < extra ? 1 : 0);
            recvcounts[p] = size_p * k; // Numero di float totali per processo (quanti dati arrivano da ogni processo)

            int off_p = p * base_size + (p < extra ? p : extra);
            displs[p] = off_p * k;      // Offset nel buffer globale (dove incollarli)
        }
    }

    // 2. Chiamata collettiva
    // Nota: Usiamo col_comm perché vogliamo raccogliere solo dai "leader" di riga
    if (grid->my_col == 0) {
        MPI_Gatherv(Y_local->data, Y_local->local_size * k, MPI_FLOAT,
                    Y_global, recvcounts, displs, MPI_FLOAT,
                    0, grid->col_comm);
    }

    // 3. Pulizia
    if (grid->rank == 0) {
        free(recvcounts);
        free(displs);
    }
}
