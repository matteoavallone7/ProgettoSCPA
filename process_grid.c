#include <stdio.h>
#include "utils.h"


void setup_process_grid(ProcessGrid *grid, int P, int Q) {
    MPI_Comm_rank(MPI_COMM_WORLD, &grid->rank);
    MPI_Comm_size(MPI_COMM_WORLD, &grid->size);

    if (grid->size != P * Q) {
        if (grid->rank == 0) {
            fprintf(stderr, "Errore: numero di processi (%d) != P*Q (%d*%d=%d)\n",
                    grid->size, P, Q, P*Q);
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    grid->P = P;
    grid->Q = Q;

    // Crea griglia cartesiana 2D
    int dims[2] = {P, Q};
    int periods[2] = {0, 0};  // Non periodica
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &grid->comm);

    // Ottengo coordinate nella griglia
    int coords[2];
    MPI_Cart_coords(grid->comm, grid->rank, 2, coords);
    grid->my_row = coords[0];
    grid->my_col = coords[1];

    // Creo communicator di riga (processi con stessa riga)
    int remain_dims[2] = {0, 1};  // Mantieni solo dimensione colonne
    MPI_Cart_sub(grid->comm, remain_dims, &grid->row_comm);

    // Creo communicator di colonna (processi con stessa colonna)
    remain_dims[0] = 1;
    remain_dims[1] = 0;  // Mantieni solo dimensione righe
    MPI_Cart_sub(grid->comm, remain_dims, &grid->col_comm);


}

void free_process_grid(ProcessGrid *grid) {
    if (grid->row_comm != MPI_COMM_NULL) {
        MPI_Comm_free(&grid->row_comm);
    }
    if (grid->col_comm != MPI_COMM_NULL) {
        MPI_Comm_free(&grid->col_comm);
    }
    if (grid->comm != MPI_COMM_NULL) {
        MPI_Comm_free(&grid->comm);
    }
}
