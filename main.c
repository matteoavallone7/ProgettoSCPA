#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "matrixgen.h"
#include "utils.h"
#include "matmul_serial.h"
#include "cuda/matmul_cuda.h"


void print_usage(const char *prog_name) {
    printf("Uso: mpirun -np <nprocs> %s [opzioni]\n", prog_name);
    printf("Opzioni:\n");
    printf("  -M <int>      : Numero di righe della matrice (default: 1000)\n");
    printf("  -N <int>      : Numero di colonne della matrice (default: 800)\n");
    printf("  -k <int>      : Numero di colonne del multivettore (default: 8)\n");
    printf("  -P <int>      : Righe della griglia di processi (default: 2)\n");
    printf("  -Q <int>      : Colonne della griglia di processi (default: 2)\n");
    printf("  -verify       : Verifica con implementazione seriale\n");
    printf("  -n_iter <int> : Numero di iterazioni (default: 1)\n");
    printf("  -h            : Mostra questo help\n");
}

int main(int argc, char *argv[]){
    MPI_Init(&argc, &argv);

    int rank, size;
    // valori di default
    int M = 1000, N = 800, k = 8;
    int P = 2, Q = 2;
    int verify = 0;
    int n_iter = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Parsing argomenti
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-M") == 0 && i + 1 < argc) {
            M = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-N") == 0 && i + 1 < argc) {
            N = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-k") == 0 && i + 1 < argc) {
            k = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-P") == 0 && i + 1 < argc) {
            P = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-Q") == 0 && i + 1 < argc) {
            Q = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-verify") == 0) {
            verify = 1;
        } else if (strcmp(argv[i], "-n_iter") == 0 && i + 1 < argc) {
            n_iter = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) print_usage(argv[0]);
            MPI_Finalize();
            return 0;
        }
    }


    if (rank == 0) {
        printf("\n=== Prodotto Matrice-Multivettore MPI+CUDA ===\n");
        printf("Dimensioni: A(%dx%d) * X(%dx%d) = Y(%dx%d)\n", M, N, N, k, M, k);
        printf("Griglia processi: %dx%d = %d processi\n", P, Q, P*Q);
        printf("\n");
    }


    if (size != P * Q) {
        if (rank == 0) {
            fprintf(stderr, "Errore: numero processi (%d) != P*Q (%d)\n", size, P*Q);
            print_usage(argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    ProcessGrid grid;
    setup_process_grid(&grid, P, Q);

    // Allocazione e generazione dati seriali (solo per verifica)
    float *A_global = NULL, *X_global = NULL, *Y_global = NULL;
    float *Y_serial = NULL;
    double T_serial = 0.0;

    if (verify && rank == 0){
        A_global = malloc(M * N * sizeof(float));
        X_global = malloc(N * k * sizeof(float));
        Y_global = malloc(M * k * sizeof(float));
        Y_serial = malloc(M * k * sizeof(float));

        generate_matrix_formula(A_global, M, N);
        generate_X_formula(X_global, N, k);

        // Calcolo seriale per verifica
        double start_serial = MPI_Wtime();
        matmul_serial(M, N, k, A_global, X_global, Y_serial);
        double end_serial = MPI_Wtime();
        T_serial = end_serial - start_serial;
        printf("Tempo calcolo seriale: %.6f s\n\n", T_serial);
    }

    LocalMatrix A_local;
    LocalMultiVector X_local, Y_local;
    double start_setup = MPI_Wtime();

    setup_local_A(&A_local, M, N, &grid);
    setup_local_X(&X_local, &grid, N, k);
    setup_local_Y(&Y_local, &grid, M, k, A_local.M_loc, A_local.row_offset);

    generate_A_block(A_local.data, A_local.M_loc, A_local.N_loc, A_local.row_offset, A_local.col_offset);


    /* if (grid.my_col == 0) {
        generate_X_block(X_local.data, X_local.local_size, X_local.offset, k);
    }

    MPI_Bcast(X_local.data, X_local.local_size * k, MPI_DOUBLE, 0, grid.row_comm); */

    // Tutti allocano un buffer temporaneo per X completo (solo col=0 lo riempie)
    float *X_full = NULL;
    int *sendcounts = calloc(grid.Q, sizeof(int));
    int *displs_x   = calloc(grid.Q, sizeof(int));

    // Calcola sendcounts e displs per ogni processo nella row_comm
    int base = N / grid.Q, extra = N % grid.Q;
    for (int q = 0; q < grid.Q; q++) {
        int sz  = base + (q < extra ? 1 : 0);
        int off = q * base + (q < extra ? q : extra);
        sendcounts[q] = sz * k;
        displs_x[q]   = off * k;
    }

    if (grid.my_col == 0) {
        X_full = malloc(N * k * sizeof(float));
        generate_X_block(X_full, N, 0, k);  // genera X globale
    }

    MPI_Scatterv(X_full, sendcounts, displs_x, MPI_FLOAT,
                 X_local.data, X_local.local_size * k, MPI_FLOAT,
                 0, grid.row_comm);

    if (grid.my_col == 0) free(X_full);
    free(sendcounts);
    free(displs_x);

    printf("Rank (%d,%d) X[0]=%f\n",
           grid.my_row, grid.my_col,
           X_local.data[0]);


    MPI_Barrier(grid.comm);
    double end_setup = MPI_Wtime();

    if (rank == 0) {
        printf("Tempo setup e generazione distribuita: %.6f s\n", end_setup - start_setup);
    }

    // Calcolo parallelo
    // Barriera iniziale per far partire tutti insieme
    CUDAContext ctx;
    double total_h2d = 0, total_comp = 0, total_d2h = 0, total_mpi = 0;

    cuda_init(&ctx, A_local.M_loc, A_local.N_loc, k);
    cuda_upload(&ctx, A_local.data, X_local.data, &total_h2d);
    float *Y_temp = malloc(ctx.m * ctx.k * sizeof(float));

    MPI_Barrier(grid.comm);

    const char *kernel_used = NULL;

    for (int i = 0; i < n_iter; i++) {
        double k_time, d2h_time;

        cuda_compute(&ctx, &k_time, &kernel_used);

        total_comp += k_time;
        cuda_download(&ctx, Y_temp, &d2h_time);
        total_d2h += d2h_time;

        double start_mpi = MPI_Wtime();
        MPI_Allreduce(Y_temp, Y_local.data, ctx.m * ctx.k, MPI_FLOAT, MPI_SUM, grid.row_comm);
        total_mpi += MPI_Wtime() - start_mpi;
    }


    // cublas benchmark
    double total_cublas = 0.0;
    double total_cublas_mpi = 0.0;

    for (int i = 0; i < n_iter; i++) {

        double cublas_time, d2h_time;

        cuda_compute_cublas(&ctx, &cublas_time);

        total_cublas += cublas_time;

        cuda_download(&ctx, Y_temp, &d2h_time);

        double start_mpi = MPI_Wtime();

        MPI_Allreduce(Y_temp,
                      Y_local.data,
                      ctx.m * ctx.k,
                      MPI_FLOAT,
                      MPI_SUM,
                      grid.row_comm);

        total_cublas_mpi += MPI_Wtime() - start_mpi;
    }

    double avg_cublas =
        total_cublas / n_iter;

    double avg_cublas_mpi =
        total_cublas_mpi / n_iter;

    double T_cublas =
        avg_cublas + avg_cublas_mpi;

    double cublas_gflops =
        (2.0 * M * N * k) / T_cublas / 1e9;

    cuda_finalize(&ctx);

    // Calcolo delle medie
    double avg_comp = total_comp / n_iter;
    double avg_d2h  = total_d2h / n_iter;
    double avg_mpi  = total_mpi / n_iter;

    // Il tempo di "calcolo del prodotto" richiesto include Kernel + MPI AllReduce
    // ma ESCLUDE i trasferimenti H2D e D2H come da specifica.
    double T_calc = avg_comp + avg_mpi;

    double T_wall;
    MPI_Reduce(&T_calc, &T_wall, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Raccolta risultati
    double start_gather = MPI_Wtime();
    if (verify) {
        // Alloca Y_global solo se serve per verifica
        if (rank == 0 && Y_global == NULL) {
            Y_global = (float*)malloc(M * k * sizeof(float));
        }
        gather_vector_gatherv(Y_global, &Y_local, &grid, M, k);
    }
    double end_gather = MPI_Wtime();

    if (rank == 0) {
        printf("\n=== Prestazioni (Media su %d iterazioni) ===\n", n_iter);
        printf("Tempo Kernel GPU (puro):  %.6f s\n", avg_comp);
        printf("Tempo MPI (AllReduce):    %.6f s\n", avg_mpi);
        printf("Tempo calcolo totale (T): %.6f s\n", T_calc);

        printf("\n--- Dettagli Trasferimenti (Esclusi da GFLOPS) ---\n");
        printf("Tempo H2D (A e X):        %.6f s\n", total_h2d);
        printf("Tempo D2H (Y_temp):       %.6f s\n", avg_d2h);

        if (verify) {
            printf("Tempo gathering:          %.6f s\n", end_gather - start_gather);
        }

        // Calcolo GFLOPS usando T_calc (Kernel + MPI)
        double gflops = (2.0 * M * N * k) / T_wall / 1e9;
        printf("\nPrestazioni Finali:       %.3f GFLOPS\n", gflops);

        double T_parallel =
            total_h2d +
            avg_comp +
            avg_d2h +
            avg_mpi +
            (end_gather - start_gather);

        double speedup = T_serial / T_parallel;
        printf("Speedup rispetto al seriale: %.2fx\n", speedup);

        if (verify) {
            printf("\n=== Verifica ===\n");
            int correct = verify_result(Y_serial, Y_global, M, k, 1e-5f, 1e-4f);
            if (correct) {
                printf("✓ Risultato CORRETTO!\n");
            } else {
                printf("✗ Risultato ERRATO!\n");
            }
            free(Y_serial);
        }

        printf("\n=== cuBLAS Baseline ===\n");
        printf("cuBLAS time: %.6f s\n", avg_cublas);
        printf("cuBLAS GFLOPS: %.3f\n", cublas_gflops);

        printf("Relative performance vs cuBLAS: %.2f%%\n",
               100.0 * gflops / cublas_gflops);

        FILE *csv = fopen("results/benchmark_results.csv", "a");
            if (csv == NULL) {
                fprintf(stderr, "Errore apertura CSV: la directory 'results/' non esiste\n");
            } else {
                fseek(csv, 0, SEEK_END);
                if (ftell(csv) == 0) {
                    fprintf(csv, "M,N,k,P,Q,n_iter,kernel,kernel_time,mpi_time,total_calc,"
                                 "h2d_time,d2h_time,gflops,speedup,cublas_gflops\n");
                }
                fprintf(csv, "%d,%d,%d,%d,%d,%d,%s,%.6f,%.6f,%.6f,%.6f,%.6f,%.3f,%.2f,%.3f\n",
                        M, N, k, P, Q, n_iter, kernel_used,
                        avg_comp, avg_mpi, T_calc,
                        total_h2d, avg_d2h,
                        gflops, speedup, cublas_gflops);
                fclose(csv);
            }

        if (A_global) free(A_global);
        if (X_global) free(X_global);
        if (Y_global) free(Y_global);
    }

    // Cleanup
    free(A_local.data);
    free(X_local.data);
    free(Y_local.data);
    free_process_grid(&grid);

    MPI_Finalize();
    return 0;
}
