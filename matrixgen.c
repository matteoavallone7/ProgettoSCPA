#include <math.h>
#include <stdlib.h>
#include "matrixgen.h"

// Generazione A locale (per versione parallela)
void generate_A_block(
    float *A,
    int M_loc,
    int N_loc,
    int row_offset,
    int col_offset
) {
    for (int i = 0; i < M_loc; i++) {
        int ig = i + row_offset;
        for (int j = 0; j < N_loc; j++) {
            int jg = j + col_offset;
            A[i * N_loc + j] = 1.0f / (1.0f + ig + jg);
        }
    }
}

// Generazione X locale (per versione parallela)
void generate_X_block(
    float *X,
    int N_loc,
    int col_offset,
    int k
) {
    for (int j = 0; j < N_loc; j++) {
        int jg = j + col_offset;
        for (int c = 0; c < k; c++) {
            X[j * k + c] = sinf((float)(jg + c));
        }
    }
}

// Generazione con formula predefinita (per versione seriale)
void generate_matrix_formula(float *A, int M, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            A[i * N + j] = 1.0f / (1.0f + i + j);
        }
    }
}

// Generazione X globale (per versione seriale)
void generate_X_formula(float *X, int N, int k) {
    for (int j = 0; j < N; j++) {
        for (int c = 0; c < k; c++) {
            X[j * k + c] = sinf((float)(j + c));
        }
    }
}
