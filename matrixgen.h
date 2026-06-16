#ifndef MATRIXGEN_H
#define MATRIXGEN_H


void generate_A_block(
    float *A,
    int M_loc,
    int N_loc,
    int row_offset,
    int col_offset
);


void generate_X_block(
    float *X,
    int N_loc,
    int col_offset,
    int k
);


void generate_matrix_formula(float *A, int M, int N);
void generate_X_formula(float *X, int N, int k);

#endif
