#ifndef MATMUL_SERIAL_H
#define MATMUL_SERIAL_H


void matmul_serial(
    int M,
    int N,
    int k,
    const float *A,
    const float *X,
    float *Y
);


int verify_result(const float *Y_serial, const float *Y_parallel,
                  int M, int k, float abs_t, float rel_t);

#endif
