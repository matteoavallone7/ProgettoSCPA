#ifndef MATMUL_CUDA_H
#define MATMUL_CUDA_H

#ifdef __cplusplus
extern "C" {
#endif

#include <cublas_v2.h>

typedef struct {
    float *d_A;
    float *d_X;
    float *d_Y;
    int m;
    int n;
    int k;
    cublasHandle_t cublas_handle;
} CUDAContext;

void cuda_upload(CUDAContext *ctx, const float *A, const float *X, double *time_h2d);
void cuda_compute(CUDAContext *ctx, double *time_comp, const char **kernel_name);
void cuda_download(CUDAContext *ctx, float *Y, double *time_d2h);
void cuda_init(CUDAContext *ctx, int m, int n, int k);
void cuda_finalize(CUDAContext *ctx);
void cuda_compute_cublas(CUDAContext *ctx, double *kernel_time);

#ifdef __cplusplus
}
#endif

#endif
