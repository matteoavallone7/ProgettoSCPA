#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include "matmul_cuda.h"

#define BLOCK_SIZE 16
#define WARP_SIZE 32
#define BM 128
#define BN 32
#define BK 32
#define TM  8
#define TK  4

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      exit(code);
   }
}


template<int K>
__global__ void matmul_kernel_registers(
    const float *__restrict__ A,
    const float *__restrict__ X,
    float *__restrict__ Y,
    int m, int n, int k)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;

    const float *A_row = A + i * n;

    float sums[K];

    #pragma unroll
    for (int j = 0; j < K; j++)
        sums[j] = 0.0f;

    for (int l = 0; l < n; l++) {

        float a_val = __ldg(&A_row[l]);

        #pragma unroll
        for (int j = 0; j < K; j++)
            if (j < k)
                sums[j] += a_val * __ldg(&X[l * k + j]);
    }

    #pragma unroll
    for (int j = 0; j < K; j++)
        if (j < k)
            Y[i * k + j] = sums[j];
}

__global__ void matmul_kernel_warp_per_row(const float *__restrict__ A, const float *__restrict__ X,
                                            float *__restrict__ Y, int m, int n, int k) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;

    if (warp_id >= m) return;


    for (int col = lane; col < k; col += WARP_SIZE) {
        float sum = 0.0f;

        for (int l = 0; l < n; l++) {
            float a_val = __ldg(&A[warp_id * n + l]);
            sum += a_val * __ldg(&X[l * k + col]);
        }

        Y[warp_id * k + col] = sum;
    }
}


__global__ void matmul_kernel_warp_per_row_gridstride(
    const float *__restrict__ A,
    const float *__restrict__ X,
    float *__restrict__ Y,
    int m,
    int n,
    int k)
{

    int global_tid =
        blockIdx.x * blockDim.x + threadIdx.x;

    int warp_id = global_tid / WARP_SIZE;

    int lane = threadIdx.x % WARP_SIZE;

    int total_warps =
        (gridDim.x * blockDim.x) / WARP_SIZE;

    // Grid-stride sulle righe
    for (int row = warp_id; row < m; row += total_warps) {
        // Ogni lane calcola una colonna diversa
        for (int col = lane; col < k; col += WARP_SIZE) {
            float sum = 0.0f;

            for (int l = 0; l < n; l++) {
                float a_val = A[row * n + l];
                sum += a_val * X[l * k + col];
            }
            Y[row * k + col] = sum;
        }
    }
}


// Kernel ottimizzato con shared memory per k piccoli
__global__ void matmul_kernel_shared(const float *__restrict__ A, const float *__restrict__ X,
                                        float *__restrict__ Y, int m, int n, int k) {
    __shared__ float A_tile[BLOCK_SIZE][BLOCK_SIZE + 1];
    __shared__ float X_tile[BLOCK_SIZE][BLOCK_SIZE + 1];

    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    float sum = 0.0f;

    // Loop su tiles
    for (int t = 0; t < (n + BLOCK_SIZE - 1) / BLOCK_SIZE; t++) {
        // Carica tile di A
        int a_col = t * BLOCK_SIZE + threadIdx.x;
        if (row < m && a_col < n) {
            A_tile[threadIdx.y][threadIdx.x] = A[row * n + a_col];
        } else {
            A_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Carica tile di X
        int x_row = t * BLOCK_SIZE + threadIdx.y;
        if (x_row < n && col < k) {
            X_tile[threadIdx.y][threadIdx.x] = X[x_row * k + col];
        } else {
            X_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Calcola prodotto parziale
        for (int i = 0; i < BLOCK_SIZE; i++) {
            float a = A_tile[threadIdx.y][i]; // aiuta a ridurre gli accessi in shared memory
            float b = X_tile[i][threadIdx.x];

            sum += a * b;
        }

        __syncthreads();
    }

    // Scrivi risultato
    if (row < m && col < k) {
        Y[row * k + col] = sum;
    }
}


__global__ void matmul_kernel_2d_reg_tiling(
    const float * __restrict__ A,
    const float * __restrict__ X,
    float * __restrict__ Y,
    int m, int n, int k)
{
    __shared__ float s_A[BM][BN];
    __shared__ float s_X[BN][BK];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int linear_tid = ty * blockDim.x + tx;

    const int row_start = blockIdx.y * BM + ty * TM;
    const int col_start = blockIdx.x * BK + tx * TK;

    float sum[TM][TK] = {{0.0f}};
    float reg_A[TM];
    float reg_X[TK];

    // Pre-calcolo degli indici base (fuori dal loop dei GFLOPS)
    // Per s_A: quanti float4 ci sono in una riga
    const int A_f4_per_row = BN / 4;
    const int a_base_row   = linear_tid / A_f4_per_row;
    const int a_base_col   = (linear_tid % A_f4_per_row) * 4;
    // Quante righe saltiamo ad ogni turno del Grid-Stride
    const int a_row_stride = (blockDim.x * blockDim.y) / A_f4_per_row;

    // Per s_X: quanti float4 ci sono in una riga
    const int X_f4_per_row = BK / 4;
    const int x_base_row   = linear_tid / X_f4_per_row;
    const int x_base_col   = (linear_tid % X_f4_per_row) * 4;
    const int x_row_stride = (blockDim.x * blockDim.y) / X_f4_per_row;

    // Quanti turni servono a 128 thread per riempire le matrici (Costanti a tempo di compilazione)
    const int STRIDES_A = (BM * BN) / (4 * (blockDim.x * blockDim.y));
    const int STRIDES_X = (BN * BK) / (4 * (blockDim.x * blockDim.y));

    for (int n_step = 0; n_step < n; n_step += BN) {

        // Caricamento s_A
        #pragma unroll
        for (int i = 0; i < STRIDES_A; ++i) {
            int row = a_base_row + i * a_row_stride;
            int col = a_base_col;

            int glob_r_A = blockIdx.y * BM + row;
            int glob_c_A = n_step + col;

            if (glob_r_A < m && glob_c_A < n) {
                reinterpret_cast<float4*>(&s_A[row][col])[0] =
                    reinterpret_cast<const float4*>(&A[glob_r_A * n + glob_c_A])[0];
            } else {
                reinterpret_cast<float4*>(&s_A[row][col])[0] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
        }

        // Caricamento s_X
        #pragma unroll
        for (int i = 0; i < STRIDES_X; ++i) {
            int row = x_base_row + i * x_row_stride;
            int col = x_base_col;

            int glob_r_X = n_step + row;
            int glob_c_X = blockIdx.x * BK + col;

            if (glob_r_X < n && glob_c_X < k) {
                reinterpret_cast<float4*>(&s_X[row][col])[0] =
                    reinterpret_cast<const float4*>(&X[glob_r_X * k + glob_c_X])[0];
            } else {
                reinterpret_cast<float4*>(&s_X[row][col])[0] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
        }

        __syncthreads();

        #pragma unroll
        for (int dot = 0; dot < BN; dot++) {
            #pragma unroll
            for (int i = 0; i < TM; i++) reg_A[i] = s_A[ty * TM + i][dot];

            #pragma unroll
            for (int j = 0; j < TK; j++) reg_X[j] = s_X[dot][tx * TK + j];

            #pragma unroll
            for (int i = 0; i < TM; i++) {
                #pragma unroll
                for (int j = 0; j < TK; j++) {
                    sum[i][j] += reg_A[i] * reg_X[j];
                }
            }
        }

        __syncthreads();
    }

    // Scrittura finale
    #pragma unroll
    for (int i = 0; i < TM; i++) {
        #pragma unroll
        for (int j = 0; j < TK; j++) {
            int glob_r = row_start + i;
            int glob_c = col_start + j;
            if (glob_r < m && glob_c < k)
                Y[glob_r * k + glob_c] = sum[i][j];
        }
    }
}

void cuda_init(CUDAContext *ctx, int m, int n, int k) {

    ctx->m = m;
    ctx->n = n;
    ctx->k = k;

    gpuErrchk(cudaMalloc(&ctx->d_A, m * n * sizeof(float)));
    gpuErrchk(cudaMalloc(&ctx->d_X, n * k * sizeof(float)));
    gpuErrchk(cudaMalloc(&ctx->d_Y, m * k * sizeof(float)));
    cublasCreate(&ctx->cublas_handle);
}

void cuda_finalize(CUDAContext *ctx) {

    cublasDestroy(ctx->cublas_handle);
    gpuErrchk(cudaFree(ctx->d_A));
    gpuErrchk(cudaFree(ctx->d_X));
    gpuErrchk(cudaFree(ctx->d_Y));
}

void cuda_upload(CUDAContext *ctx, const float *A, const float *X, double *time_h2d) {
    cudaEvent_t start, stop;
	float milliseconds;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
    gpuErrchk(cudaMemcpy(ctx->d_A, A, ctx->m * ctx->n * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(ctx->d_X, X, ctx->n * ctx->k * sizeof(float), cudaMemcpyHostToDevice));
    cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	*time_h2d = milliseconds / 1000.0;
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
}

void cuda_download(CUDAContext *ctx, float *Y, double *time_d2h) {
    cudaEvent_t start, stop;
	float milliseconds;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
    gpuErrchk(cudaMemcpy(Y, ctx->d_Y, ctx->m * ctx->k * sizeof(float), cudaMemcpyDeviceToHost));
    cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	*time_d2h = milliseconds / 1000.0;
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
}

void cuda_compute(CUDAContext *ctx, double *kernel_time, const char **kernel_name) {
    int m = ctx->m;
    int n = ctx->n;
    int k = ctx->k;

    float *d_A = ctx->d_A;
    float *d_X = ctx->d_X;
    float *d_Y = ctx->d_Y;

    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    int num_sms;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);

    const int THREADS = 256;

    if (k <= 16) {
        *kernel_name = "reg";
        int blocks = (m + THREADS - 1) / THREADS;

        if (k <=4) {
            matmul_kernel_registers<4><<<blocks, THREADS>>>(d_A, d_X, d_Y, m, n, k);
        } else if (k <= 8) {
            matmul_kernel_registers<8><<<blocks, THREADS>>>(d_A, d_X, d_Y, m, n, k);
        } else {
            matmul_kernel_registers<16><<<blocks, THREADS>>>(d_A, d_X, d_Y, m, n, k);
        }

	} else if (k <= 32) {

	    long long work = (long long)m * n;

        if (work <= 32LL * 1024 * 1024) {
            *kernel_name = "warp";
            int blocks = (m * WARP_SIZE + THREADS - 1) / THREADS;
            matmul_kernel_warp_per_row<<<blocks, THREADS>>>(d_A, d_X, d_Y, m, n, k);
        } else {
            *kernel_name = "warp_gs";
            int blocks = 4 * num_sms;
            matmul_kernel_warp_per_row_gridstride<<<blocks, THREADS>>>(d_A, d_X, d_Y, m, n, k);
        }

	} else {
	    if (k <= 64) {
	    *kernel_name = "shared";
		dim3 block(BLOCK_SIZE, BLOCK_SIZE);
		dim3 grid((k + BLOCK_SIZE - 1) / BLOCK_SIZE, (m + BLOCK_SIZE - 1) / BLOCK_SIZE);
		matmul_kernel_shared<<<grid, block>>>(d_A, d_X, d_Y, m, n, k);
		} else {
	    *kernel_name = "shared_rt";
		dim3 block(BK / TK, BM / TM);
		dim3 grid((k + BK - 1) / BK, (m + BM - 1) / BM);
        matmul_kernel_2d_reg_tiling<<<grid, block>>>(d_A, d_X, d_Y, m, n, k);
		}
	}
	gpuErrchk(cudaPeekAtLastError());
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float ms;
	cudaEventElapsedTime(&ms, start, stop);
	*kernel_time = ms / 1000.0;

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
}

// cublas reference: calcola il prodotto matriciale locale Y = A_local * X_local
void cuda_compute_cublas(CUDAContext *ctx, double *kernel_time) {
    int m = ctx->m, n = ctx->n, k = ctx->k;
    const float alpha = 1.f, beta = 0.f;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    cublasSgemm(ctx->cublas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        k, m, n,              // (rows of result, cols of result, inner dim)
        &alpha,
        ctx->d_X, k,          // Xᵀ in col-major, leading dim = k
        ctx->d_A, n,          // Aᵀ in col-major, leading dim = n
        &beta,
        ctx->d_Y, k);         // Yᵀ in col-major, leading dim = k

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    *kernel_time = ms / 1000.0;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}
