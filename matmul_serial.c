#include "matmul_serial.h"
#include <string.h>
#include <stdio.h>
#include <math.h>


void matmul_serial(
    int M,
    int N,
    int k,
    const float *A,
    const float *X,
    float *Y)
{
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < k; j++) {
            float sum = 0.0f;
            for (int l = 0; l < N; l++) {
                sum += A[i * N + l] * X[l * k + j];
            }
            Y[i * k + j] = sum;
        }
    }
}

int verify_result(const float *Y_serial,
                  const float *Y_parallel,
                  int M,
                  int k,
                  float abs_t,
                  float rel_t)
{
    int errors = 0;

    float max_abs_error = 0.0f;
    float max_rel_error = 0.0f;

    for (int i = 0; i < M * k; i++) {

        float ref = Y_serial[i];
        float got = Y_parallel[i];

        float abs_error = fabsf(ref - got);

        float denom = fmaxf(fabsf(ref), fabsf(got));

        if (denom < 1e-7f)
            denom = 1.0f;

        float rel_error = abs_error / denom;

        max_abs_error = fmaxf(max_abs_error, abs_error);
        max_rel_error = fmaxf(max_rel_error, rel_error);

        // fallisce se entrambi sono troppo grandi
        if (abs_error > abs_t && rel_error > rel_t) {

            errors++;

            if (errors <= 10) {
                printf(
                    "Error at %d:\n"
                    "  expected = %.7f\n"
                    "  obtained = %.7f\n"
                    "  abs err  = %e\n"
                    "  rel err  = %e\n",
                    i,
                    ref,
                    got,
                    abs_error,
                    rel_error
                );
            }
        }
    }

    printf("Max absolute error: %e\n", max_abs_error);
    printf("Max relative error: %e\n", max_rel_error);
    printf("Errors (abs>%e && rel>%e): %d\n",
           abs_t,
           rel_t,
           errors);

    return errors == 0;
}
