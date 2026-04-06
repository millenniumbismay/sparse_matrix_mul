#include <string.h>
#include <dispatch/dispatch.h>
#include <Accelerate/Accelerate.h>

/*
 * Sparse CSR(A) × Dense(B) → Dense(C) with GCD parallelism
 * Uses vDSP_vsma (scalar multiply-add) for vectorized inner loop
 * Parallel across rows of A via GCD dispatch_apply
 */
void sparse_matmul(
    const int *indptr,
    const int *indices,
    const float *data,
    const float *b,
    float *result,
    int m,
    int n,
    int k
) {
    memset(result, 0, (size_t)m * k * sizeof(float));

    dispatch_apply(m, dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0),
        ^(size_t i) {
            float *row_result = result + i * k;
            for (int idx = indptr[i]; idx < indptr[i + 1]; idx++) {
                int col = indices[idx];
                float val = data[idx];
                /* row_result += val * B[col,:] using vDSP */
                vDSP_vsma(b + (size_t)col * k, 1, &val, row_result, 1, row_result, 1, k);
            }
        }
    );
}
