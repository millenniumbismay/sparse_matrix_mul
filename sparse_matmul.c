#include <string.h>
#include <Accelerate/Accelerate.h>

/*
 * Sparse A (CSR) × Dense B → Dense C
 * For each row i of A, finds non-zero entries and uses Accelerate's
 * cblas_saxpy to accumulate: C[i,:] += a_val * B[k,:]
 * This exploits sparsity in A while using BLAS for the vectorized inner loop.
 */
void sparse_matmul(
    const int *indptr,
    const int *indices,
    const float *data,
    const float *b,
    float *result,
    int m,       /* rows of A / rows of result */
    int n,       /* cols of A / rows of B */
    int k        /* cols of B / cols of result */
) {
    memset(result, 0, (size_t)m * k * sizeof(float));

    for (int i = 0; i < m; i++) {
        float *row_result = result + (size_t)i * k;
        for (int idx = indptr[i]; idx < indptr[i + 1]; idx++) {
            int col = indices[idx];
            float val = data[idx];
            /* C[i,:] += val * B[col,:] — vectorized via BLAS saxpy */
            cblas_saxpy(k, val, b + (size_t)col * k, 1, row_result, 1);
        }
    }
}
