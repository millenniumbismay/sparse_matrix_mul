#include <stdlib.h>
#include <string.h>

/*
 * Sparse matrix multiplication using CSR format for both A and B.
 * C = A * B where A is rows_a x K, B is K x cols_b.
 * Result is stored as a flat dense array (rows_a * cols_b).
 */
void sparse_matmul(
    int rows_a, int cols_b,
    int *a_rowptr, int *a_colidx, long long *a_vals,
    int *b_rowptr, int *b_colidx, long long *b_vals,
    long long *result
) {
    memset(result, 0, (long long)rows_a * cols_b * sizeof(long long));

    for (int i = 0; i < rows_a; i++) {
        long long *res_row = result + (long long)i * cols_b;
        for (int ak = a_rowptr[i]; ak < a_rowptr[i + 1]; ak++) {
            int k = a_colidx[ak];
            long long a_val = a_vals[ak];
            for (int bj = b_rowptr[k]; bj < b_rowptr[k + 1]; bj++) {
                res_row[b_colidx[bj]] += a_val * b_vals[bj];
            }
        }
    }
}
