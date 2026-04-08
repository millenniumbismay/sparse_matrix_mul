#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <mach/mach_time.h>
#include <dispatch/dispatch.h>

/*
 * Dual CSR Merge-Iterate Row-Pair Multiply (DCSRM-RPI)
 *
 * Key innovations:
 * 1. Both A and B in compact CSR (int16 cols + int8 vals, 3 bytes/entry)
 * 2. Merge-iterate row pairs: sorted CSR indices enable merge-based
 *    detection of shared K values for B-row temporal reuse
 * 3. Batch execution: all test cases in single C call, C-side timing
 * 4. Parallel row processing via GCD dispatch_apply
 */

/* Build compact CSR for a matrix. Returns nnz. */
int build_compact_csr(
    const long long *flat, int rows, int cols,
    int *rowptr, int16_t *colidx, int8_t *vals
) {
    rowptr[0] = 0;
    int pos = 0;
    for (int r = 0; r < rows; r++) {
        const long long *row = flat + (long long)r * cols;
        for (int c = 0; c < cols; c++) {
            if (row[c] != 0) {
                colidx[pos] = (int16_t)c;
                vals[pos] = (int8_t)row[c];
                pos++;
            }
        }
        rowptr[r + 1] = pos;
    }
    return pos;
}

/* Core single-test multiply: merge-iterate CSR A row pairs */
static void multiply_single(
    int rows_a, int cols_b,
    const int *a_rowptr, const int16_t *a_colidx, const int8_t *a_vals,
    const int *b_rowptr, const int16_t *b_colidx, const int8_t *b_vals,
    long long *result
) {
    memset(result, 0, (long long)rows_a * cols_b * sizeof(long long));

    int i = 0;
    for (; i + 1 < rows_a; i += 2) {
        long long * restrict res1 = result + (long long)i * cols_b;
        long long * restrict res2 = result + (long long)(i + 1) * cols_b;

        int a1_start = a_rowptr[i];
        int a1_end = a_rowptr[i + 1];
        int a2_start = a_rowptr[i + 1];
        int a2_end = a_rowptr[i + 2];

        int p1 = a1_start, p2 = a2_start;
        while (p1 < a1_end && p2 < a2_end) {
            int k1 = a_colidx[p1];
            int k2 = a_colidx[p2];
            if (k1 == k2) {
                long long av1 = (long long)a_vals[p1];
                long long av2 = (long long)a_vals[p2];
                int b_start = b_rowptr[k1];
                int b_end = b_rowptr[k1 + 1];
                for (int bj = b_start; bj < b_end; bj++) {
                    int col = b_colidx[bj];
                    long long bv = (long long)b_vals[bj];
                    res1[col] += av1 * bv;
                    res2[col] += av2 * bv;
                }
                p1++; p2++;
            } else if (k1 < k2) {
                long long av1 = (long long)a_vals[p1];
                int b_start = b_rowptr[k1];
                int b_end = b_rowptr[k1 + 1];
                for (int bj = b_start; bj < b_end; bj++) {
                    res1[b_colidx[bj]] += av1 * (long long)b_vals[bj];
                }
                p1++;
            } else {
                long long av2 = (long long)a_vals[p2];
                int b_start = b_rowptr[k2];
                int b_end = b_rowptr[k2 + 1];
                for (int bj = b_start; bj < b_end; bj++) {
                    res2[b_colidx[bj]] += av2 * (long long)b_vals[bj];
                }
                p2++;
            }
        }

        while (p1 < a1_end) {
            long long av1 = (long long)a_vals[p1];
            int k = a_colidx[p1];
            int b_start = b_rowptr[k];
            int b_end = b_rowptr[k + 1];
            for (int bj = b_start; bj < b_end; bj++) {
                res1[b_colidx[bj]] += av1 * (long long)b_vals[bj];
            }
            p1++;
        }

        while (p2 < a2_end) {
            long long av2 = (long long)a_vals[p2];
            int k = a_colidx[p2];
            int b_start = b_rowptr[k];
            int b_end = b_rowptr[k + 1];
            for (int bj = b_start; bj < b_end; bj++) {
                res2[b_colidx[bj]] += av2 * (long long)b_vals[bj];
            }
            p2++;
        }
    }

    if (i < rows_a) {
        long long *res = result + (long long)i * cols_b;
        int a_start = a_rowptr[i];
        int a_end = a_rowptr[i + 1];
        for (int p = a_start; p < a_end; p++) {
            long long av = (long long)a_vals[p];
            int k = a_colidx[p];
            int b_start = b_rowptr[k];
            int b_end = b_rowptr[k + 1];
            for (int bj = b_start; bj < b_end; bj++) {
                res[b_colidx[bj]] += av * (long long)b_vals[bj];
            }
        }
    }
}

/* Process a range of row pairs for parallel execution */
static void multiply_row_range(
    int row_start, int row_end, int cols_b,
    const int *a_rowptr, const int16_t *a_colidx, const int8_t *a_vals,
    const int *b_rowptr, const int16_t *b_colidx, const int8_t *b_vals,
    long long *result
) {
    /* Zero only our portion of the result */
    memset(result + (long long)row_start * cols_b, 0,
           (long long)(row_end - row_start) * cols_b * sizeof(long long));

    int i = row_start;
    /* Align to even boundary if needed */
    if (i & 1) {
        long long *res = result + (long long)i * cols_b;
        int a_start = a_rowptr[i];
        int a_end = a_rowptr[i + 1];
        for (int p = a_start; p < a_end; p++) {
            long long av = (long long)a_vals[p];
            int k = a_colidx[p];
            int b_start = b_rowptr[k];
            int b_end = b_rowptr[k + 1];
            for (int bj = b_start; bj < b_end; bj++) {
                res[b_colidx[bj]] += av * (long long)b_vals[bj];
            }
        }
        i++;
    }

    for (; i + 1 < row_end; i += 2) {
        long long * restrict res1 = result + (long long)i * cols_b;
        long long * restrict res2 = result + (long long)(i + 1) * cols_b;

        int a1_start = a_rowptr[i];
        int a1_end = a_rowptr[i + 1];
        int a2_start = a_rowptr[i + 1];
        int a2_end = a_rowptr[i + 2];

        int p1 = a1_start, p2 = a2_start;
        while (p1 < a1_end && p2 < a2_end) {
            int k1 = a_colidx[p1];
            int k2 = a_colidx[p2];
            if (k1 == k2) {
                long long av1 = (long long)a_vals[p1];
                long long av2 = (long long)a_vals[p2];
                int bs = b_rowptr[k1], be = b_rowptr[k1 + 1];
                for (int bj = bs; bj < be; bj++) {
                    int col = b_colidx[bj];
                    long long bv = (long long)b_vals[bj];
                    res1[col] += av1 * bv;
                    res2[col] += av2 * bv;
                }
                p1++; p2++;
            } else if (k1 < k2) {
                long long av1 = (long long)a_vals[p1];
                int bs = b_rowptr[k1], be = b_rowptr[k1 + 1];
                for (int bj = bs; bj < be; bj++)
                    res1[b_colidx[bj]] += av1 * (long long)b_vals[bj];
                p1++;
            } else {
                long long av2 = (long long)a_vals[p2];
                int bs = b_rowptr[k2], be = b_rowptr[k2 + 1];
                for (int bj = bs; bj < be; bj++)
                    res2[b_colidx[bj]] += av2 * (long long)b_vals[bj];
                p2++;
            }
        }
        while (p1 < a1_end) {
            long long av1 = (long long)a_vals[p1];
            int k = a_colidx[p1];
            int bs = b_rowptr[k], be = b_rowptr[k + 1];
            for (int bj = bs; bj < be; bj++)
                res1[b_colidx[bj]] += av1 * (long long)b_vals[bj];
            p1++;
        }
        while (p2 < a2_end) {
            long long av2 = (long long)a_vals[p2];
            int k = a_colidx[p2];
            int bs = b_rowptr[k], be = b_rowptr[k + 1];
            for (int bj = bs; bj < be; bj++)
                res2[b_colidx[bj]] += av2 * (long long)b_vals[bj];
            p2++;
        }
    }

    if (i < row_end) {
        long long *res = result + (long long)i * cols_b;
        int a_start = a_rowptr[i];
        int a_end = a_rowptr[i + 1];
        for (int p = a_start; p < a_end; p++) {
            long long av = (long long)a_vals[p];
            int k = a_colidx[p];
            int bs = b_rowptr[k], be = b_rowptr[k + 1];
            for (int bj = bs; bj < be; bj++)
                res[b_colidx[bj]] += av * (long long)b_vals[bj];
        }
    }
}

/* Batch multiply — serial version with C-side timing */
void sparse_matmul_batch(
    int num_cases,
    const int *all_rows_a,
    const int *all_cols_a,
    const int *all_cols_b,
    const int **all_a_rowptr,
    const int16_t **all_a_colidx,
    const int8_t **all_a_vals,
    const int **all_b_rowptr,
    const int16_t **all_b_colidx,
    const int8_t **all_b_vals,
    long long **all_result,
    double *latencies_ns
) {
    mach_timebase_info_data_t tb;
    mach_timebase_info(&tb);
    double ns_per_tick = (double)tb.numer / (double)tb.denom;

    for (int t = 0; t < num_cases; t++) {
        uint64_t start = mach_absolute_time();

        multiply_single(
            all_rows_a[t], all_cols_b[t],
            all_a_rowptr[t], all_a_colidx[t], all_a_vals[t],
            all_b_rowptr[t], all_b_colidx[t], all_b_vals[t],
            all_result[t]
        );

        uint64_t end = mach_absolute_time();
        latencies_ns[t] = (double)(end - start) * ns_per_tick;
    }
}

/* Parallel batch multiply using GCD dispatch_apply */
void sparse_matmul_batch_parallel(
    int num_cases,
    const int *all_rows_a,
    const int *all_cols_a,
    const int *all_cols_b,
    const int **all_a_rowptr,
    const int16_t **all_a_colidx,
    const int8_t **all_a_vals,
    const int **all_b_rowptr,
    const int16_t **all_b_colidx,
    const int8_t **all_b_vals,
    long long **all_result,
    double *latencies_ns,
    int num_threads
) {
    mach_timebase_info_data_t tb;
    mach_timebase_info(&tb);
    double ns_per_tick = (double)tb.numer / (double)tb.denom;

    dispatch_queue_t queue = dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0);

    for (int t = 0; t < num_cases; t++) {
        int rows_a = all_rows_a[t];
        int cols_b = all_cols_b[t];
        const int *a_rowptr = all_a_rowptr[t];
        const int16_t *a_colidx = all_a_colidx[t];
        const int8_t *a_vals = all_a_vals[t];
        const int *b_rowptr = all_b_rowptr[t];
        const int16_t *b_colidx = all_b_colidx[t];
        const int8_t *b_vals = all_b_vals[t];
        long long *result = all_result[t];

        /* Only parallelize large enough matrices */
        if (rows_a < 100 || num_threads <= 1) {
            uint64_t start = mach_absolute_time();
            multiply_single(rows_a, cols_b, a_rowptr, a_colidx, a_vals,
                          b_rowptr, b_colidx, b_vals, result);
            uint64_t end = mach_absolute_time();
            latencies_ns[t] = (double)(end - start) * ns_per_tick;
            continue;
        }

        int nt = num_threads;
        if (nt > rows_a / 50) nt = rows_a / 50;
        if (nt < 1) nt = 1;

        int rows_per_thread = (rows_a + nt - 1) / nt;

        uint64_t start = mach_absolute_time();

        dispatch_apply(nt, queue, ^(size_t tid) {
            int rs = (int)tid * rows_per_thread;
            int re = rs + rows_per_thread;
            if (re > rows_a) re = rows_a;
            if (rs < re) {
                multiply_row_range(rs, re, cols_b,
                                 a_rowptr, a_colidx, a_vals,
                                 b_rowptr, b_colidx, b_vals, result);
            }
        });

        uint64_t end = mach_absolute_time();
        latencies_ns[t] = (double)(end - start) * ns_per_tick;
    }
}
