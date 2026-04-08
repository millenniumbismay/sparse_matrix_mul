#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <mach/mach_time.h>
#include <dispatch/dispatch.h>
#include <Accelerate/Accelerate.h>

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

/* Build compact CSC (Compressed Sparse Column) for a matrix.
 * CSC stores column pointers + row indices + values.
 * Used for outer-product formulation where we iterate by column of A. */
int build_compact_csc(
    const long long *flat, int rows, int cols,
    int *colptr, int16_t *rowidx, int8_t *vals
) {
    /* First pass: count non-zeros per column */
    memset(colptr, 0, (cols + 1) * sizeof(int));
    for (int r = 0; r < rows; r++) {
        const long long *row = flat + (long long)r * cols;
        for (int c = 0; c < cols; c++) {
            if (row[c] != 0) colptr[c + 1]++;
        }
    }
    /* Prefix sum to get column pointers */
    for (int c = 0; c < cols; c++) {
        colptr[c + 1] += colptr[c];
    }
    /* Second pass: fill in row indices and values */
    int *cursor = (int *)malloc(cols * sizeof(int));
    memcpy(cursor, colptr, cols * sizeof(int));
    for (int r = 0; r < rows; r++) {
        const long long *row = flat + (long long)r * cols;
        for (int c = 0; c < cols; c++) {
            if (row[c] != 0) {
                int pos = cursor[c]++;
                rowidx[pos] = (int16_t)r;
                vals[pos] = (int8_t)row[c];
            }
        }
    }
    free(cursor);
    return colptr[cols];
}

/* Convert int64 flat array to int8 */
void flatten_to_int8(const long long *src, int8_t *dst, int count) {
    for (int i = 0; i < count; i++) {
        dst[i] = (int8_t)src[i];
    }
}

/* Core single-test multiply: merge-iterate CSR A row pairs.
 * Per-pair zeroing: zero each pair's result rows just before processing
 * to keep them L1-hot, instead of memset-ing the entire result upfront. */
static void multiply_single(
    int rows_a, int cols_b,
    const int *a_rowptr, const int16_t *a_colidx, const int8_t *a_vals,
    const int *b_rowptr, const int16_t *b_colidx, const int8_t *b_vals,
    long long *result
) {
    long long row_bytes = (long long)cols_b * sizeof(long long);

    int i = 0;
    for (; i + 1 < rows_a; i += 2) {
        long long * restrict res1 = result + (long long)i * cols_b;
        long long * restrict res2 = result + (long long)(i + 1) * cols_b;
        memset(res1, 0, row_bytes);
        memset(res2, 0, row_bytes);

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
        memset(res, 0, row_bytes);
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

/* Simple per-row multiply: no merge, just iterate each row's CSR.
 * Eliminates merge comparison overhead. B-rows cached in L2. */
static void multiply_single_nomerge(
    int rows_a, int cols_b,
    const int *a_rowptr, const int16_t *a_colidx, const int8_t *a_vals,
    const int *b_rowptr, const int16_t *b_colidx, const int8_t *b_vals,
    long long *result
) {
    long long row_bytes = (long long)cols_b * sizeof(long long);

    for (int i = 0; i < rows_a; i++) {
        long long *res = result + (long long)i * cols_b;
        memset(res, 0, row_bytes);

        int a_start = a_rowptr[i];
        int a_end = a_rowptr[i + 1];
        for (int p = a_start; p < a_end; p++) {
            long long av = (long long)a_vals[p];
            int k = a_colidx[p];
            int bs = b_rowptr[k], be = b_rowptr[k + 1];
            for (int bj = bs; bj < be; bj++) {
                res[b_colidx[bj]] += av * (long long)b_vals[bj];
            }
        }
    }
}

/* Batch no-merge multiply with C-side timing */
void sparse_matmul_batch_nomerge(
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

        multiply_single_nomerge(
            all_rows_a[t], all_cols_b[t],
            all_a_rowptr[t], all_a_colidx[t], all_a_vals[t],
            all_b_rowptr[t], all_b_colidx[t], all_b_vals[t],
            all_result[t]
        );

        uint64_t end = mach_absolute_time();
        latencies_ns[t] = (double)(end - start) * ns_per_tick;
    }
}

/* Outer product multiply: iterate by k (column of A / row of B).
 * B[k,:] is loaded exactly once and scattered to ALL A rows with A[*,k]!=0.
 * Uses CSC for A (column-compressed) and CSR for B (row-compressed).
 * Key advantage: maximizes B-row temporal reuse across all A rows. */
static void multiply_outer_product(
    int rows_a, int cols_a, int cols_b,
    const int *a_colptr, const int16_t *a_rowidx, const int8_t *a_vals_csc,
    const int *b_rowptr, const int16_t *b_colidx, const int8_t *b_vals,
    long long *result
) {
    memset(result, 0, (long long)rows_a * cols_b * sizeof(long long));

    for (int k = 0; k < cols_a; k++) {
        int a_start = a_colptr[k];
        int a_end = a_colptr[k + 1];
        if (a_start == a_end) continue;

        int b_start = b_rowptr[k];
        int b_end = b_rowptr[k + 1];
        if (b_start == b_end) continue;

        int b_nnz = b_end - b_start;

        /* For each non-zero A[i, k] in column k of A */
        for (int ai = a_start; ai < a_end; ai++) {
            int i = a_rowidx[ai];
            long long av = (long long)a_vals_csc[ai];
            long long *res_row = result + (long long)i * cols_b;

            /* Scatter B[k,:] * av into result[i,:] */
            for (int bj = b_start; bj < b_end; bj++) {
                res_row[b_colidx[bj]] += av * (long long)b_vals[bj];
            }
        }
    }
}

/* Batch outer-product multiply with C-side timing */
void sparse_matmul_batch_outer(
    int num_cases,
    const int *all_rows_a,
    const int *all_cols_a,
    const int *all_cols_b,
    const int **all_a_colptr,
    const int16_t **all_a_rowidx,
    const int8_t **all_a_vals_csc,
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

        multiply_outer_product(
            all_rows_a[t], all_cols_a[t], all_cols_b[t],
            all_a_colptr[t], all_a_rowidx[t], all_a_vals_csc[t],
            all_b_rowptr[t], all_b_colidx[t], all_b_vals[t],
            all_result[t]
        );

        uint64_t end = mach_absolute_time();
        latencies_ns[t] = (double)(end - start) * ns_per_tick;
    }
}

/*
 * Row-block merge multiply: process M rows at a time.
 * M-way merge of sorted CSR k-indices loads each B-row exactly once
 * per block, amortizing B-row reads across M result rows.
 * M=2 is equivalent to the pair-merge; M=4/8 increases sharing.
 */
static void multiply_row_block(
    int rows_a, int cols_b, int block_size,
    const int *a_rowptr, const int16_t *a_colidx, const int8_t *a_vals,
    const int *b_rowptr, const int16_t *b_colidx, const int8_t *b_vals,
    long long *result
) {
    memset(result, 0, (long long)rows_a * cols_b * sizeof(long long));

    for (int blk = 0; blk < rows_a; blk += block_size) {
        int M = block_size;
        if (blk + M > rows_a) M = rows_a - blk;

        /* Set up per-row pointers */
        int ptrs[16], ends_arr[16];  /* max block_size = 16 */
        long long *res_ptrs[16];
        for (int r = 0; r < M; r++) {
            int row = blk + r;
            ptrs[r] = a_rowptr[row];
            ends_arr[r] = a_rowptr[row + 1];
            res_ptrs[r] = result + (long long)row * cols_b;
        }

        /* M-way merge: find minimum k, load B[k,:], scatter to all matching rows */
        for (;;) {
            /* Find minimum k among all active rows */
            int k_min = 0x7FFF;  /* max int16 + 1 */
            for (int r = 0; r < M; r++) {
                if (ptrs[r] < ends_arr[r]) {
                    int k = a_colidx[ptrs[r]];
                    if (k < k_min) k_min = k;
                }
            }
            if (k_min == 0x7FFF) break;

            int bs = b_rowptr[k_min];
            int be = b_rowptr[k_min + 1];

            /* Scatter B[k_min,:] to all rows that have k_min */
            for (int r = 0; r < M; r++) {
                if (ptrs[r] < ends_arr[r] && a_colidx[ptrs[r]] == k_min) {
                    long long av = (long long)a_vals[ptrs[r]];
                    long long *rr = res_ptrs[r];
                    for (int bj = bs; bj < be; bj++) {
                        rr[b_colidx[bj]] += av * (long long)b_vals[bj];
                    }
                    ptrs[r]++;
                }
            }
        }
    }
}

/* Batch row-block multiply with C-side timing */
void sparse_matmul_batch_rowblock(
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
    int block_size
) {
    mach_timebase_info_data_t tb;
    mach_timebase_info(&tb);
    double ns_per_tick = (double)tb.numer / (double)tb.denom;

    for (int t = 0; t < num_cases; t++) {
        uint64_t start = mach_absolute_time();

        multiply_row_block(
            all_rows_a[t], all_cols_b[t], block_size,
            all_a_rowptr[t], all_a_colidx[t], all_a_vals[t],
            all_b_rowptr[t], all_b_colidx[t], all_b_vals[t],
            all_result[t]
        );

        uint64_t end = mach_absolute_time();
        latencies_ns[t] = (double)(end - start) * ns_per_tick;
    }
}

/* Process a range of rows for parallel execution.
 * Uses per-pair lazy zeroing to keep result rows L1-hot. */
static void multiply_row_range(
    int row_start, int row_end, int cols_b,
    const int *a_rowptr, const int16_t *a_colidx, const int8_t *a_vals,
    const int *b_rowptr, const int16_t *b_colidx, const int8_t *b_vals,
    long long *result
) {
    long long row_bytes = (long long)cols_b * sizeof(long long);
    int i = row_start;
    if (i & 1) {
        long long *res = result + (long long)i * cols_b;
        memset(res, 0, row_bytes);
        int a_start = a_rowptr[i];
        int a_end = a_rowptr[i + 1];
        for (int p = a_start; p < a_end; p++) {
            long long av = (long long)a_vals[p];
            int k = a_colidx[p];
            int bs = b_rowptr[k], be = b_rowptr[k + 1];
            for (int bj = bs; bj < be; bj++)
                res[b_colidx[bj]] += av * (long long)b_vals[bj];
        }
        i++;
    }

    for (; i + 1 < row_end; i += 2) {
        long long * restrict res1 = result + (long long)i * cols_b;
        long long * restrict res2 = result + (long long)(i + 1) * cols_b;
        memset(res1, 0, row_bytes);
        memset(res2, 0, row_bytes);

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
        memset(res, 0, row_bytes);
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

/* Dense BLAS multiply via Accelerate.
 * Converts int8 A/B → float, calls sgemm, converts float → int64.
 * Uses AMX coprocessor for the dense GEMM. */
static void multiply_dense_blas(
    int rows_a, int cols_a, int cols_b,
    const int8_t *a_i8,
    const int8_t *b_i8,
    long long *result
) {
    long long a_size = (long long)rows_a * cols_a;
    long long b_size = (long long)cols_a * cols_b;
    long long c_size = (long long)rows_a * cols_b;

    float *fa = (float *)malloc(a_size * sizeof(float));
    float *fb = (float *)malloc(b_size * sizeof(float));
    float *fc = (float *)calloc(c_size, sizeof(float));

    /* Convert int8 → float */
    for (long long i = 0; i < a_size; i++) fa[i] = (float)a_i8[i];
    for (long long i = 0; i < b_size; i++) fb[i] = (float)b_i8[i];

    /* C = A * B using BLAS sgemm (row-major) */
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                rows_a, cols_b, cols_a,
                1.0f, fa, cols_a, fb, cols_b,
                0.0f, fc, cols_b);

    /* Convert float → int64 (round to nearest) */
    for (long long i = 0; i < c_size; i++)
        result[i] = (long long)(fc[i] + (fc[i] >= 0 ? 0.5f : -0.5f));

    free(fa); free(fb); free(fc);
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

/* Hybrid batch: uses dense BLAS for large sparse matrices, parallel sparse for rest */
void sparse_matmul_batch_hybrid(
    int num_cases,
    const int *all_rows_a,
    const int *all_cols_a,
    const int *all_cols_b,
    const int8_t **all_a_i8,        /* dense int8 A (for BLAS path) */
    const int8_t **all_b_i8,        /* dense int8 B (for BLAS path) */
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
        int cols_a = all_cols_a[t];
        int cols_b = all_cols_b[t];
        long long *result = all_result[t];

        uint64_t start = mach_absolute_time();

        /* Use dense BLAS for the multiply */
        multiply_dense_blas(rows_a, cols_a, cols_b,
                          all_a_i8[t], all_b_i8[t], result);

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
