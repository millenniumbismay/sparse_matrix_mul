#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <mach/mach_time.h>
#include <dispatch/dispatch.h>
#include <Accelerate/Accelerate.h>
#include <arm_neon.h>

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

/* Core single-test multiply: independent row processing.
 * Process each row independently — simpler loop with no merge branches.
 * Per-row zeroing keeps result rows L1-hot.
 *
 * Experiment 22 finding: merge-iterate's 3-way branch misprediction cost
 * (~990 cycles/pair) exceeds B-row sharing benefit (~605 cycles/pair) at
 * our density/size. Simpler code = better branch prediction = equal or
 * faster performance. */
static void multiply_single(
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

/* Dense B multi-axpy with int16 intermediate + int32 output.
 * Each 4-product sum ≤ 400, so int16 acc holds ~80 iterations (32000/400).
 * int16 NEON ops process 8 values per instruction (vs 4 for int32) → 2x throughput.
 * Flush int16→int32 every 80 iterations. */
/* Process one tile of TILE_W columns with register-resident int16 accumulator.
 * TILE_W = 128 uses 16 NEON registers for the accumulator. */
#define TILE_W 128
#define TILE_VECS (TILE_W / 8)  /* 16 int16x8 vectors */
#define FLUSH_INTERVAL 80

static inline void process_tile_neon(
    int tile_start, int tile_end,
    int a_start, int a_end,
    const int16_t * restrict a_colidx,
    const int8_t * restrict a_vals,
    const int8_t * restrict b_dense,
    int cols_b,
    int32_t * restrict res
) {
    int tw = tile_end - tile_start;
    int tw_vec = tw / 8;
    int tw_rem = tw & 7;

    /* Register-resident accumulator: up to 16 int16x8 vectors */
    int16x8_t acc[TILE_VECS];
    for (int v = 0; v < tw_vec; v++) acc[v] = vdupq_n_s16(0);

    int iters_since_flush = 0;
    int p = a_start;

    /* 4-way unrolled A entries */
    for (; p + 3 < a_end; p += 4) {
        int8x8_t av0 = vdup_n_s8(a_vals[p]);
        int8x8_t av1 = vdup_n_s8(a_vals[p+1]);
        int8x8_t av2 = vdup_n_s8(a_vals[p+2]);
        int8x8_t av3 = vdup_n_s8(a_vals[p+3]);
        const int8_t *br0 = b_dense + (long long)a_colidx[p]   * cols_b + tile_start;
        const int8_t *br1 = b_dense + (long long)a_colidx[p+1] * cols_b + tile_start;
        const int8_t *br2 = b_dense + (long long)a_colidx[p+2] * cols_b + tile_start;
        const int8_t *br3 = b_dense + (long long)a_colidx[p+3] * cols_b + tile_start;

        for (int v = 0; v < tw_vec; v++) {
            int off = v * 8;
            acc[v] = vmlal_s8(acc[v], av0, vld1_s8(br0 + off));
            acc[v] = vmlal_s8(acc[v], av1, vld1_s8(br1 + off));
            acc[v] = vmlal_s8(acc[v], av2, vld1_s8(br2 + off));
            acc[v] = vmlal_s8(acc[v], av3, vld1_s8(br3 + off));
        }

        iters_since_flush++;
        if (iters_since_flush >= FLUSH_INTERVAL) {
            int32_t *rp = res + tile_start;
            for (int v = 0; v < tw_vec; v++) {
                int off = v * 8;
                int32x4_t lo = vld1q_s32(rp + off);
                int32x4_t hi = vld1q_s32(rp + off + 4);
                lo = vaddw_s16(lo, vget_low_s16(acc[v]));
                hi = vaddw_s16(hi, vget_high_s16(acc[v]));
                vst1q_s32(rp + off, lo);
                vst1q_s32(rp + off + 4, hi);
                acc[v] = vdupq_n_s16(0);
            }
            iters_since_flush = 0;
        }
    }

    /* Scalar remainder for leftover A entries */
    for (; p < a_end; p++) {
        int8x8_t av = vdup_n_s8(a_vals[p]);
        const int8_t *br = b_dense + (long long)a_colidx[p] * cols_b + tile_start;
        for (int v = 0; v < tw_vec; v++) {
            acc[v] = vmlal_s8(acc[v], av, vld1_s8(br + v * 8));
        }
        iters_since_flush++;
    }

    /* Final flush: widen int16 → int32 and add to result */
    if (iters_since_flush > 0) {
        int32_t *rp = res + tile_start;
        for (int v = 0; v < tw_vec; v++) {
            int off = v * 8;
            int32x4_t lo = vld1q_s32(rp + off);
            int32x4_t hi = vld1q_s32(rp + off + 4);
            lo = vaddw_s16(lo, vget_low_s16(acc[v]));
            hi = vaddw_s16(hi, vget_high_s16(acc[v]));
            vst1q_s32(rp + off, lo);
            vst1q_s32(rp + off + 4, hi);
        }
    }

    /* Handle remaining columns (< 8) with scalar */
    if (tw_rem > 0) {
        int j_start = tile_start + tw_vec * 8;
        int16_t sc_acc[8] = {0};
        int sp = a_start;
        int sc_flush = 0;
        for (; sp < a_end; sp++) {
            int16_t av = (int16_t)a_vals[sp];
            const int8_t *br = b_dense + (long long)a_colidx[sp] * cols_b;
            for (int j = 0; j < tw_rem; j++)
                sc_acc[j] += av * (int16_t)br[j_start + j];
            sc_flush++;
            if (sc_flush >= FLUSH_INTERVAL * 4) {
                for (int j = 0; j < tw_rem; j++) {
                    res[j_start + j] += (int32_t)sc_acc[j];
                    sc_acc[j] = 0;
                }
                sc_flush = 0;
            }
        }
        for (int j = 0; j < tw_rem; j++)
            res[j_start + j] += (int32_t)sc_acc[j];
    }
}

static void multiply_dense_axpy(
    int rows_a, int cols_a, int cols_b,
    const int * restrict a_rowptr, const int16_t * restrict a_colidx,
    const int8_t * restrict a_vals,
    const int8_t * restrict b_dense,
    int32_t * restrict result
) {
    for (int i = 0; i < rows_a; i++) {
        int32_t * restrict res = result + (long long)i * cols_b;
        int a_start = a_rowptr[i];
        int a_end = a_rowptr[i + 1];
        memset(res, 0, (long long)cols_b * sizeof(int32_t));
        if (a_start == a_end) continue;

        /* Process columns in tiles with register-resident accumulator */
        int j;
        for (j = 0; j + TILE_W <= cols_b; j += TILE_W) {
            process_tile_neon(j, j + TILE_W, a_start, a_end,
                            a_colidx, a_vals, b_dense, cols_b, res);
        }
        if (j < cols_b) {
            process_tile_neon(j, cols_b, a_start, a_end,
                            a_colidx, a_vals, b_dense, cols_b, res);
        }
    }
}

/* Dense multi-axpy row-range for parallel dispatch — NEON register-tiled */
static void multiply_dense_axpy_range(
    int row_start, int row_end, int cols_a, int cols_b,
    const int * restrict a_rowptr, const int16_t * restrict a_colidx,
    const int8_t * restrict a_vals,
    const int8_t * restrict b_dense,
    int32_t * restrict result
) {
    for (int i = row_start; i < row_end; i++) {
        int32_t * restrict res = result + (long long)i * cols_b;
        int a_start = a_rowptr[i];
        int a_end = a_rowptr[i + 1];
        memset(res, 0, (long long)cols_b * sizeof(int32_t));
        if (a_start == a_end) continue;

        int j;
        for (j = 0; j + TILE_W <= cols_b; j += TILE_W) {
            process_tile_neon(j, j + TILE_W, a_start, a_end,
                            a_colidx, a_vals, b_dense, cols_b, res);
        }
        if (j < cols_b) {
            process_tile_neon(j, cols_b, a_start, a_end,
                            a_colidx, a_vals, b_dense, cols_b, res);
        }
    }
}

/* Experiment 22: Row-reordered nomerge with B-data L1 locality.
 * Sort A rows by their first non-zero column so consecutive processed
 * rows access similar B-rows, improving L1 cache reuse.
 * Uses simple single-row processing (no merge) for tight inner loop. */
static void multiply_single_reordered(
    int rows_a, int cols_b,
    const int *a_rowptr, const int16_t *a_colidx, const int8_t *a_vals,
    const int *b_rowptr, const int16_t *b_colidx, const int8_t *b_vals,
    long long *result,
    const int *row_order  /* permutation: process row_order[0], row_order[1], ... */
) {
    long long row_bytes = (long long)cols_b * sizeof(long long);

    for (int idx = 0; idx < rows_a; idx++) {
        int i = row_order[idx];
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

/* Build row ordering sorted by median non-zero column for B-data locality */
void build_row_order(
    int rows_a,
    const int *a_rowptr, const int16_t *a_colidx,
    int *row_order
) {
    /* Compute sort key for each row: median non-zero column index */
    int *keys = (int *)malloc(rows_a * sizeof(int));
    for (int i = 0; i < rows_a; i++) {
        row_order[i] = i;
        int start = a_rowptr[i], end = a_rowptr[i + 1];
        if (start < end) {
            int mid = (start + end) / 2;
            keys[i] = a_colidx[mid];  /* median column */
        } else {
            keys[i] = 0;
        }
    }
    /* Simple insertion sort (good enough for <1000 rows) */
    for (int i = 1; i < rows_a; i++) {
        int key = keys[i], val = row_order[i];
        int j = i - 1;
        while (j >= 0 && keys[j] > key) {
            keys[j + 1] = keys[j];
            row_order[j + 1] = row_order[j];
            j--;
        }
        keys[j + 1] = key;
        row_order[j + 1] = val;
    }
    free(keys);
}

/* Batch reordered multiply with C-side timing */
void sparse_matmul_batch_reordered(
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
    const int **all_row_orders  /* pre-computed row orderings */
) {
    mach_timebase_info_data_t tb;
    mach_timebase_info(&tb);
    double ns_per_tick = (double)tb.numer / (double)tb.denom;

    for (int t = 0; t < num_cases; t++) {
        uint64_t start = mach_absolute_time();

        multiply_single_reordered(
            all_rows_a[t], all_cols_b[t],
            all_a_rowptr[t], all_a_colidx[t], all_a_vals[t],
            all_b_rowptr[t], all_b_colidx[t], all_b_vals[t],
            all_result[t],
            all_row_orders[t]
        );

        uint64_t end = mach_absolute_time();
        latencies_ns[t] = (double)(end - start) * ns_per_tick;
    }
}

/* Experiment 22: Pre-classified merge-iterate (PCMI).
 * For each row pair, precompute the merge classification:
 *   - "both" entries: shared k values (B-row loaded once, scattered to 2 rows)
 *   - "first" entries: k only in row i (scatter to row i only)
 *   - "second" entries: k only in row i+1 (scatter to row i+1 only)
 * Then process each group in a tight loop with NO conditional branches.
 * Eliminates 3-way branch misprediction while keeping B-row temporal reuse. */
static void multiply_single_pcmi(
    int rows_a, int cols_b,
    const int *a_rowptr, const int16_t *a_colidx, const int8_t *a_vals,
    const int *b_rowptr, const int16_t *b_colidx, const int8_t *b_vals,
    long long *result
) {
    long long row_bytes = (long long)cols_b * sizeof(long long);

    /* Pre-allocated classification buffers (sized for worst case per row) */
    int max_nnz = 0;
    for (int i = 0; i < rows_a; i++) {
        int nnz = a_rowptr[i + 1] - a_rowptr[i];
        if (nnz > max_nnz) max_nnz = nnz;
    }

    int16_t *both_k   = (int16_t *)malloc(max_nnz * sizeof(int16_t));
    int8_t  *both_av1  = (int8_t  *)malloc(max_nnz);
    int8_t  *both_av2  = (int8_t  *)malloc(max_nnz);
    int16_t *first_k   = (int16_t *)malloc(max_nnz * sizeof(int16_t));
    int8_t  *first_av  = (int8_t  *)malloc(max_nnz);
    int16_t *second_k  = (int16_t *)malloc(max_nnz * sizeof(int16_t));
    int8_t  *second_av = (int8_t  *)malloc(max_nnz);

    int i = 0;
    for (; i + 1 < rows_a; i += 2) {
        long long * restrict res1 = result + (long long)i * cols_b;
        long long * restrict res2 = result + (long long)(i + 1) * cols_b;
        memset(res1, 0, row_bytes);
        memset(res2, 0, row_bytes);

        int a1_start = a_rowptr[i], a1_end = a_rowptr[i + 1];
        int a2_start = a_rowptr[i + 1], a2_end = a_rowptr[i + 2];

        /* Phase 1: Classify by merging sorted A-row CSR entries */
        int n_both = 0, n_first = 0, n_second = 0;
        int p1 = a1_start, p2 = a2_start;
        while (p1 < a1_end && p2 < a2_end) {
            int k1 = a_colidx[p1], k2 = a_colidx[p2];
            if (k1 == k2) {
                both_k[n_both] = k1;
                both_av1[n_both] = a_vals[p1];
                both_av2[n_both] = a_vals[p2];
                n_both++;
                p1++; p2++;
            } else if (k1 < k2) {
                first_k[n_first] = k1;
                first_av[n_first] = a_vals[p1];
                n_first++;
                p1++;
            } else {
                second_k[n_second] = k2;
                second_av[n_second] = a_vals[p2];
                n_second++;
                p2++;
            }
        }
        while (p1 < a1_end) {
            first_k[n_first] = a_colidx[p1];
            first_av[n_first] = a_vals[p1];
            n_first++;
            p1++;
        }
        while (p2 < a2_end) {
            second_k[n_second] = a_colidx[p2];
            second_av[n_second] = a_vals[p2];
            n_second++;
            p2++;
        }

        /* Phase 2a: Process "both" entries — 1 B-load, 2 scatter streams */
        for (int j = 0; j < n_both; j++) {
            int k = both_k[j];
            long long av1 = (long long)both_av1[j];
            long long av2 = (long long)both_av2[j];
            int bs = b_rowptr[k], be = b_rowptr[k + 1];
            for (int bj = bs; bj < be; bj++) {
                int col = b_colidx[bj];
                long long bv = (long long)b_vals[bj];
                res1[col] += av1 * bv;
                res2[col] += av2 * bv;
            }
        }

        /* Phase 2b: Process "first-only" entries — scatter to res1 */
        for (int j = 0; j < n_first; j++) {
            int k = first_k[j];
            long long av1 = (long long)first_av[j];
            int bs = b_rowptr[k], be = b_rowptr[k + 1];
            for (int bj = bs; bj < be; bj++) {
                res1[b_colidx[bj]] += av1 * (long long)b_vals[bj];
            }
        }

        /* Phase 2c: Process "second-only" entries — scatter to res2 */
        for (int j = 0; j < n_second; j++) {
            int k = second_k[j];
            long long av2 = (long long)second_av[j];
            int bs = b_rowptr[k], be = b_rowptr[k + 1];
            for (int bj = bs; bj < be; bj++) {
                res2[b_colidx[bj]] += av2 * (long long)b_vals[bj];
            }
        }
    }

    /* Handle odd last row */
    if (i < rows_a) {
        long long *res = result + (long long)i * cols_b;
        memset(res, 0, row_bytes);
        int a_start = a_rowptr[i], a_end = a_rowptr[i + 1];
        for (int p = a_start; p < a_end; p++) {
            long long av = (long long)a_vals[p];
            int k = a_colidx[p];
            int bs = b_rowptr[k], be = b_rowptr[k + 1];
            for (int bj = bs; bj < be; bj++) {
                res[b_colidx[bj]] += av * (long long)b_vals[bj];
            }
        }
    }

    free(both_k); free(both_av1); free(both_av2);
    free(first_k); free(first_av);
    free(second_k); free(second_av);
}

/* Experiment 22: int32 accumulator with deferred widening.
 * Accumulate into int32 (4 bytes) instead of int64 (8 bytes):
 * - memset is 50% cheaper (4KB vs 8KB per row for 1000 cols)
 * - result rows fit better in L1 (2 rows × 4KB = 8KB vs 16KB)
 * - widen to int64 at end of each row (sequential, NEON-friendly)
 * Products fit int16 (max |10*10|=100), accumulation fits int32
 * (max 1000 terms * 100 = 100000 << 2^31). */
static void multiply_single_i32(
    int rows_a, int cols_b,
    const int *a_rowptr, const int16_t *a_colidx, const int8_t *a_vals,
    const int *b_rowptr, const int16_t *b_colidx, const int8_t *b_vals,
    long long *result
) {
    /* Scratch int32 accumulators — two rows at a time */
    int32_t *acc = (int32_t *)malloc((long long)2 * cols_b * sizeof(int32_t));
    int32_t row_bytes32 = cols_b * sizeof(int32_t);

    int i = 0;
    for (; i + 1 < rows_a; i += 2) {
        int32_t * restrict acc1 = acc;
        int32_t * restrict acc2 = acc + cols_b;
        memset(acc1, 0, row_bytes32);
        memset(acc2, 0, row_bytes32);

        int a1_start = a_rowptr[i];
        int a1_end = a_rowptr[i + 1];
        int a2_start = a_rowptr[i + 1];
        int a2_end = a_rowptr[i + 2];

        int p1 = a1_start, p2 = a2_start;
        while (p1 < a1_end && p2 < a2_end) {
            int k1 = a_colidx[p1];
            int k2 = a_colidx[p2];
            if (k1 == k2) {
                int32_t av1 = (int32_t)a_vals[p1];
                int32_t av2 = (int32_t)a_vals[p2];
                int b_start = b_rowptr[k1];
                int b_end = b_rowptr[k1 + 1];
                for (int bj = b_start; bj < b_end; bj++) {
                    int col = b_colidx[bj];
                    int32_t bv = (int32_t)b_vals[bj];
                    acc1[col] += av1 * bv;
                    acc2[col] += av2 * bv;
                }
                p1++; p2++;
            } else if (k1 < k2) {
                int32_t av1 = (int32_t)a_vals[p1];
                int b_start = b_rowptr[k1];
                int b_end = b_rowptr[k1 + 1];
                for (int bj = b_start; bj < b_end; bj++) {
                    acc1[b_colidx[bj]] += av1 * (int32_t)b_vals[bj];
                }
                p1++;
            } else {
                int32_t av2 = (int32_t)a_vals[p2];
                int b_start = b_rowptr[k2];
                int b_end = b_rowptr[k2 + 1];
                for (int bj = b_start; bj < b_end; bj++) {
                    acc2[b_colidx[bj]] += av2 * (int32_t)b_vals[bj];
                }
                p2++;
            }
        }

        while (p1 < a1_end) {
            int32_t av1 = (int32_t)a_vals[p1];
            int k = a_colidx[p1];
            int b_start = b_rowptr[k];
            int b_end = b_rowptr[k + 1];
            for (int bj = b_start; bj < b_end; bj++) {
                acc1[b_colidx[bj]] += av1 * (int32_t)b_vals[bj];
            }
            p1++;
        }

        while (p2 < a2_end) {
            int32_t av2 = (int32_t)a_vals[p2];
            int k = a_colidx[p2];
            int b_start = b_rowptr[k];
            int b_end = b_rowptr[k + 1];
            for (int bj = b_start; bj < b_end; bj++) {
                acc2[b_colidx[bj]] += av2 * (int32_t)b_vals[bj];
            }
            p2++;
        }

        /* Widen int32 → int64 (sequential, NEON auto-vectorizable) */
        long long * restrict res1 = result + (long long)i * cols_b;
        long long * restrict res2 = result + (long long)(i + 1) * cols_b;
        for (int j = 0; j < cols_b; j++) res1[j] = (long long)acc1[j];
        for (int j = 0; j < cols_b; j++) res2[j] = (long long)acc2[j];
    }

    if (i < rows_a) {
        int32_t *acc1 = acc;
        memset(acc1, 0, row_bytes32);
        int a_start = a_rowptr[i];
        int a_end = a_rowptr[i + 1];
        for (int p = a_start; p < a_end; p++) {
            int32_t av = (int32_t)a_vals[p];
            int k = a_colidx[p];
            int b_start = b_rowptr[k];
            int b_end = b_rowptr[k + 1];
            for (int bj = b_start; bj < b_end; bj++) {
                acc1[b_colidx[bj]] += av * (int32_t)b_vals[bj];
            }
        }
        long long *res = result + (long long)i * cols_b;
        for (int j = 0; j < cols_b; j++) res[j] = (long long)acc1[j];
    }

    free(acc);
}

/* Experiment 22b: No-merge single-row with int32 accumulator.
 * Simpler loop (no 3-way branch), better branch prediction.
 * Tests whether merge overhead > B-row sharing benefit at this level. */
static void multiply_single_nomerge_i32(
    int rows_a, int cols_b,
    const int *a_rowptr, const int16_t *a_colidx, const int8_t *a_vals,
    const int *b_rowptr, const int16_t *b_colidx, const int8_t *b_vals,
    long long *result
) {
    int32_t *acc = (int32_t *)malloc((long long)cols_b * sizeof(int32_t));
    int32_t row_bytes32 = cols_b * sizeof(int32_t);

    for (int i = 0; i < rows_a; i++) {
        memset(acc, 0, row_bytes32);

        int a_start = a_rowptr[i];
        int a_end = a_rowptr[i + 1];
        for (int p = a_start; p < a_end; p++) {
            int32_t av = (int32_t)a_vals[p];
            int k = a_colidx[p];
            int bs = b_rowptr[k], be = b_rowptr[k + 1];
            for (int bj = bs; bj < be; bj++) {
                acc[b_colidx[bj]] += av * (int32_t)b_vals[bj];
            }
        }

        /* Widen to int64 */
        long long *res = result + (long long)i * cols_b;
        for (int j = 0; j < cols_b; j++) res[j] = (long long)acc[j];
    }

    free(acc);
}

/* Diagnostic: memset-only version for cost breakdown analysis */
static void multiply_single_memset_only(
    int rows_a, int cols_b,
    const int *a_rowptr, const int16_t *a_colidx, const int8_t *a_vals,
    const int *b_rowptr, const int16_t *b_colidx, const int8_t *b_vals,
    long long *result
) {
    long long row_bytes = (long long)cols_b * sizeof(long long);
    for (int i = 0; i < rows_a; i++) {
        memset(result + (long long)i * cols_b, 0, row_bytes);
    }
}

/* Diagnostic: scatter-only version (no memset, wrong results) */
static void multiply_single_scatter_only(
    int rows_a, int cols_b,
    const int *a_rowptr, const int16_t *a_colidx, const int8_t *a_vals,
    const int *b_rowptr, const int16_t *b_colidx, const int8_t *b_vals,
    long long *result
) {
    int i = 0;
    for (; i + 1 < rows_a; i += 2) {
        long long * restrict res1 = result + (long long)i * cols_b;
        long long * restrict res2 = result + (long long)(i + 1) * cols_b;

        int a1_start = a_rowptr[i], a1_end = a_rowptr[i + 1];
        int a2_start = a_rowptr[i + 1], a2_end = a_rowptr[i + 2];

        int p1 = a1_start, p2 = a2_start;
        while (p1 < a1_end && p2 < a2_end) {
            int k1 = a_colidx[p1], k2 = a_colidx[p2];
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
    if (i < rows_a) {
        long long *res = result + (long long)i * cols_b;
        int a_start = a_rowptr[i], a_end = a_rowptr[i + 1];
        for (int p = a_start; p < a_end; p++) {
            long long av = (long long)a_vals[p];
            int k = a_colidx[p];
            int bs = b_rowptr[k], be = b_rowptr[k + 1];
            for (int bj = bs; bj < be; bj++)
                res[b_colidx[bj]] += av * (long long)b_vals[bj];
        }
    }
}

/* Batch diagnostic with C-side timing */
void sparse_matmul_batch_diag(
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
    int mode  /* 0=memset_only, 1=scatter_only */
) {
    mach_timebase_info_data_t tb;
    mach_timebase_info(&tb);
    double ns_per_tick = (double)tb.numer / (double)tb.denom;

    for (int t = 0; t < num_cases; t++) {
        uint64_t start = mach_absolute_time();

        if (mode == 0) {
            multiply_single_memset_only(
                all_rows_a[t], all_cols_b[t],
                all_a_rowptr[t], all_a_colidx[t], all_a_vals[t],
                all_b_rowptr[t], all_b_colidx[t], all_b_vals[t],
                all_result[t]
            );
        } else {
            multiply_single_scatter_only(
                all_rows_a[t], all_cols_b[t],
                all_a_rowptr[t], all_a_colidx[t], all_a_vals[t],
                all_b_rowptr[t], all_b_colidx[t], all_b_vals[t],
                all_result[t]
            );
        }

        uint64_t end = mach_absolute_time();
        latencies_ns[t] = (double)(end - start) * ns_per_tick;
    }
}

/*
 * Radix-sort-based multiply: collect scatter contributions into a buffer,
 * radix sort by column, then accumulate sequentially.
 * Converts O(nnz) random writes into O(nnz) sequential reads + O(cols_b) sequential writes.
 * Uses counting sort on column indices (O(n+k) where k=cols_b).
 */
static void multiply_single_radix(
    int rows_a, int cols_b,
    const int *a_rowptr, const int16_t *a_colidx, const int8_t *a_vals,
    const int *b_rowptr, const int16_t *b_colidx, const int8_t *b_vals,
    long long *result
) {
    /* Working buffers — sized for worst case per row.
     * At 30% density on 1000 cols, ~300 A entries * ~300 B entries = ~90K.
     * Use heap allocation for safety. */
    int max_scatter = 0;
    for (int i = 0; i < rows_a; i++) {
        int a_nnz = a_rowptr[i + 1] - a_rowptr[i];
        long long row_scatter = 0;
        for (int p = a_rowptr[i]; p < a_rowptr[i + 1]; p++) {
            int k = a_colidx[p];
            row_scatter += b_rowptr[k + 1] - b_rowptr[k];
        }
        if (row_scatter > max_scatter) max_scatter = (int)row_scatter;
    }

    int16_t *buf_col = (int16_t *)malloc(max_scatter * sizeof(int16_t));
    int16_t *buf_val = (int16_t *)malloc(max_scatter * sizeof(int16_t));
    int16_t *sort_col = (int16_t *)malloc(max_scatter * sizeof(int16_t));
    int16_t *sort_val = (int16_t *)malloc(max_scatter * sizeof(int16_t));
    int *counts = (int *)calloc(cols_b + 1, sizeof(int));

    long long row_bytes = (long long)cols_b * sizeof(long long);

    for (int i = 0; i < rows_a; i++) {
        long long *res = result + (long long)i * cols_b;
        memset(res, 0, row_bytes);

        /* Phase 1: Collect scatter contributions */
        int buf_len = 0;
        int a_start = a_rowptr[i];
        int a_end = a_rowptr[i + 1];
        for (int p = a_start; p < a_end; p++) {
            int16_t av = (int16_t)a_vals[p];
            int k = a_colidx[p];
            int bs = b_rowptr[k], be = b_rowptr[k + 1];
            for (int bj = bs; bj < be; bj++) {
                buf_col[buf_len] = b_colidx[bj];
                buf_val[buf_len] = av * (int16_t)b_vals[bj];
                buf_len++;
            }
        }

        if (buf_len == 0) continue;

        /* Phase 2: Counting sort by column */
        memset(counts, 0, (cols_b + 1) * sizeof(int));
        for (int j = 0; j < buf_len; j++) counts[buf_col[j] + 1]++;
        for (int j = 0; j < cols_b; j++) counts[j + 1] += counts[j];
        for (int j = 0; j < buf_len; j++) {
            int pos = counts[buf_col[j]]++;
            sort_col[pos] = buf_col[j];
            sort_val[pos] = buf_val[j];
        }

        /* Phase 3: Sequential accumulation */
        for (int j = 0; j < buf_len; j++) {
            res[sort_col[j]] += (long long)sort_val[j];
        }
    }

    free(buf_col); free(buf_val); free(sort_col); free(sort_val); free(counts);
}

/* Batch radix multiply with C-side timing */
void sparse_matmul_batch_radix(
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

        multiply_single_radix(
            all_rows_a[t], all_cols_b[t],
            all_a_rowptr[t], all_a_colidx[t], all_a_vals[t],
            all_b_rowptr[t], all_b_colidx[t], all_b_vals[t],
            all_result[t]
        );

        uint64_t end = mach_absolute_time();
        latencies_ns[t] = (double)(end - start) * ns_per_tick;
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
 * Simple per-row processing with per-row zeroing. */
static void multiply_row_range(
    int row_start, int row_end, int cols_b,
    const int *a_rowptr, const int16_t *a_colidx, const int8_t *a_vals,
    const int *b_rowptr, const int16_t *b_colidx, const int8_t *b_vals,
    long long *result
) {
    long long row_bytes = (long long)cols_b * sizeof(long long);

    for (int i = row_start; i < row_end; i++) {
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

/* Convert int8 array to float32 for BLAS */
void convert_i8_to_f32(const int8_t *src, float *dst, int count) {
    int i = 0;
    for (; i + 7 < count; i += 8) {
        int8x8_t v = vld1_s8(src + i);
        int16x8_t v16 = vmovl_s8(v);
        int32x4_t lo = vmovl_s16(vget_low_s16(v16));
        int32x4_t hi = vmovl_s16(vget_high_s16(v16));
        vst1q_f32(dst + i, vcvtq_f32_s32(lo));
        vst1q_f32(dst + i + 4, vcvtq_f32_s32(hi));
    }
    for (; i < count; i++) dst[i] = (float)src[i];
}

/* Dense BLAS multiply with pre-allocated float32 buffers.
 * Uses cblas_sgemm (AMX-accelerated on Apple Silicon). */
static void multiply_dense_blas_i32(
    int rows_a, int cols_a, int cols_b,
    const float *a_f32,
    const float *b_f32,
    float *c_f32,
    int32_t *result
) {
    long long c_size = (long long)rows_a * cols_b;

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                rows_a, cols_b, cols_a,
                1.0f, a_f32, cols_a, b_f32, cols_b,
                0.0f, c_f32, cols_b);

    /* Convert float32 → int32 with NEON round-to-nearest */
    long long i = 0;
    for (; i + 3 < c_size; i += 4) {
        float32x4_t fv = vld1q_f32(c_f32 + i);
        int32x4_t iv = vcvtnq_s32_f32(fv);
        vst1q_s32(result + i, iv);
    }
    for (; i < c_size; i++)
        result[i] = (int32_t)lroundf(c_f32[i]);
}

/* Batch BLAS multiply — serial */
void sparse_matmul_batch_blas(
    int num_cases,
    const int *all_rows_a,
    const int *all_cols_a,
    const int *all_cols_b,
    const float **all_a_f32,
    const float **all_b_f32,
    float **all_c_f32,
    int32_t **all_result,
    double *latencies_ns
) {
    mach_timebase_info_data_t tb;
    mach_timebase_info(&tb);
    double ns_per_tick = (double)tb.numer / (double)tb.denom;

    for (int t = 0; t < num_cases; t++) {
        uint64_t start = mach_absolute_time();

        multiply_dense_blas_i32(
            all_rows_a[t], all_cols_a[t], all_cols_b[t],
            all_a_f32[t], all_b_f32[t], all_c_f32[t], all_result[t]
        );

        uint64_t end = mach_absolute_time();
        latencies_ns[t] = (double)(end - start) * ns_per_tick;
    }
}

/* Batch BLAS multiply — parallel (BLAS already threads internally) */
void sparse_matmul_batch_blas_parallel(
    int num_cases,
    const int *all_rows_a,
    const int *all_cols_a,
    const int *all_cols_b,
    const float **all_a_f32,
    const float **all_b_f32,
    float **all_c_f32,
    int32_t **all_result,
    double *latencies_ns,
    int num_threads
) {
    mach_timebase_info_data_t tb;
    mach_timebase_info(&tb);
    double ns_per_tick = (double)tb.numer / (double)tb.denom;

    for (int t = 0; t < num_cases; t++) {
        uint64_t start = mach_absolute_time();

        multiply_dense_blas_i32(
            all_rows_a[t], all_cols_a[t], all_cols_b[t],
            all_a_f32[t], all_b_f32[t], all_c_f32[t], all_result[t]
        );

        uint64_t end = mach_absolute_time();
        latencies_ns[t] = (double)(end - start) * ns_per_tick;
    }
}

/* Adaptive hybrid: route based on estimated work ratio.
 * BLAS cost ∝ rows_a * cols_a * cols_b (dense compute)
 * Sparse cost ∝ nnz(A) * cols_b (only nonzero entries)
 * Use BLAS when dense compute is cheaper per unit of AMX throughput.
 *
 * Empirically: BLAS wins when nnz(A)/cols_b > threshold (large dense output)
 * or when matrix is large enough for AMX to amortize setup.
 * Simple heuristic: use sparse work estimate vs BLAS work. */
#define BLAS_MIN_SIZE 30000  /* minimum rows*cols for BLAS to be worthwhile */

static inline int should_use_blas(int rows_a, int cols_a, int cols_b, int nnz_a) {
    long long mat_size = (long long)rows_a * cols_a;
    if (mat_size < BLAS_MIN_SIZE) return 0;

    /* Sparse work = nnz_a * cols_b (per-tile overhead adds ~30%)
     * BLAS work = rows_a * cols_a * cols_b but AMX is ~10x faster per op
     * Route to BLAS if: blas_work/10 < sparse_work*1.3
     * i.e. if density > ~13% × (AMX_advantage/sparse_overhead) */
    long long sparse_work = (long long)nnz_a * cols_b;
    long long blas_work = (long long)rows_a * cols_a * cols_b;
    return blas_work < sparse_work * 8;
}

void sparse_matmul_batch_adaptive(
    int num_cases,
    const int *all_rows_a,
    const int *all_cols_a,
    const int *all_cols_b,
    /* Sparse args */
    const int **all_a_rowptr,
    const int16_t **all_a_colidx,
    const int8_t **all_a_vals,
    const int8_t **all_b_i8,
    /* BLAS args */
    const float **all_a_f32,
    const float **all_b_f32,
    float **all_c_f32,
    /* Output */
    int32_t **all_result,
    double *latencies_ns
) {
    mach_timebase_info_data_t tb;
    mach_timebase_info(&tb);
    double ns_per_tick = (double)tb.numer / (double)tb.denom;

    dispatch_queue_t queue = dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0);

    for (int t = 0; t < num_cases; t++) {
        int rows_a = all_rows_a[t];
        int cols_a = all_cols_a[t];
        int cols_b = all_cols_b[t];
        int nnz_a = all_a_rowptr[t][rows_a];
        int use_blas = should_use_blas(rows_a, cols_a, cols_b, nnz_a);

        uint64_t start = mach_absolute_time();

        if (use_blas) {
            multiply_dense_blas_i32(
                rows_a, cols_a, cols_b,
                all_a_f32[t], all_b_f32[t], all_c_f32[t], all_result[t]
            );
        } else if (rows_a >= 200) {
            /* Large NEON case: use intra-case parallelism */
            int rows_per_task = 20;
            int num_tasks = (rows_a + rows_per_task - 1) / rows_per_task;

            const int *a_rowptr = all_a_rowptr[t];
            const int16_t *a_colidx = all_a_colidx[t];
            const int8_t *a_vals = all_a_vals[t];
            const int8_t *b_dense = all_b_i8[t];
            int32_t *result = all_result[t];

            dispatch_apply(num_tasks, queue, ^(size_t tid) {
                int rs = (int)tid * rows_per_task;
                int re = rs + rows_per_task;
                if (re > rows_a) re = rows_a;
                if (rs < re) {
                    multiply_dense_axpy_range(rs, re, cols_a, cols_b,
                                            a_rowptr, a_colidx, a_vals,
                                            b_dense, result);
                }
            });
        } else {
            multiply_dense_axpy(
                rows_a, cols_a, cols_b,
                all_a_rowptr[t], all_a_colidx[t], all_a_vals[t],
                all_b_i8[t], all_result[t]
            );
        }

        uint64_t end = mach_absolute_time();
        latencies_ns[t] = (double)(end - start) * ns_per_tick;
    }
}

void sparse_matmul_batch_adaptive_parallel(
    int num_cases,
    const int *all_rows_a,
    const int *all_cols_a,
    const int *all_cols_b,
    /* Sparse args */
    const int **all_a_rowptr,
    const int16_t **all_a_colidx,
    const int8_t **all_a_vals,
    const int8_t **all_b_i8,
    /* BLAS args */
    const float **all_a_f32,
    const float **all_b_f32,
    float **all_c_f32,
    /* Output */
    int32_t **all_result,
    double *latencies_ns,
    int num_threads
) {
    mach_timebase_info_data_t tb;
    mach_timebase_info(&tb);
    double ns_per_tick = (double)tb.numer / (double)tb.denom;

    dispatch_queue_t queue = dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0);

    for (int t = 0; t < num_cases; t++) {
        int nnz_a = all_a_rowptr[t][all_rows_a[t]];
        int use_blas = should_use_blas(
            all_rows_a[t], all_cols_a[t], all_cols_b[t], nnz_a);

        if (use_blas) {
            /* BLAS already threads internally */
            uint64_t start = mach_absolute_time();
            multiply_dense_blas_i32(
                all_rows_a[t], all_cols_a[t], all_cols_b[t],
                all_a_f32[t], all_b_f32[t], all_c_f32[t], all_result[t]
            );
            uint64_t end = mach_absolute_time();
            latencies_ns[t] = (double)(end - start) * ns_per_tick;
        } else {
            int rows_a = all_rows_a[t];
            int cols_a = all_cols_a[t];
            int cols_b = all_cols_b[t];
            const int *a_rowptr = all_a_rowptr[t];
            const int16_t *a_colidx = all_a_colidx[t];
            const int8_t *a_vals = all_a_vals[t];
            const int8_t *b_dense = all_b_i8[t];
            int32_t *result = all_result[t];

            if (rows_a < 100) {
                uint64_t start = mach_absolute_time();
                multiply_dense_axpy(rows_a, cols_a, cols_b,
                                  a_rowptr, a_colidx, a_vals, b_dense, result);
                uint64_t end = mach_absolute_time();
                latencies_ns[t] = (double)(end - start) * ns_per_tick;
            } else {
                int rows_per_task = 20;
                int num_tasks = (rows_a + rows_per_task - 1) / rows_per_task;
                if (num_tasks < num_threads) num_tasks = num_threads;

                uint64_t start = mach_absolute_time();
                dispatch_apply(num_tasks, queue, ^(size_t tid) {
                    int rs = (int)tid * rows_per_task;
                    int re = rs + rows_per_task;
                    if (re > rows_a) re = rows_a;
                    if (rs < re) {
                        multiply_dense_axpy_range(rs, re, cols_a, cols_b,
                                                a_rowptr, a_colidx, a_vals,
                                                b_dense, result);
                    }
                });
                uint64_t end = mach_absolute_time();
                latencies_ns[t] = (double)(end - start) * ns_per_tick;
            }
        }
    }
}

/* Batch-parallel adaptive: run all test cases concurrently via GCD,
 * each using the best algorithm for its size/sparsity. */
void sparse_matmul_batch_adaptive_batch_parallel(
    int num_cases,
    const int *all_rows_a,
    const int *all_cols_a,
    const int *all_cols_b,
    const int **all_a_rowptr,
    const int16_t **all_a_colidx,
    const int8_t **all_a_vals,
    const int8_t **all_b_i8,
    const float **all_a_f32,
    const float **all_b_f32,
    float **all_c_f32,
    int32_t **all_result,
    double *latencies_ns,
    int num_threads
) {
    mach_timebase_info_data_t tb;
    mach_timebase_info(&tb);
    double ns_per_tick = (double)tb.numer / (double)tb.denom;

    dispatch_queue_t queue = dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0);

    dispatch_apply(num_cases, queue, ^(size_t t) {
        int nnz_a = all_a_rowptr[t][all_rows_a[t]];
        int rows_a = all_rows_a[t];
        int cols_a = all_cols_a[t];
        int cols_b = all_cols_b[t];

        uint64_t start = mach_absolute_time();

        /* For batch-parallel, always use sparse NEON (avoids BLAS thread contention) */
        multiply_dense_axpy(
            rows_a, cols_a, cols_b,
            all_a_rowptr[t], all_a_colidx[t], all_a_vals[t],
            all_b_i8[t], all_result[t]
        );

        uint64_t end = mach_absolute_time();
        latencies_ns[t] = (double)(end - start) * ns_per_tick;
    });
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

/* Batch PCMI multiply with C-side timing */
void sparse_matmul_batch_pcmi(
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

        multiply_single_pcmi(
            all_rows_a[t], all_cols_b[t],
            all_a_rowptr[t], all_a_colidx[t], all_a_vals[t],
            all_b_rowptr[t], all_b_colidx[t], all_b_vals[t],
            all_result[t]
        );

        uint64_t end = mach_absolute_time();
        latencies_ns[t] = (double)(end - start) * ns_per_tick;
    }
}

/* Batch int32 merge multiply with C-side timing */
void sparse_matmul_batch_i32(
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

        multiply_single_i32(
            all_rows_a[t], all_cols_b[t],
            all_a_rowptr[t], all_a_colidx[t], all_a_vals[t],
            all_b_rowptr[t], all_b_colidx[t], all_b_vals[t],
            all_result[t]
        );

        uint64_t end = mach_absolute_time();
        latencies_ns[t] = (double)(end - start) * ns_per_tick;
    }
}

/* Batch no-merge int32 multiply with C-side timing */
void sparse_matmul_batch_nomerge_i32(
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

        multiply_single_nomerge_i32(
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

/* Batch dense axpy multiply — serial */
void sparse_matmul_batch_dense_axpy(
    int num_cases,
    const int *all_rows_a,
    const int *all_cols_a,
    const int *all_cols_b,
    const int **all_a_rowptr,
    const int16_t **all_a_colidx,
    const int8_t **all_a_vals,
    const int8_t **all_b_i8,
    int32_t **all_result,
    double *latencies_ns
) {
    mach_timebase_info_data_t tb;
    mach_timebase_info(&tb);
    double ns_per_tick = (double)tb.numer / (double)tb.denom;

    for (int t = 0; t < num_cases; t++) {
        uint64_t start = mach_absolute_time();

        multiply_dense_axpy(
            all_rows_a[t], all_cols_a[t], all_cols_b[t],
            all_a_rowptr[t], all_a_colidx[t], all_a_vals[t],
            all_b_i8[t], all_result[t]
        );

        uint64_t end = mach_absolute_time();
        latencies_ns[t] = (double)(end - start) * ns_per_tick;
    }
}

/* Batch dense axpy multiply — parallel (GCD) */
void sparse_matmul_batch_dense_axpy_parallel(
    int num_cases,
    const int *all_rows_a,
    const int *all_cols_a,
    const int *all_cols_b,
    const int **all_a_rowptr,
    const int16_t **all_a_colidx,
    const int8_t **all_a_vals,
    const int8_t **all_b_i8,
    int32_t **all_result,
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
        const int *a_rowptr = all_a_rowptr[t];
        const int16_t *a_colidx = all_a_colidx[t];
        const int8_t *a_vals = all_a_vals[t];
        const int8_t *b_dense = all_b_i8[t];
        int32_t *result = all_result[t];

        if (rows_a < 100 || num_threads <= 1) {
            uint64_t start = mach_absolute_time();
            multiply_dense_axpy(rows_a, cols_a, cols_b,
                              a_rowptr, a_colidx, a_vals,
                              b_dense, result);
            uint64_t end = mach_absolute_time();
            latencies_ns[t] = (double)(end - start) * ns_per_tick;
            continue;
        }

        int rows_per_task = 20;
        int num_tasks = (rows_a + rows_per_task - 1) / rows_per_task;
        if (num_tasks < num_threads) num_tasks = num_threads;

        uint64_t start = mach_absolute_time();

        dispatch_apply(num_tasks, queue, ^(size_t tid) {
            int rs = (int)tid * rows_per_task;
            int re = rs + rows_per_task;
            if (re > rows_a) re = rows_a;
            if (rs < re) {
                multiply_dense_axpy_range(rs, re, cols_a, cols_b,
                                        a_rowptr, a_colidx, a_vals,
                                        b_dense, result);
            }
        });

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

        /* Fine-grained task decomposition: 20 rows per task for dynamic
         * load balancing via GCD work-stealing. */
        int rows_per_task = 20;
        int num_tasks = (rows_a + rows_per_task - 1) / rows_per_task;
        if (num_tasks < num_threads) num_tasks = num_threads;

        uint64_t start = mach_absolute_time();

        dispatch_apply(num_tasks, queue, ^(size_t tid) {
            int rs = (int)tid * rows_per_task;
            int re = rs + rows_per_task;
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
