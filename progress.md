# Experiment Progress

## Branch: sparse_matrix_mul/apr07_0116

---

### Experiment 1 — Baseline
- **Tag**: apr07_0116 — Experiment 1 — f932bf5
- **Algorithm**: Naive O(n^3) triple-loop matrix multiplication
- **Time Complexity**: O(m * n * p) average and worst case, where A is m×n and B is n×p
- **Pros**: Simple, correct, easy to understand
- **Cons**: Does not exploit sparsity at all — multiplies every element including zeros
- **Result**: 50/50 passed, avg latency 9838.68 ms
- **Observation**: Baseline established. Many test cases with large dimensions (800-1000) take 20-50 seconds. Clear opportunity to skip zero elements.

---

### Experiment 2 — Sparse DOK Multiplication

- **Tag**: apr07_0116 — Experiment 2 — 245fc5c
- **Algorithm**: Row-indexed sparse representation of B. For each non-zero A[i][k], look up non-zero entries in row k of B and accumulate products. Skips all zero elements.
- **Time Complexity**: O(nnz(A) * avg_nnz_per_row(B)) average case. O(m*n*p) worst case (dense matrices).
- **Pros**: 68.8x speedup over baseline. Simple implementation. Scales with actual non-zeros.
- **Cons**: Still iterates over all of A to check for zeros row-by-row. Dictionary lookup overhead.
- **Result**: 50/50 passed, avg latency 143.00 ms
- **Observation**: Massive improvement. Next steps: also sparsify A's representation, try Gustavson's algorithm with dense accumulator, or explore C-level acceleration via ctypes.

---
