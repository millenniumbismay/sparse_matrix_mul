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

### Experiment 3 — Fully Sparse A+B with Local Var Caching

- **Tag**: apr07_0116 — Experiment 3 — aae3799
- **Algorithm**: Pre-extract non-zeros from both A and B as list-of-tuples per row. Cache result_row as local variable. Only iterate non-zero pairs.
- **Time Complexity**: O(nnz(A) * avg_nnz_per_row(B)) average. Same worst case O(m*n*p).
- **Pros**: 16% improvement over exp 2. Eliminates scanning zero positions in A. Local var caching reduces dict lookups.
- **Cons**: List comprehension for sparse extraction still touches all elements once during setup.
- **Result**: 50/50 passed, avg latency 120.29 ms
- **Observation**: Incremental gain from avoiding zero-checks in A. Next: try C extension via ctypes or algorithmic changes like blocking.

---

### Experiment 4 — C Extension via ctypes (CSR Sparse Matmul)

- **Tag**: apr07_0116 — Experiment 4 — b84133c
- **Algorithm**: Same sparse CSR algorithm but implemented in C. Python converts dense matrices to CSR format, passes to C function via ctypes.
- **Time Complexity**: O(nnz(A) * avg_nnz_per_row(B)) with ~100x lower constant factor.
- **Pros**: 324x speedup over baseline. 4x over pure Python sparse.
- **Cons**: Python-to-C marshaling overhead. Platform-specific binary.
- **Result**: 50/50 passed, avg latency 30.34 ms
- **Observation**: Bottleneck now in Python-side CSR construction and result conversion.

---

### Experiment 5 — Dense-Input C Extension (DISCARDED)

- **Tag**: apr07_0116 — Experiment 5 — 62f8179
- **Algorithm**: Pass flat dense arrays to C, build CSR in C. Intended to eliminate Python CSR overhead.
- **Result**: 50/50 passed, avg latency 35.55 ms (WORSE than exp 4)
- **Observation**: Moving ALL elements through Python-C boundary is slower than only marshaling non-zeros. Reverted to exp 4.

---
