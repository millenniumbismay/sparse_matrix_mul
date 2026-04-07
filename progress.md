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
