This repo aims to optimize the time complexity for Sparse Matrix multiplication using AutoResearch.

## Current Results

![Optimization Progress](results_plot.png)

| Experiment | Latency | Status | Observation |
|---|---|---|---|
| 1 (baseline) | 9838.68 ms | keep | Naive O(n^3) triple-loop |
| 2 | 143.00 ms | keep | Sparse DOK — skip zeros, 68.8x speedup |
| 3 | 120.29 ms | keep | Fully sparse A+B, local var caching |
| 4 | 30.34 ms | keep | C extension via ctypes, 324x vs baseline |
| 5 | 35.55 ms | discard | Dense-input C — worse than CSR marshaling |
| 6 | 20.02 ms | keep | Zero-copy buffer, 490x vs baseline |
| 7 | 18.51 ms | keep | Full-C pipeline |
| 8 | 19.36 ms | discard | Hybrid row-scan + itertools |
| 9 | 1.79 ms | keep | Pre-flatten: true C perf, 5507x |
| 10 | 2.43 ms | discard | Outer product — worse cache locality |
| 11 | 1.61 ms | keep | Direct A-scan, 6111x vs baseline |
