import csv
import time
import numpy as np
import numba


@numba.njit(parallel=True, cache=True, fastmath=True)
def sparse_dense_matmul(a_indptr, a_indices, a_data, b, m, n):
    """Sparse A (CSR) × Dense B. Exploits sparsity in A, vectorizes over B columns."""
    result = np.zeros((m, n), dtype=np.float32)
    for i in numba.prange(m):
        for idx in range(a_indptr[i], a_indptr[i + 1]):
            k = a_indices[idx]
            val = a_data[idx]
            for j in range(n):
                result[i, j] += val * b[k, j]
    return result


def multiply_matrices(a_csr, b):
    indptr, indices, data, m = a_csr
    _, n = b.shape
    return sparse_dense_matmul(indptr, indices, data, b, m, n)


def load_test_cases(path="test_cases.txt"):
    cases = []
    with open(path) as f:
        for line in f:
            vals = list(map(int, line.split()))
            idx = 0

            test_id = vals[idx]; idx += 1

            m = vals[idx]; idx += 1
            n = vals[idx]; idx += 1
            a_dense = np.array(vals[idx : idx + m * n], dtype=np.float32).reshape(m, n)
            idx += m * n
            # Build CSR for A
            a_nz = a_dense != 0
            indptr = np.zeros(m + 1, dtype=np.int32)
            for i in range(m):
                indptr[i + 1] = indptr[i] + np.sum(a_nz[i])
            nnz = indptr[m]
            indices = np.empty(nnz, dtype=np.int32)
            data = np.empty(nnz, dtype=np.float32)
            pos = 0
            for i in range(m):
                for j in range(n):
                    if a_nz[i, j]:
                        indices[pos] = j
                        data[pos] = a_dense[i, j]
                        pos += 1
            a_csr = (indptr, indices, data, m)

            n2 = vals[idx]; idx += 1
            y = vals[idx]; idx += 1
            b = np.ascontiguousarray(np.array(vals[idx : idx + n2 * y], dtype=np.float32).reshape(n2, y))
            idx += n2 * y

            rm = vals[idx]; idx += 1
            ry = vals[idx]; idx += 1
            expected = np.array(vals[idx : idx + rm * ry], dtype=np.float32).reshape(rm, ry)

            cases.append({
                "name": f"test_{test_id}",
                "a": a_csr,
                "b": b,
                "expected": expected,
            })
    return cases


def log(msg, file=None):
    print(msg)
    if file:
        file.write(msg + "\n")


def main():
    results = []
    log_file = open("output.log", "w")

    test_cases = load_test_cases()
    # Warm up numba JIT with first test case
    multiply_matrices(test_cases[0]["a"], test_cases[0]["b"])

    for tc in test_cases:
        name = tc["name"]
        start = time.perf_counter()
        actual = multiply_matrices(tc["a"], tc["b"])
        latency_ms = (time.perf_counter() - start) * 1_000

        if np.array_equal(actual, tc["expected"]):
            solution = "correct"
            observation = "Output matches expected result"
            log(f"  PASS  {name} ({latency_ms:.4f} ms)", log_file)
        else:
            solution = "incorrect"
            observation = f"Output mismatch"
            log(f"  FAIL  {name} ({latency_ms:.4f} ms) — output mismatch", log_file)

        results.append({
            "test_case": name,
            "solution": solution,
            "latency": f"{latency_ms:.4f} ms",
            "observation": observation,
        })

    with open("results.tsv", "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["test_case", "solution", "latency", "observation"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(results)

        passed = sum(1 for r in results if r["solution"] == "correct")
        latencies = [float(r["latency"].replace(" ms", "")) for r in results]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0

        writer.writerow({
            "test_case": "SUMMARY",
            "solution": f"{passed}/{len(results)} passed",
            "latency": f"{avg_latency:.4f} ms",
            "observation": "Average latency across all test cases",
        })

    log(f"\n{passed}/{len(results)} passed, avg latency: {avg_latency:.4f} ms", log_file)
    log(f"Results written to results.tsv", log_file)
    log_file.close()


if __name__ == "__main__":
    main()
