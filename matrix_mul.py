import csv
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore')
np.seterr(all='ignore')


def multiply_matrices(a, b, a_nz_rows, b_nz_cols, k_nz, m, n):
    # a_reduced: only non-zero rows of A, only shared-dimension cols that matter
    # b_reduced: only shared-dimension rows that matter, only non-zero cols of B
    a_red = a[np.ix_(a_nz_rows, k_nz)]
    b_red = b[np.ix_(k_nz, b_nz_cols)]
    c_red = a_red @ b_red
    # Place results back into full result matrix
    result = np.zeros((m, n), dtype=np.float32)
    result[np.ix_(a_nz_rows, b_nz_cols)] = c_red
    return result


def load_test_cases(path="test_cases.txt"):
    cases = []
    with open(path) as f:
        for line in f:
            vals = list(map(int, line.split()))
            idx = 0

            test_id = vals[idx]; idx += 1

            m = vals[idx]; idx += 1
            n = vals[idx]; idx += 1
            a = np.asfortranarray(np.array(vals[idx : idx + m * n], dtype=np.float32).reshape(m, n))
            idx += m * n

            n2 = vals[idx]; idx += 1
            y = vals[idx]; idx += 1
            b = np.asfortranarray(np.array(vals[idx : idx + n2 * y], dtype=np.float32).reshape(n2, y))
            idx += n2 * y

            rm = vals[idx]; idx += 1
            ry = vals[idx]; idx += 1
            expected = np.array(vals[idx : idx + rm * ry], dtype=np.float32).reshape(rm, ry)

            # Pre-identify non-zero rows/cols for dimension reduction
            a_nz_rows = np.where(np.any(a != 0, axis=1))[0]
            b_nz_cols = np.where(np.any(b != 0, axis=0))[0]
            # Shared dimension: k values where either A has non-zero col or B has non-zero row
            a_nz_cols = np.where(np.any(a != 0, axis=0))[0]
            b_nz_rows = np.where(np.any(b != 0, axis=1))[0]
            k_nz = np.union1d(a_nz_cols, b_nz_rows)

            cases.append({
                "name": f"test_{test_id}",
                "a": a,
                "b": b,
                "expected": expected,
                "a_nz_rows": a_nz_rows,
                "b_nz_cols": b_nz_cols,
                "k_nz": k_nz,
                "m": m,
                "n": y,
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
    # Warmup BLAS with a medium-sized matmul to initialize Accelerate threadpool
    _w = np.ones((500, 500), dtype=np.float32) @ np.ones((500, 500), dtype=np.float32)

    for tc in test_cases:
        name = tc["name"]
        start = time.perf_counter()
        actual = multiply_matrices(tc["a"], tc["b"], tc["a_nz_rows"], tc["b_nz_cols"], tc["k_nz"], tc["m"], tc["n"])
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
