import csv
import time
import ctypes
import os
import array

# Load C extension
_dir = os.path.dirname(os.path.abspath(__file__))
_lib = ctypes.CDLL(os.path.join(_dir, "sparse_matmul.dylib"))
_lib.sparse_matmul.restype = None
_lib.sparse_matmul.argtypes = [
    ctypes.c_int, ctypes.c_int,  # rows_a, cols_b
    ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_longlong),  # A CSR
    ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_longlong),  # B CSR
    ctypes.POINTER(ctypes.c_longlong),  # result
]


def _to_csr(matrix, rows, cols):
    """Convert dense matrix to CSR arrays."""
    rowptr = array.array('i')
    colidx = array.array('i')
    vals = array.array('q')  # signed long long
    rowptr.append(0)
    for i in range(rows):
        row = matrix[i]
        for j in range(cols):
            if row[j] != 0:
                colidx.append(j)
                vals.append(row[j])
        rowptr.append(len(colidx))
    return rowptr, colidx, vals


def multiply_matrices(a, b):
    rows_a, cols_a = len(a), len(a[0])
    rows_b, cols_b = len(b), len(b[0])

    if cols_a != rows_b:
        raise ValueError(
            f"Incompatible dimensions: ({rows_a}x{cols_a}) and ({rows_b}x{cols_b})"
        )

    # Convert to CSR
    a_rowptr, a_colidx, a_vals = _to_csr(a, rows_a, cols_a)
    b_rowptr, b_colidx, b_vals = _to_csr(b, rows_b, cols_b)

    # Allocate result
    result = (ctypes.c_longlong * (rows_a * cols_b))()

    # Call C function
    _lib.sparse_matmul(
        rows_a, cols_b,
        (ctypes.c_int * len(a_rowptr))(*a_rowptr),
        (ctypes.c_int * len(a_colidx))(*a_colidx) if a_colidx else (ctypes.c_int * 1)(),
        (ctypes.c_longlong * len(a_vals))(*a_vals) if a_vals else (ctypes.c_longlong * 1)(),
        (ctypes.c_int * len(b_rowptr))(*b_rowptr),
        (ctypes.c_int * len(b_colidx))(*b_colidx) if b_colidx else (ctypes.c_int * 1)(),
        (ctypes.c_longlong * len(b_vals))(*b_vals) if b_vals else (ctypes.c_longlong * 1)(),
        result,
    )

    # Convert back to list of lists
    return [list(result[i * cols_b:(i + 1) * cols_b]) for i in range(rows_a)]


def load_test_cases(path="test_cases.txt"):
    cases = []
    with open(path) as f:
        for line in f:
            vals = list(map(int, line.split()))
            idx = 0

            test_id = vals[idx]; idx += 1

            m = vals[idx]; idx += 1
            n = vals[idx]; idx += 1
            a = [vals[idx + i * n : idx + (i + 1) * n] for i in range(m)]
            idx += m * n

            n2 = vals[idx]; idx += 1
            y = vals[idx]; idx += 1
            b = [vals[idx + i * y : idx + (i + 1) * y] for i in range(n2)]
            idx += n2 * y

            rm = vals[idx]; idx += 1
            ry = vals[idx]; idx += 1
            expected = [vals[idx + i * ry : idx + (i + 1) * ry] for i in range(rm)]

            cases.append({
                "name": f"test_{test_id}",
                "a": a,
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

    for tc in load_test_cases():
        name = tc["name"]
        start = time.perf_counter()
        actual = multiply_matrices(tc["a"], tc["b"])
        latency_ms = (time.perf_counter() - start) * 1_000

        try:
            assert actual == tc["expected"], (
                f"Expected {tc['expected']}, got {actual}"
            )
            solution = "correct"
            observation = "Output matches expected result"
            log(f"  PASS  {name} ({latency_ms:.4f} ms)", log_file)
        except AssertionError as e:
            solution = "incorrect"
            observation = str(e)
            log(f"  FAIL  {name} ({latency_ms:.4f} ms) — {e}", log_file)

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
