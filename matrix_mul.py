import csv
import time
import ctypes
import os
import array
import struct

# Load C extension
_dir = os.path.dirname(os.path.abspath(__file__))
_lib = ctypes.CDLL(os.path.join(_dir, "sparse_matmul.dylib"))
_lib.sparse_matmul.restype = None
_lib.sparse_matmul.argtypes = [
    ctypes.c_int, ctypes.c_int,  # rows_a, cols_b
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,  # A CSR (raw pointers)
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,  # B CSR (raw pointers)
    ctypes.c_void_p,  # result
]

_int_size = struct.calcsize('i')
_ll_size = struct.calcsize('q')


def _to_csr(matrix):
    """Convert dense matrix to CSR arrays using array module."""
    rowptr = array.array('i', [0])
    colidx = array.array('i')
    vals = array.array('q')
    ca = colidx.append
    va = vals.append
    nnz = 0
    for row in matrix:
        for j, v in enumerate(row):
            if v:
                ca(j)
                va(v)
                nnz += 1
        rowptr.append(nnz)
    return rowptr, colidx, vals


def multiply_matrices(a, b):
    rows_a, cols_a = len(a), len(a[0])
    rows_b, cols_b = len(b), len(b[0])

    if cols_a != rows_b:
        raise ValueError(
            f"Incompatible dimensions: ({rows_a}x{cols_a}) and ({rows_b}x{cols_b})"
        )

    # Convert to CSR — array.array gives us buffer protocol for zero-copy
    a_rp, a_ci, a_v = _to_csr(a)
    b_rp, b_ci, b_v = _to_csr(b)

    # Allocate result
    result = array.array('q', bytes(rows_a * cols_b * _ll_size))

    # Get raw buffer addresses (zero-copy pass to C)
    a_rp_addr = a_rp.buffer_info()[0]
    a_ci_addr = a_ci.buffer_info()[0] if a_ci else 0
    a_v_addr = a_v.buffer_info()[0] if a_v else 0
    b_rp_addr = b_rp.buffer_info()[0]
    b_ci_addr = b_ci.buffer_info()[0] if b_ci else 0
    b_v_addr = b_v.buffer_info()[0] if b_v else 0
    res_addr = result.buffer_info()[0]

    _lib.sparse_matmul(
        rows_a, cols_b,
        a_rp_addr, a_ci_addr, a_v_addr,
        b_rp_addr, b_ci_addr, b_v_addr,
        res_addr,
    )

    # Convert result to list-of-lists using struct for speed
    flat = list(result)
    return [flat[i * cols_b:(i + 1) * cols_b] for i in range(rows_a)]


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
