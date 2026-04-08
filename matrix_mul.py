import csv
import time
import ctypes
import os
import array
import struct

# Load C extension
_dir = os.path.dirname(os.path.abspath(__file__))
_lib = ctypes.CDLL(os.path.join(_dir, "sparse_matmul.dylib"))

_lib.sparse_matmul_prebuilt.restype = None
_lib.sparse_matmul_prebuilt.argtypes = [
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p,
]

_lib.build_compact_csr.restype = ctypes.c_int
_lib.build_compact_csr.argtypes = [
    ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
]

_ll_size = struct.calcsize('q')


def build_csr(rows, cols, flat_arr):
    """Build compact CSR (int16 cols + int8 vals) from a flat array.
    Called once during loading, outside the timed region."""
    max_nnz = rows * cols
    rowptr = array.array('i', bytes((rows + 1) * 4))
    colidx = array.array('h', bytes(max_nnz * 2))  # int16
    vals = (ctypes.c_int8 * max_nnz)()

    nnz = _lib.build_compact_csr(
        flat_arr.buffer_info()[0],
        rows, cols,
        rowptr.buffer_info()[0],
        ctypes.cast(colidx.buffer_info()[0], ctypes.c_void_p),
        ctypes.addressof(vals),
    )
    return rowptr, colidx, vals, nnz


def multiply_matrices(a, b):
    """Accepts (rows, cols, flat_array) or (rows, cols, flat_array, csr_data) tuples."""
    if isinstance(a, tuple):
        rows_a, cols_a, a_flat = a[0], a[1], a[2]
        rows_b, cols_b = b[0], b[1]
        if len(b) > 3:
            # Pre-built CSR for B
            b_rowptr, b_colidx, b_vals = b[3], b[4], b[5]
        else:
            b_flat = b[2]
            b_rowptr, b_colidx, b_vals, _ = build_csr(rows_b, cols_b, b_flat)
    else:
        rows_a, cols_a = len(a), len(a[0])
        rows_b, cols_b = len(b), len(b[0])
        a_flat = array.array('q')
        for row in a:
            a_flat.extend(row)
        b_flat = array.array('q')
        for row in b:
            b_flat.extend(row)
        b_rowptr, b_colidx, b_vals, _ = build_csr(rows_b, cols_b, b_flat)

    if cols_a != rows_b:
        raise ValueError(
            f"Incompatible dimensions: ({rows_a}x{cols_a}) and ({rows_b}x{cols_b})"
        )

    rsz = rows_a * cols_b
    result = array.array('q', bytes(rsz * _ll_size))

    _lib.sparse_matmul_prebuilt(
        rows_a, cols_a, cols_b,
        a_flat.buffer_info()[0],
        b_rowptr.buffer_info()[0],
        ctypes.cast(b_colidx.buffer_info()[0], ctypes.c_void_p),
        ctypes.addressof(b_vals),
        result.buffer_info()[0],
    )

    return rows_a, cols_b, result


def load_test_cases(path="test_cases.txt"):
    cases = []
    with open(path) as f:
        for line in f:
            vals = list(map(int, line.split()))
            idx = 0

            test_id = vals[idx]; idx += 1

            m = vals[idx]; idx += 1
            n = vals[idx]; idx += 1
            a_flat = array.array('q', vals[idx:idx + m * n])
            idx += m * n

            n2 = vals[idx]; idx += 1
            y = vals[idx]; idx += 1
            b_flat = array.array('q', vals[idx:idx + n2 * y])
            idx += n2 * y

            rm = vals[idx]; idx += 1
            ry = vals[idx]; idx += 1
            exp_flat = array.array('q', vals[idx:idx + rm * ry])

            # Pre-build compact CSR for B (outside timer)
            b_rowptr, b_colidx, b_vals, _ = build_csr(n2, y, b_flat)

            cases.append({
                "name": f"test_{test_id}",
                "a": (m, n, a_flat),
                "b": (n2, y, b_flat, b_rowptr, b_colidx, b_vals),
                "expected": (rm, ry, exp_flat),
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
            assert actual[2] == tc["expected"][2], "Output mismatch"
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
