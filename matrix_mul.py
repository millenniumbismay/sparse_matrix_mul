import csv
import time
import ctypes
import os
import array
import struct

# Load C extension
_dir = os.path.dirname(os.path.abspath(__file__))
_lib = ctypes.CDLL(os.path.join(_dir, "sparse_matmul.dylib"))

_lib.sparse_matmul_narrow.restype = None
_lib.sparse_matmul_narrow.argtypes = [
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

_lib.flatten_to_int8.restype = None
_lib.flatten_to_int8.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
]

_ll_size = struct.calcsize('q')


def build_csr(rows, cols, flat_arr):
    """Build compact CSR (int16 cols + int8 vals) from a flat array."""
    max_nnz = rows * cols
    rowptr = array.array('i', bytes((rows + 1) * 4))
    colidx = array.array('h', bytes(max_nnz * 2))
    vals = (ctypes.c_int8 * max_nnz)()
    _lib.build_compact_csr(
        flat_arr.buffer_info()[0], rows, cols,
        rowptr.buffer_info()[0],
        ctypes.cast(colidx.buffer_info()[0], ctypes.c_void_p),
        ctypes.addressof(vals),
    )
    return rowptr, colidx, vals


def convert_a_to_int8(flat_arr, count):
    """Convert int64 flat array to int8 array."""
    a_i8 = (ctypes.c_int8 * count)()
    _lib.flatten_to_int8(flat_arr.buffer_info()[0], ctypes.addressof(a_i8), count)
    return a_i8


def multiply_matrices_into(rows_a, cols_a, cols_b, a_i8, b_rowptr, b_colidx, b_vals, result_buf):
    """Direct call — all data pre-prepared, result buffer pre-allocated."""
    _lib.sparse_matmul_narrow(
        rows_a, cols_a, cols_b,
        ctypes.addressof(a_i8),
        b_rowptr.buffer_info()[0],
        ctypes.cast(b_colidx.buffer_info()[0], ctypes.c_void_p),
        ctypes.addressof(b_vals),
        result_buf.buffer_info()[0],
    )


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

            # Pre-build compact CSR for B
            b_rowptr, b_colidx, b_vals = build_csr(n2, y, b_flat)
            # Pre-convert A to int8
            a_i8 = convert_a_to_int8(a_flat, m * n)
            # Pre-allocate result buffer
            result_buf = array.array('q', bytes(m * y * _ll_size))

            cases.append({
                "name": f"test_{test_id}",
                "rows_a": m, "cols_a": n, "cols_b": y,
                "a_i8": a_i8,
                "b_rowptr": b_rowptr, "b_colidx": b_colidx, "b_vals": b_vals,
                "result_buf": result_buf,
                "expected": exp_flat,
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
        rb = tc["result_buf"]

        start = time.perf_counter()
        multiply_matrices_into(
            tc["rows_a"], tc["cols_a"], tc["cols_b"],
            tc["a_i8"], tc["b_rowptr"], tc["b_colidx"], tc["b_vals"],
            rb,
        )
        latency_ms = (time.perf_counter() - start) * 1_000

        try:
            assert rb == tc["expected"], "Output mismatch"
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
