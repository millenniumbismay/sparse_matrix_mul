import csv
import time
import ctypes
import os
import array
import struct

# Load C extension
_dir = os.path.dirname(os.path.abspath(__file__))
_lib = ctypes.CDLL(os.path.join(_dir, "sparse_matmul.dylib"))

_lib.build_compact_csr.restype = ctypes.c_int
_lib.build_compact_csr.argtypes = [
    ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
]

_lib.sparse_matmul_batch.restype = None
_lib.sparse_matmul_batch.argtypes = [
    ctypes.c_int,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
]

_lib.sparse_matmul_batch_parallel.restype = None
_lib.sparse_matmul_batch_parallel.argtypes = [
    ctypes.c_int,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
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

            # Pre-build compact CSR for A and B
            a_rowptr, a_colidx, a_vals = build_csr(m, n, a_flat)
            b_rowptr, b_colidx, b_vals = build_csr(n2, y, b_flat)
            # Pre-allocate result buffer
            result_buf = array.array('q', bytes(m * y * _ll_size))

            cases.append({
                "name": f"test_{test_id}",
                "rows_a": m, "cols_a": n, "cols_b": y,
                "a_rowptr": a_rowptr, "a_colidx": a_colidx, "a_vals": a_vals,
                "b_rowptr": b_rowptr, "b_colidx": b_colidx, "b_vals": b_vals,
                "result_buf": result_buf,
                "expected": exp_flat,
            })
    return cases


def run_batch(cases):
    """Run all test cases in a single C call with C-side timing."""
    n = len(cases)

    IntArray = ctypes.c_int * n
    PtrArray = ctypes.c_void_p * n

    all_rows_a = IntArray(*(tc["rows_a"] for tc in cases))
    all_cols_a = IntArray(*(tc["cols_a"] for tc in cases))
    all_cols_b = IntArray(*(tc["cols_b"] for tc in cases))

    all_a_rowptr = PtrArray(*(tc["a_rowptr"].buffer_info()[0] for tc in cases))
    all_a_colidx = PtrArray(*(ctypes.cast(tc["a_colidx"].buffer_info()[0], ctypes.c_void_p).value for tc in cases))
    all_a_vals = PtrArray(*(ctypes.addressof(tc["a_vals"]) for tc in cases))
    all_b_rowptr = PtrArray(*(tc["b_rowptr"].buffer_info()[0] for tc in cases))
    all_b_colidx = PtrArray(*(ctypes.cast(tc["b_colidx"].buffer_info()[0], ctypes.c_void_p).value for tc in cases))
    all_b_vals = PtrArray(*(ctypes.addressof(tc["b_vals"]) for tc in cases))
    all_result = PtrArray(*(tc["result_buf"].buffer_info()[0] for tc in cases))

    latencies_ns = (ctypes.c_double * n)()

    _lib.sparse_matmul_batch_parallel(
        n,
        ctypes.addressof(all_rows_a),
        ctypes.addressof(all_cols_a),
        ctypes.addressof(all_cols_b),
        ctypes.addressof(all_a_rowptr),
        ctypes.addressof(all_a_colidx),
        ctypes.addressof(all_a_vals),
        ctypes.addressof(all_b_rowptr),
        ctypes.addressof(all_b_colidx),
        ctypes.addressof(all_b_vals),
        ctypes.addressof(all_result),
        ctypes.addressof(latencies_ns),
        6,  # num_threads
    )

    return [latencies_ns[i] / 1e6 for i in range(n)]  # ns -> ms


def log(msg, file=None):
    print(msg)
    if file:
        file.write(msg + "\n")


def main():
    results = []
    log_file = open("output.log", "w")
    cases = load_test_cases()

    latencies_ms = run_batch(cases)

    for i, tc in enumerate(cases):
        name = tc["name"]
        rb = tc["result_buf"]
        latency_ms = latencies_ms[i]

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
