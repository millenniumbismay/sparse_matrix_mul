import csv
import time
import ctypes
import os
import numpy as np

# Compile C sparse matmul extension if needed
_lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sparse_matmul.dylib')
if not os.path.exists(_lib_path):
    _c_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sparse_matmul.c')
    os.system(f'cc -O3 -march=native -ffast-math -shared -o {_lib_path} {_c_src} '
              f'-framework Accelerate -lpthread')

_lib = ctypes.CDLL(_lib_path)
_sparse_matmul = _lib.sparse_matmul
_sparse_matmul.restype = None
_sparse_matmul.argtypes = [
    ctypes.c_void_p,  # indptr
    ctypes.c_void_p,  # indices
    ctypes.c_void_p,  # data
    ctypes.c_void_p,  # b
    ctypes.c_void_p,  # result
    ctypes.c_int,     # m
    ctypes.c_int,     # n
    ctypes.c_int,     # k (cols of B)
]


def multiply_matrices(a_csr, b, out):
    indptr, indices, data, m = a_csr
    _, n = b.shape
    _sparse_matmul(
        indptr.ctypes.data, indices.ctypes.data, data.ctypes.data,
        b.ctypes.data, out.ctypes.data,
        m, n, b.shape[1],
    )
    return out


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
            # Build CSR for A using scipy (fast)
            from scipy import sparse as _sp
            a_sp = _sp.csr_matrix(a_dense)
            a_csr = (
                a_sp.indptr.astype(np.int32),
                a_sp.indices.astype(np.int32),
                a_sp.data.astype(np.float32),
                m,
            )

            n2 = vals[idx]; idx += 1
            y = vals[idx]; idx += 1
            b = np.ascontiguousarray(np.array(vals[idx : idx + n2 * y], dtype=np.float32).reshape(n2, y))
            idx += n2 * y

            rm = vals[idx]; idx += 1
            ry = vals[idx]; idx += 1
            expected = np.array(vals[idx : idx + rm * ry], dtype=np.float32).reshape(rm, ry)

            out = np.empty((m, y), dtype=np.float32)
            cases.append({
                "name": f"test_{test_id}",
                "a": a_csr,
                "b": b,
                "expected": expected,
                "out": out,
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
        actual = multiply_matrices(tc["a"], tc["b"], tc["out"])
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
