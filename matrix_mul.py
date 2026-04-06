import csv
import time
import ctypes
import numpy as np

# Load Accelerate framework and get cblas_sgemm
_acc = ctypes.CDLL('/System/Library/Frameworks/Accelerate.framework/Accelerate')
_sgemm = _acc.cblas_sgemm
_sgemm.restype = None
_sgemm.argtypes = [
    ctypes.c_int,   # Order (CblasRowMajor=101, CblasColMajor=102)
    ctypes.c_int,   # TransA
    ctypes.c_int,   # TransB
    ctypes.c_int,   # M
    ctypes.c_int,   # N
    ctypes.c_int,   # K
    ctypes.c_float,  # alpha
    ctypes.c_void_p, # A
    ctypes.c_int,   # lda
    ctypes.c_void_p, # B
    ctypes.c_int,   # ldb
    ctypes.c_float,  # beta
    ctypes.c_void_p, # C
    ctypes.c_int,   # ldc
]

CblasRowMajor = 101
CblasNoTrans = 111


def multiply_matrices(a, b, c, a_ptr, b_ptr, c_ptr, m, k, n):
    # All pointers and dimensions pre-computed in loader
    _sgemm(
        101, 111, 111,  # RowMajor, NoTrans, NoTrans (inline constants)
        m, n, k,
        1.0,
        a_ptr, k,
        b_ptr, n,
        0.0,
        c_ptr, n,
    )
    return c


def load_test_cases(path="test_cases.txt"):
    cases = []
    with open(path) as f:
        for line in f:
            vals = list(map(int, line.split()))
            idx = 0

            test_id = vals[idx]; idx += 1

            m = vals[idx]; idx += 1
            n = vals[idx]; idx += 1
            a = np.ascontiguousarray(np.array(vals[idx : idx + m * n], dtype=np.float32).reshape(m, n))
            idx += m * n

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
                "a": a,
                "b": b,
                "expected": expected,
                "out": out,
                "a_ptr": a.ctypes.data,
                "b_ptr": b.ctypes.data,
                "c_ptr": out.ctypes.data,
                "m": m,
                "k": n,
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

    for tc in load_test_cases():
        name = tc["name"]
        start = time.perf_counter()
        actual = multiply_matrices(tc["a"], tc["b"], tc["out"], tc["a_ptr"], tc["b_ptr"], tc["c_ptr"], tc["m"], tc["k"], tc["n"])
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
