import csv
import time
import numpy as np

STRASSEN_THRESHOLD = 256


def strassen(a, b):
    m, k = a.shape
    _, n = b.shape

    # Use standard BLAS for small matrices
    if m <= STRASSEN_THRESHOLD or k <= STRASSEN_THRESHOLD or n <= STRASSEN_THRESHOLD:
        return a @ b

    # Pad to even dimensions
    mp = m + (m & 1)
    kp = k + (k & 1)
    np_ = n + (n & 1)

    if mp != m or kp != k:
        a_pad = np.zeros((mp, kp), dtype=np.float32)
        a_pad[:m, :k] = a
    else:
        a_pad = a

    if kp != k or np_ != n:
        b_pad = np.zeros((kp, np_), dtype=np.float32)
        b_pad[:k, :n] = b
    else:
        b_pad = b

    h_m = mp // 2
    h_k = kp // 2
    h_n = np_ // 2

    # Split into quadrants
    a11 = a_pad[:h_m, :h_k]
    a12 = a_pad[:h_m, h_k:]
    a21 = a_pad[h_m:, :h_k]
    a22 = a_pad[h_m:, h_k:]

    b11 = b_pad[:h_k, :h_n]
    b12 = b_pad[:h_k, h_n:]
    b21 = b_pad[h_k:, :h_n]
    b22 = b_pad[h_k:, h_n:]

    # Strassen's 7 products (use @ for BLAS-backed sub-multiplications)
    s1 = (a11 + a22) @ (b11 + b22)
    s2 = (a21 + a22) @ b11
    s3 = a11 @ (b12 - b22)
    s4 = a22 @ (b21 - b11)
    s5 = (a11 + a12) @ b22
    s6 = (a21 - a11) @ (b11 + b12)
    s7 = (a12 - a22) @ (b21 + b22)

    # Combine
    c = np.empty((mp, np_), dtype=np.float32)
    c[:h_m, :h_n] = s1 + s4 - s5 + s7
    c[:h_m, h_n:] = s3 + s5
    c[h_m:, :h_n] = s2 + s4
    c[h_m:, h_n:] = s1 - s2 + s3 + s6

    return c[:m, :n]


def multiply_matrices(a, b):
    return strassen(a, b)


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
