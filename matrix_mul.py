import csv
import time


def multiply_matrices(a, b):
    rows_a, cols_a = len(a), len(a[0])
    rows_b, cols_b = len(b), len(b[0])

    if cols_a != rows_b:
        raise ValueError(
            f"Incompatible dimensions: ({rows_a}x{cols_a}) and ({rows_b}x{cols_b})"
        )

    # Build sparse representation of A: for each row, list of (col, value) where value != 0
    a_sparse = []
    for i in range(rows_a):
        row_nz = []
        for k in range(cols_a):
            if a[i][k] != 0:
                row_nz.append((k, a[i][k]))
        a_sparse.append(row_nz)

    # Build sparse representation of B: for each row, list of (col, value) where value != 0
    b_sparse = []
    for k in range(rows_b):
        row_nz = []
        for j in range(cols_b):
            if b[k][j] != 0:
                row_nz.append((j, b[k][j]))
        b_sparse.append(row_nz)

    # Multiply using sparse row-by-row: for each non-zero a[i][k], scatter a[i][k] * b[k][j]
    result = [[0] * cols_b for _ in range(rows_a)]
    for i in range(rows_a):
        for k, a_val in a_sparse[i]:
            for j, b_val in b_sparse[k]:
                result[i][j] += a_val * b_val

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
