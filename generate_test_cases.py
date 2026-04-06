import argparse
import random


def multiply_matrices(a, b):
    rows_a, cols_a = len(a), len(a[0])
    cols_b = len(b[0])
    result = [[0] * cols_b for _ in range(rows_a)]
    for i in range(rows_a):
        for j in range(cols_b):
            for k in range(cols_a):
                result[i][j] += a[i][k] * b[k][j]
    return result


def generate_sparse_matrix(rows, cols, sparsity_pct, val_range=(-10, 10)):
    matrix = []
    for _ in range(rows):
        row = []
        for _ in range(cols):
            if random.randint(1, 100) <= sparsity_pct:
                row.append(0)
            else:
                row.append(random.randint(*val_range))
        matrix.append(row)
    return matrix


def matrix_to_flat(matrix):
    return [str(v) for row in matrix for v in row]


def generate_test_cases(num_cases=10, dim_range=(10, 1000), sparsity_range=(70, 99), val_range=(-10, 10)):
    with open("test_cases.txt", "w") as f:
        for test_id in range(1, num_cases + 1):
            m = random.randint(*dim_range)
            n = random.randint(*dim_range)
            y = random.randint(*dim_range)

            sparsity_a = random.randint(*sparsity_range)
            sparsity_b = random.randint(*sparsity_range)

            a = generate_sparse_matrix(m, n, sparsity_a, val_range)
            b = generate_sparse_matrix(n, y, sparsity_b, val_range)
            expected = multiply_matrices(a, b)

            parts = [
                str(test_id),
                str(m), str(n), *matrix_to_flat(a),
                str(n), str(y), *matrix_to_flat(b),
                str(m), str(y), *matrix_to_flat(expected),
            ]
            f.write(" ".join(parts) + "\n")

            print(f"  test_{test_id}: A={m}x{n} ({sparsity_a}% sparse), B={n}x{y} ({sparsity_b}% sparse)")

    print(f"\nGenerated {num_cases} test cases in test_cases.txt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate sparse matrix multiplication test cases")
    parser.add_argument("-n", "--num-cases", type=int, default=10, help="number of test cases (default: 10)")
    parser.add_argument("--min-sparsity", type=int, default=70, help="min sparsity %% (default: 70)")
    parser.add_argument("--max-sparsity", type=int, default=99, help="max sparsity %% (default: 99)")
    parser.add_argument("--min-dim", type=int, default=10, help="min rows/cols (default: 10)")
    parser.add_argument("--max-dim", type=int, default=1000, help="max rows/cols (default: 1000)")
    args = parser.parse_args()

    generate_test_cases(
        num_cases=args.num_cases,
        dim_range=(args.min_dim, args.max_dim),
        sparsity_range=(args.min_sparsity, args.max_sparsity),
    )
