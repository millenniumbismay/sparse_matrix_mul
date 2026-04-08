cases = []
with open('test_cases.txt') as f:
    for line in f:
        vals = list(map(int, line.split()))
        idx = 0
        test_id = vals[idx]; idx += 1
        m = vals[idx]; idx += 1
        n = vals[idx]; idx += 1
        a_data = vals[idx:idx + m * n]; idx += m * n
        n2 = vals[idx]; idx += 1
        y = vals[idx]; idx += 1
        b_data = vals[idx:idx + n2 * y]; idx += n2 * y
        rm = vals[idx]; idx += 1
        ry = vals[idx]; idx += 1
        exp_data = vals[idx:idx + rm * ry]

        a_nnz = sum(1 for x in a_data if x != 0)
        b_nnz = sum(1 for x in b_data if x != 0)
        c_nnz = sum(1 for x in exp_data if x != 0)
        a_total = m * n
        b_total = n2 * y
        c_total = rm * ry

        cases.append((test_id, m, n, y,
                      100*(1 - a_nnz/a_total),
                      100*(1 - b_nnz/b_total),
                      100*(1 - c_nnz/c_total)))

print("  ID    m    n    p  A_sp%  B_sp%  C_sp%")
for t in cases:
    print("%4d %4d %4d %4d %6.1f %6.1f %6.1f" % t)

c_sparsities = [t[6] for t in cases]
print("\nOutput sparsity: min=%.1f%%, max=%.1f%%, avg=%.1f%%" % (
    min(c_sparsities), max(c_sparsities), sum(c_sparsities)/len(c_sparsities)))

# Also compute: what fraction of work is in the top-5 test cases?
dims = [(t[0], t[1]*t[2]*t[3], t[1], t[2], t[3]) for t in cases]
dims.sort(key=lambda x: -x[1])
total_flops = sum(d[1] for d in dims)
print("\nTop 10 test cases by m*n*p:")
cumul = 0
for tid, mnp, m, n, p in dims[:10]:
    cumul += mnp
    print("  test_%d: %dx%dx%d = %d (cumul %.1f%%)" % (tid, m, n, p, mnp, 100*cumul/total_flops))
