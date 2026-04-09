import csv
import time
import ctypes
import os
import array
import struct

# Load C extension
_dir = os.path.dirname(os.path.abspath(__file__))
_lib = ctypes.CDLL(os.path.join(_dir, "sparse_matmul.dylib"))

_lib.flatten_to_int8.restype = None
_lib.flatten_to_int8.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
]

_lib.build_compact_csr.restype = ctypes.c_int
_lib.build_compact_csr.argtypes = [
    ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
]

_lib.build_compact_csc.restype = ctypes.c_int
_lib.build_compact_csc.argtypes = [
    ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
]

_lib.sparse_matmul_batch_outer.restype = None
_lib.sparse_matmul_batch_outer.argtypes = [
    ctypes.c_int,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
]

_lib.sparse_matmul_batch_radix.restype = None
_lib.sparse_matmul_batch_radix.argtypes = [
    ctypes.c_int,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
]

_lib.sparse_matmul_batch_diag.restype = None
_lib.sparse_matmul_batch_diag.argtypes = [
    ctypes.c_int,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
]

_lib.sparse_matmul_batch_nomerge.restype = None
_lib.sparse_matmul_batch_nomerge.argtypes = [
    ctypes.c_int,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
]

_lib.sparse_matmul_batch_rowblock.restype = None
_lib.sparse_matmul_batch_rowblock.argtypes = [
    ctypes.c_int,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
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

_lib.convert_i8_to_f32.restype = None
_lib.convert_i8_to_f32.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
]

_lib.sparse_matmul_batch_blas.restype = None
_lib.sparse_matmul_batch_blas.argtypes = [
    ctypes.c_int,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
]

_lib.sparse_matmul_batch_blas_parallel.restype = None
_lib.sparse_matmul_batch_blas_parallel.argtypes = [
    ctypes.c_int,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
]

_lib.build_row_order.restype = None
_lib.build_row_order.argtypes = [
    ctypes.c_int,
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p,
]

_lib.sparse_matmul_batch_reordered.restype = None
_lib.sparse_matmul_batch_reordered.argtypes = [
    ctypes.c_int,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
]

_lib.sparse_matmul_batch_pcmi.restype = None
_lib.sparse_matmul_batch_pcmi.argtypes = [
    ctypes.c_int,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
]

_lib.sparse_matmul_batch_i32.restype = None
_lib.sparse_matmul_batch_i32.argtypes = [
    ctypes.c_int,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
]

_lib.sparse_matmul_batch_nomerge_i32.restype = None
_lib.sparse_matmul_batch_nomerge_i32.argtypes = [
    ctypes.c_int,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
]

_lib.sparse_matmul_batch_dense_axpy.restype = None
_lib.sparse_matmul_batch_dense_axpy.argtypes = [
    ctypes.c_int,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p,  # b_i8
    ctypes.c_void_p,
    ctypes.c_void_p,
]

_lib.sparse_matmul_batch_dense_axpy_parallel.restype = None
_lib.sparse_matmul_batch_dense_axpy_parallel.argtypes = [
    ctypes.c_int,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p,  # b_i8
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
]

_lib.sparse_matmul_batch_hybrid.restype = None
_lib.sparse_matmul_batch_hybrid.argtypes = [
    ctypes.c_int,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_void_p,  # dense int8 A, B
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


def build_csc(rows, cols, flat_arr):
    """Build compact CSC (int16 rows + int8 vals) from a flat array."""
    max_nnz = rows * cols
    colptr = array.array('i', bytes((cols + 1) * 4))
    rowidx = array.array('h', bytes(max_nnz * 2))
    vals = (ctypes.c_int8 * max_nnz)()
    _lib.build_compact_csc(
        flat_arr.buffer_info()[0], rows, cols,
        colptr.buffer_info()[0],
        ctypes.cast(rowidx.buffer_info()[0], ctypes.c_void_p),
        ctypes.addressof(vals),
    )
    return colptr, rowidx, vals


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
            # Pre-build CSC for A (for outer product formulation)
            a_colptr, a_rowidx, a_vals_csc = build_csc(m, n, a_flat)
            # Pre-compute dense int8 arrays for BLAS path
            a_i8 = (ctypes.c_int8 * (m * n))()
            _lib.flatten_to_int8(a_flat.buffer_info()[0], ctypes.addressof(a_i8), m * n)
            b_i8 = (ctypes.c_int8 * (n2 * y))()
            _lib.flatten_to_int8(b_flat.buffer_info()[0], ctypes.addressof(b_i8), n2 * y)
            # Pre-allocate result buffers (int32 for dense_axpy, int64 for legacy)
            result_buf = array.array('q', bytes(m * y * _ll_size))
            result_buf_i32 = array.array('i', bytes(m * y * 4))

            # Pre-compute row ordering for B-data locality
            row_order = array.array('i', bytes(m * 4))
            _lib.build_row_order(
                m,
                a_rowptr.buffer_info()[0],
                ctypes.cast(a_colidx.buffer_info()[0], ctypes.c_void_p),
                row_order.buffer_info()[0],
            )

            # Pre-compute float32 versions for BLAS path
            a_f32 = (ctypes.c_float * (m * n))()
            _lib.convert_i8_to_f32(ctypes.addressof(a_i8), ctypes.addressof(a_f32), m * n)
            b_f32 = (ctypes.c_float * (n2 * y))()
            _lib.convert_i8_to_f32(ctypes.addressof(b_i8), ctypes.addressof(b_f32), n2 * y)
            c_f32 = (ctypes.c_float * (m * y))()  # scratch buffer

            # Pre-compute int32 expected values for comparison
            exp_i32 = array.array('i', (int(v) for v in exp_flat))

            cases.append({
                "name": f"test_{test_id}",
                "rows_a": m, "cols_a": n, "cols_b": y,
                "a_i8": a_i8, "b_i8": b_i8,
                "a_rowptr": a_rowptr, "a_colidx": a_colidx, "a_vals": a_vals,
                "a_colptr": a_colptr, "a_rowidx": a_rowidx, "a_vals_csc": a_vals_csc,
                "b_rowptr": b_rowptr, "b_colidx": b_colidx, "b_vals": b_vals,
                "row_order": row_order,
                "a_f32": a_f32, "b_f32": b_f32, "c_f32": c_f32,
                "result_buf": result_buf,
                "result_buf_i32": result_buf_i32,
                "expected": exp_flat,
                "expected_i32": exp_i32,
            })
    return cases


def run_batch(cases, method="serial"):
    """Run all test cases in a single C call with C-side timing."""
    n = len(cases)

    IntArray = ctypes.c_int * n
    PtrArray = ctypes.c_void_p * n

    all_rows_a = IntArray(*(tc["rows_a"] for tc in cases))
    all_cols_a = IntArray(*(tc["cols_a"] for tc in cases))
    all_cols_b = IntArray(*(tc["cols_b"] for tc in cases))

    all_b_rowptr = PtrArray(*(tc["b_rowptr"].buffer_info()[0] for tc in cases))
    all_b_colidx = PtrArray(*(ctypes.cast(tc["b_colidx"].buffer_info()[0], ctypes.c_void_p).value for tc in cases))
    all_b_vals = PtrArray(*(ctypes.addressof(tc["b_vals"]) for tc in cases))
    use_i32 = method in ("dense_axpy", "dense_axpy_parallel", "blas", "blas_parallel")
    all_result = PtrArray(*(tc["result_buf_i32" if use_i32 else "result_buf"].buffer_info()[0] for tc in cases))

    latencies_ns = (ctypes.c_double * n)()

    if method == "outer":
        all_a_colptr = PtrArray(*(tc["a_colptr"].buffer_info()[0] for tc in cases))
        all_a_rowidx = PtrArray(*(ctypes.cast(tc["a_rowidx"].buffer_info()[0], ctypes.c_void_p).value for tc in cases))
        all_a_vals_csc = PtrArray(*(ctypes.addressof(tc["a_vals_csc"]) for tc in cases))

        _lib.sparse_matmul_batch_outer(
            n,
            ctypes.addressof(all_rows_a),
            ctypes.addressof(all_cols_a),
            ctypes.addressof(all_cols_b),
            ctypes.addressof(all_a_colptr),
            ctypes.addressof(all_a_rowidx),
            ctypes.addressof(all_a_vals_csc),
            ctypes.addressof(all_b_rowptr),
            ctypes.addressof(all_b_colidx),
            ctypes.addressof(all_b_vals),
            ctypes.addressof(all_result),
            ctypes.addressof(latencies_ns),
        )
    elif method == "radix":
        all_a_rowptr = PtrArray(*(tc["a_rowptr"].buffer_info()[0] for tc in cases))
        all_a_colidx = PtrArray(*(ctypes.cast(tc["a_colidx"].buffer_info()[0], ctypes.c_void_p).value for tc in cases))
        all_a_vals = PtrArray(*(ctypes.addressof(tc["a_vals"]) for tc in cases))

        _lib.sparse_matmul_batch_radix(
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
        )
    elif method == "diag_memset" or method == "diag_scatter":
        mode = 0 if method == "diag_memset" else 1
        all_a_rowptr = PtrArray(*(tc["a_rowptr"].buffer_info()[0] for tc in cases))
        all_a_colidx = PtrArray(*(ctypes.cast(tc["a_colidx"].buffer_info()[0], ctypes.c_void_p).value for tc in cases))
        all_a_vals = PtrArray(*(ctypes.addressof(tc["a_vals"]) for tc in cases))

        _lib.sparse_matmul_batch_diag(
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
            mode,
        )
    elif method == "nomerge":
        all_a_rowptr = PtrArray(*(tc["a_rowptr"].buffer_info()[0] for tc in cases))
        all_a_colidx = PtrArray(*(ctypes.cast(tc["a_colidx"].buffer_info()[0], ctypes.c_void_p).value for tc in cases))
        all_a_vals = PtrArray(*(ctypes.addressof(tc["a_vals"]) for tc in cases))

        _lib.sparse_matmul_batch_nomerge(
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
        )
    elif method.startswith("block"):
        block_size = int(method.replace("block", ""))
        all_a_rowptr = PtrArray(*(tc["a_rowptr"].buffer_info()[0] for tc in cases))
        all_a_colidx = PtrArray(*(ctypes.cast(tc["a_colidx"].buffer_info()[0], ctypes.c_void_p).value for tc in cases))
        all_a_vals = PtrArray(*(ctypes.addressof(tc["a_vals"]) for tc in cases))

        _lib.sparse_matmul_batch_rowblock(
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
            block_size,
        )
    elif method == "reordered":
        all_a_rowptr = PtrArray(*(tc["a_rowptr"].buffer_info()[0] for tc in cases))
        all_a_colidx = PtrArray(*(ctypes.cast(tc["a_colidx"].buffer_info()[0], ctypes.c_void_p).value for tc in cases))
        all_a_vals = PtrArray(*(ctypes.addressof(tc["a_vals"]) for tc in cases))
        all_row_orders = PtrArray(*(tc["row_order"].buffer_info()[0] for tc in cases))

        _lib.sparse_matmul_batch_reordered(
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
            ctypes.addressof(all_row_orders),
        )
    elif method == "pcmi":
        all_a_rowptr = PtrArray(*(tc["a_rowptr"].buffer_info()[0] for tc in cases))
        all_a_colidx = PtrArray(*(ctypes.cast(tc["a_colidx"].buffer_info()[0], ctypes.c_void_p).value for tc in cases))
        all_a_vals = PtrArray(*(ctypes.addressof(tc["a_vals"]) for tc in cases))

        _lib.sparse_matmul_batch_pcmi(
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
        )
    elif method == "i32":
        all_a_rowptr = PtrArray(*(tc["a_rowptr"].buffer_info()[0] for tc in cases))
        all_a_colidx = PtrArray(*(ctypes.cast(tc["a_colidx"].buffer_info()[0], ctypes.c_void_p).value for tc in cases))
        all_a_vals = PtrArray(*(ctypes.addressof(tc["a_vals"]) for tc in cases))

        _lib.sparse_matmul_batch_i32(
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
        )
    elif method == "nomerge_i32":
        all_a_rowptr = PtrArray(*(tc["a_rowptr"].buffer_info()[0] for tc in cases))
        all_a_colidx = PtrArray(*(ctypes.cast(tc["a_colidx"].buffer_info()[0], ctypes.c_void_p).value for tc in cases))
        all_a_vals = PtrArray(*(ctypes.addressof(tc["a_vals"]) for tc in cases))

        _lib.sparse_matmul_batch_nomerge_i32(
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
        )
    elif method == "dense_axpy":
        all_a_rowptr = PtrArray(*(tc["a_rowptr"].buffer_info()[0] for tc in cases))
        all_a_colidx = PtrArray(*(ctypes.cast(tc["a_colidx"].buffer_info()[0], ctypes.c_void_p).value for tc in cases))
        all_a_vals = PtrArray(*(ctypes.addressof(tc["a_vals"]) for tc in cases))
        all_b_i8 = PtrArray(*(ctypes.addressof(tc["b_i8"]) for tc in cases))

        _lib.sparse_matmul_batch_dense_axpy(
            n,
            ctypes.addressof(all_rows_a),
            ctypes.addressof(all_cols_a),
            ctypes.addressof(all_cols_b),
            ctypes.addressof(all_a_rowptr),
            ctypes.addressof(all_a_colidx),
            ctypes.addressof(all_a_vals),
            ctypes.addressof(all_b_i8),
            ctypes.addressof(all_result),
            ctypes.addressof(latencies_ns),
        )
    elif method == "blas":
        all_a_f32 = PtrArray(*(ctypes.addressof(tc["a_f32"]) for tc in cases))
        all_b_f32 = PtrArray(*(ctypes.addressof(tc["b_f32"]) for tc in cases))
        all_c_f32 = PtrArray(*(ctypes.addressof(tc["c_f32"]) for tc in cases))

        _lib.sparse_matmul_batch_blas(
            n,
            ctypes.addressof(all_rows_a),
            ctypes.addressof(all_cols_a),
            ctypes.addressof(all_cols_b),
            ctypes.addressof(all_a_f32),
            ctypes.addressof(all_b_f32),
            ctypes.addressof(all_c_f32),
            ctypes.addressof(all_result),
            ctypes.addressof(latencies_ns),
        )
    elif method == "blas_parallel":
        all_a_f32 = PtrArray(*(ctypes.addressof(tc["a_f32"]) for tc in cases))
        all_b_f32 = PtrArray(*(ctypes.addressof(tc["b_f32"]) for tc in cases))
        all_c_f32 = PtrArray(*(ctypes.addressof(tc["c_f32"]) for tc in cases))

        _lib.sparse_matmul_batch_blas_parallel(
            n,
            ctypes.addressof(all_rows_a),
            ctypes.addressof(all_cols_a),
            ctypes.addressof(all_cols_b),
            ctypes.addressof(all_a_f32),
            ctypes.addressof(all_b_f32),
            ctypes.addressof(all_c_f32),
            ctypes.addressof(all_result),
            ctypes.addressof(latencies_ns),
            6,
        )
    elif method == "dense_axpy_parallel":
        all_a_rowptr = PtrArray(*(tc["a_rowptr"].buffer_info()[0] for tc in cases))
        all_a_colidx = PtrArray(*(ctypes.cast(tc["a_colidx"].buffer_info()[0], ctypes.c_void_p).value for tc in cases))
        all_a_vals = PtrArray(*(ctypes.addressof(tc["a_vals"]) for tc in cases))
        all_b_i8 = PtrArray(*(ctypes.addressof(tc["b_i8"]) for tc in cases))

        _lib.sparse_matmul_batch_dense_axpy_parallel(
            n,
            ctypes.addressof(all_rows_a),
            ctypes.addressof(all_cols_a),
            ctypes.addressof(all_cols_b),
            ctypes.addressof(all_a_rowptr),
            ctypes.addressof(all_a_colidx),
            ctypes.addressof(all_a_vals),
            ctypes.addressof(all_b_i8),
            ctypes.addressof(all_result),
            ctypes.addressof(latencies_ns),
            6,
        )
    elif method == "parallel":
        all_a_rowptr = PtrArray(*(tc["a_rowptr"].buffer_info()[0] for tc in cases))
        all_a_colidx = PtrArray(*(ctypes.cast(tc["a_colidx"].buffer_info()[0], ctypes.c_void_p).value for tc in cases))
        all_a_vals = PtrArray(*(ctypes.addressof(tc["a_vals"]) for tc in cases))

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
            6,
        )
    else:
        all_a_rowptr = PtrArray(*(tc["a_rowptr"].buffer_info()[0] for tc in cases))
        all_a_colidx = PtrArray(*(ctypes.cast(tc["a_colidx"].buffer_info()[0], ctypes.c_void_p).value for tc in cases))
        all_a_vals = PtrArray(*(ctypes.addressof(tc["a_vals"]) for tc in cases))

        _lib.sparse_matmul_batch(
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
        )

    return [latencies_ns[i] / 1e6 for i in range(n)]  # ns -> ms


def log(msg, file=None):
    print(msg)
    if file:
        file.write(msg + "\n")


def run_experiment(cases, method="serial", num_runs=5):
    """Run experiment num_runs times, return (mean, std, per_case_latencies)."""
    import statistics
    all_avgs = []
    last_latencies = None
    for run in range(num_runs):
        latencies_ms = run_batch(cases, method=method)
        avg = sum(latencies_ms) / len(latencies_ms)
        all_avgs.append(avg)
        last_latencies = latencies_ms
    mean = statistics.mean(all_avgs)
    std = statistics.stdev(all_avgs) if len(all_avgs) > 1 else 0.0
    return mean, std, last_latencies


def main():
    import sys
    results = []
    log_file = open("output.log", "w")
    cases = load_test_cases()

    # Always run both serial (core algorithm) and parallel
    serial_mean, serial_std, serial_latencies = run_experiment(cases, method="blas")
    parallel_mean, parallel_std, parallel_latencies = run_experiment(cases, method="blas_parallel")

    # Use serial run for correctness checking (int32 result vs int32 expected)
    for i, tc in enumerate(cases):
        name = tc["name"]
        rb = tc["result_buf_i32"]
        s_lat = serial_latencies[i]
        p_lat = parallel_latencies[i]

        try:
            assert rb == tc["expected_i32"], "Output mismatch"
            solution = "correct"
            observation = "Output matches expected result"
            log(f"  PASS  {name} (serial: {s_lat:.4f} ms, parallel: {p_lat:.4f} ms)", log_file)
        except AssertionError as e:
            solution = "incorrect"
            observation = str(e)
            log(f"  FAIL  {name} (serial: {s_lat:.4f} ms, parallel: {p_lat:.4f} ms) — {e}", log_file)

        results.append({
            "test_case": name,
            "solution": solution,
            "serial_latency": f"{s_lat:.4f} ms",
            "parallel_latency": f"{p_lat:.4f} ms",
            "observation": observation,
        })

    with open("results.tsv", "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["test_case", "solution", "serial_latency", "parallel_latency", "observation"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(results)

        passed = sum(1 for r in results if r["solution"] == "correct")

        writer.writerow({
            "test_case": "SUMMARY",
            "solution": f"{passed}/{len(results)} passed",
            "serial_latency": f"{serial_mean:.4f} ms",
            "parallel_latency": f"{parallel_mean:.4f} ms",
            "observation": f"mean ± std over 5 runs",
        })

    log(f"\n{passed}/{len(results)} passed", log_file)
    log(f"Serial:   {serial_mean:.4f} ± {serial_std:.4f} ms (core algorithm)", log_file)
    log(f"Parallel: {parallel_mean:.4f} ± {parallel_std:.4f} ms (6 threads)", log_file)
    log(f"Results written to results.tsv", log_file)
    log_file.close()
    print(f"Serial:   {serial_mean:.4f} ± {serial_std:.4f} ms (core algorithm)")
    print(f"Parallel: {parallel_mean:.4f} ± {parallel_std:.4f} ms (6 threads)")


if __name__ == "__main__":
    main()
