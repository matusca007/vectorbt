# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""Numba-compiled functions.

Provides an arsenal of Numba-compiled functions for records and mapped arrays.
These only accept NumPy arrays and other Numba-compatible types.

!!! note
    vectorbt treats matrices as first-class citizens and expects input arrays to be
    2-dim, unless function has suffix `_1d` or is meant to be input to another function.
    Data is processed along index (axis 0).

    All functions passed as argument must be Numba-compiled.

    Records must retain the order they were created in."""

import numpy as np
from numba import prange
from numba.np.numpy_support import as_dtype

from vectorbt import _typing as tp
from vectorbt.nb_registry import register_jit, register_generated_jit
from vectorbt.generic import nb as generic_nb


# ############# Generation ############# #


@register_jit(cache=True)
def generate_ids_nb(col_arr: tp.Array1d, n_cols: int) -> tp.Array1d:
    """Generate the monotonically increasing id array based on the column index array."""
    col_idxs = np.full(n_cols, 0, dtype=np.int_)
    out = np.empty_like(col_arr)
    for c in range(len(col_arr)):
        out[c] = col_idxs[col_arr[c]]
        col_idxs[col_arr[c]] += 1
    return out


# ############# Indexing ############# #


@register_jit(cache=True)
def col_range_nb(col_arr: tp.Array1d, n_cols: int) -> tp.ColRange:
    """Build column range for sorted column array.

    Creates a 2-dim array with first column being start indices (inclusive) and
    second column being end indices (exclusive).

    !!! note
        Requires `col_arr` to be in ascending order. This can be done by sorting."""
    col_range = np.full((n_cols, 2), -1, dtype=np.int_)
    last_col = -1

    for c in range(col_arr.shape[0]):
        col = col_arr[c]
        if col < last_col:
            raise ValueError("col_arr must be in ascending order")
        if col != last_col:
            if last_col != -1:
                col_range[last_col, 1] = c
            col_range[col, 0] = c
            last_col = col
        if c == col_arr.shape[0] - 1:
            col_range[col, 1] = c + 1
    return col_range


@register_jit(cache=True)
def col_range_select_nb(col_range: tp.ColRange, new_cols: tp.Array1d) -> tp.Tuple[tp.Array1d, tp.Array1d]:
    """Perform indexing on a sorted array using column range `col_range`.

    Returns indices of elements corresponding to columns in `new_cols` and a new column array."""
    col_range = col_range[new_cols]
    new_n = np.sum(col_range[:, 1] - col_range[:, 0])
    indices_out = np.empty(new_n, dtype=np.int_)
    col_arr_out = np.empty(new_n, dtype=np.int_)
    j = 0

    for c in range(new_cols.shape[0]):
        from_r = col_range[c, 0]
        to_r = col_range[c, 1]
        if from_r == -1 or to_r == -1:
            continue
        rang = np.arange(from_r, to_r)
        indices_out[j:j + rang.shape[0]] = rang
        col_arr_out[j:j + rang.shape[0]] = c
        j += rang.shape[0]
    return indices_out, col_arr_out


@register_jit(cache=True)
def record_col_range_select_nb(records: tp.RecordArray, col_range: tp.ColRange,
                               new_cols: tp.Array1d) -> tp.RecordArray:
    """Perform indexing on sorted records using column range `col_range`.

    Returns new records."""
    col_range = col_range[new_cols]
    new_n = np.sum(col_range[:, 1] - col_range[:, 0])
    out = np.empty(new_n, dtype=records.dtype)
    j = 0

    for c in range(new_cols.shape[0]):
        from_r = col_range[c, 0]
        to_r = col_range[c, 1]
        if from_r == -1 or to_r == -1:
            continue
        col_records = np.copy(records[from_r:to_r])
        col_records['col'][:] = c  # don't forget to assign new column indices
        out[j:j + col_records.shape[0]] = col_records
        j += col_records.shape[0]
    return out


@register_jit(cache=True)
def col_map_nb(col_arr: tp.Array1d, n_cols: int) -> tp.ColMap:
    """Build a map between columns and their indices.

    Returns an array with indices segmented by column, and an array with count per segment.

    Works well for unsorted column arrays."""
    col_lens_out = np.full(n_cols, 0, dtype=np.int_)
    for c in range(col_arr.shape[0]):
        col = col_arr[c]
        col_lens_out[col] += 1

    col_start_idxs = np.cumsum(col_lens_out) - col_lens_out
    col_idxs_out = np.empty((col_arr.shape[0],), dtype=np.int_)
    col_i = np.full(n_cols, 0, dtype=np.int_)
    for c in range(col_arr.shape[0]):
        col = col_arr[c]
        col_idxs_out[col_start_idxs[col] + col_i[col]] = c
        col_i[col] += 1

    return col_idxs_out, col_lens_out


@register_jit(cache=True)
def col_map_select_nb(col_map: tp.ColMap, new_cols: tp.Array1d) -> tp.Tuple[tp.Array1d, tp.Array1d]:
    """Same as `mapped_col_range_select_nb` but using column map `col_map`."""
    col_idxs, col_lens = col_map
    col_start_idxs = np.cumsum(col_lens) - col_lens
    total_count = np.sum(col_lens[new_cols])
    idxs_out = np.empty(total_count, dtype=np.int_)
    col_arr_out = np.empty(total_count, dtype=np.int_)
    j = 0

    for new_col_i in range(len(new_cols)):
        new_col = new_cols[new_col_i]
        col_len = col_lens[new_col]
        if col_len == 0:
            continue
        col_start_idx = col_start_idxs[new_col]
        idxs = col_idxs[col_start_idx:col_start_idx + col_len]
        idxs_out[j:j + col_len] = idxs
        col_arr_out[j:j + col_len] = new_col_i
        j += col_len
    return idxs_out, col_arr_out


@register_jit(cache=True)
def record_col_map_select_nb(records: tp.RecordArray, col_map: tp.ColMap, new_cols: tp.Array1d) -> tp.RecordArray:
    """Same as `record_col_range_select_nb` but using column map `col_map`."""
    col_idxs, col_lens = col_map
    col_start_idxs = np.cumsum(col_lens) - col_lens
    out = np.empty(np.sum(col_lens[new_cols]), dtype=records.dtype)
    j = 0

    for new_col_i in range(len(new_cols)):
        new_col = new_cols[new_col_i]
        col_len = col_lens[new_col]
        if col_len == 0:
            continue
        col_start_idx = col_start_idxs[new_col]
        idxs = col_idxs[col_start_idx:col_start_idx + col_len]
        col_records = np.copy(records[idxs])
        col_records['col'][:] = new_col_i
        out[j:j + col_len] = col_records
        j += col_len
    return out


# ############# Sorting ############# #


@register_jit(cache=True)
def is_col_sorted_nb(col_arr: tp.Array1d) -> bool:
    """Check whether the column array is sorted."""
    for i in range(len(col_arr) - 1):
        if col_arr[i + 1] < col_arr[i]:
            return False
    return True


@register_jit(cache=True)
def is_col_idx_sorted_nb(col_arr: tp.Array1d, id_arr: tp.Array1d) -> bool:
    """Check whether the column and index arrays are sorted."""
    for i in range(len(col_arr) - 1):
        if col_arr[i + 1] < col_arr[i]:
            return False
        if col_arr[i + 1] == col_arr[i] and id_arr[i + 1] < id_arr[i]:
            return False
    return True


# ############# Mapping ############# #


@register_jit(cache=True, tags={'can_parallel'})
def top_n_mapped_nb(mapped_arr: tp.Array1d, col_map: tp.ColMap, n: int) -> tp.Array1d:
    """Returns mask of top N mapped elements."""
    col_idxs, col_lens = col_map
    col_start_idxs = np.cumsum(col_lens) - col_lens
    out = np.full(mapped_arr.shape[0], False, dtype=np.bool_)

    for col in prange(col_lens.shape[0]):
        col_len = col_lens[col]
        if col_len == 0:
            continue
        col_start_idx = col_start_idxs[col]
        idxs = col_idxs[col_start_idx:col_start_idx + col_len]
        out[idxs[np.argsort(mapped_arr[idxs])[-n:]]] = True
    return out


@register_jit(cache=True, tags={'can_parallel'})
def bottom_n_mapped_nb(mapped_arr: tp.Array1d, col_map: tp.ColMap, n: int) -> tp.Array1d:
    """Returns mask of bottom N mapped elements."""
    col_idxs, col_lens = col_map
    col_start_idxs = np.cumsum(col_lens) - col_lens
    out = np.full(mapped_arr.shape[0], False, dtype=np.bool_)

    for col in prange(col_lens.shape[0]):
        col_len = col_lens[col]
        if col_len == 0:
            continue
        col_start_idx = col_start_idxs[col]
        idxs = col_idxs[col_start_idx:col_start_idx + col_len]
        out[idxs[np.argsort(mapped_arr[idxs])[:n]]] = True
    return out


@register_jit(tags={'can_parallel'})
def apply_on_mapped_nb(mapped_arr: tp.Array1d, col_map: tp.ColMap,
                       apply_func_nb: tp.MappedApplyFunc, *args) -> tp.Array1d:
    """Apply function on mapped array per column.

    Returns the same shape as `mapped_arr`.

    `apply_func_nb` must accept the values of the column and `*args`. Must return an array."""
    col_idxs, col_lens = col_map
    col_start_idxs = np.cumsum(col_lens) - col_lens
    out = np.empty(mapped_arr.shape[0], dtype=np.float_)

    for col in range(col_lens.shape[0]):
        col_len = col_lens[col]
        if col_len == 0:
            continue
        col_start_idx = col_start_idxs[col]
        idxs = col_idxs[col_start_idx:col_start_idx + col_len]
        out[idxs] = apply_func_nb(mapped_arr[idxs], *args)
    return out


@register_jit(tags={'can_parallel'})
def apply_on_mapped_meta_nb(n_mapped: int, col_map: tp.ColMap,
                            apply_func_nb: tp.MappedApplyMetaFunc, *args) -> tp.Array1d:
    """Meta version of `apply_on_mapped_nb`.

    `apply_func_nb` must accept the mapped indices, the column index, and `*args`. Must return an array."""
    col_idxs, col_lens = col_map
    col_start_idxs = np.cumsum(col_lens) - col_lens
    out = np.empty(n_mapped, dtype=np.float_)

    for col in range(col_lens.shape[0]):
        col_len = col_lens[col]
        if col_len == 0:
            continue
        col_start_idx = col_start_idxs[col]
        idxs = col_idxs[col_start_idx:col_start_idx + col_len]
        out[idxs] = apply_func_nb(idxs, col, *args)
    return out


@register_jit(tags={'can_parallel'})
def map_records_nb(records: tp.RecordArray, map_func_nb: tp.RecordsMapFunc, *args) -> tp.Array1d:
    """Map each record to a single value.

    `map_func_nb` must accept a single record and `*args`. Must return a single value."""
    out = np.empty(records.shape[0], dtype=np.float_)

    for ridx in prange(records.shape[0]):
        out[ridx] = map_func_nb(records[ridx], *args)
    return out


@register_jit(tags={'can_parallel'})
def map_records_meta_nb(n_records: int, map_func_nb: tp.MappedReduceMetaFunc, *args) -> tp.Array1d:
    """Meta version of `map_records_nb`.

    `map_func_nb` must accept the record index and `*args`. Must return a single value."""
    out = np.empty(n_records, dtype=np.float_)

    for ridx in prange(n_records):
        out[ridx] = map_func_nb(ridx, *args)
    return out


@register_jit(tags={'can_parallel'})
def apply_on_records_nb(records: tp.RecordArray, col_map: tp.ColMap,
                        apply_func_nb: tp.RecordsApplyFunc, *args) -> tp.Array1d:
    """Apply function on records per column.

    Returns the same shape as `records`.

    `apply_func_nb` must accept the records of the column and `*args`. Must return an array."""
    col_idxs, col_lens = col_map
    col_start_idxs = np.cumsum(col_lens) - col_lens
    out = np.empty(records.shape[0], dtype=np.float_)

    for col in prange(col_lens.shape[0]):
        col_len = col_lens[col]
        if col_len == 0:
            continue
        col_start_idx = col_start_idxs[col]
        idxs = col_idxs[col_start_idx:col_start_idx + col_len]
        out[idxs] = apply_func_nb(records[idxs], *args)
    return out


@register_jit(tags={'can_parallel'})
def apply_on_records_meta_nb(n_records: int, col_map: tp.ColMap,
                             apply_func_nb: tp.RecordsApplyMetaFunc, *args) -> tp.Array1d:
    """Meta version of `apply_on_records_nb`.

    `apply_func_nb` must accept the record indices, the column index, and `*args`. Must return an array."""
    col_idxs, col_lens = col_map
    col_start_idxs = np.cumsum(col_lens) - col_lens
    out = np.empty(n_records, dtype=np.float_)

    for col in prange(col_lens.shape[0]):
        col_len = col_lens[col]
        if col_len == 0:
            continue
        col_start_idx = col_start_idxs[col]
        idxs = col_idxs[col_start_idx:col_start_idx + col_len]
        out[idxs] = apply_func_nb(idxs, col, *args)
    return out


# ############# Coverage ############# #


@register_jit(cache=True)
def mapped_has_conflicts_nb(col_arr: tp.Array1d, idx_arr: tp.Array1d, target_shape: tp.Shape) -> bool:
    """Check whether mapped array has positional conflicts."""
    temp = np.zeros(target_shape)

    for i in range(len(col_arr)):
        if temp[idx_arr[i], col_arr[i]] > 0:
            return True
        temp[idx_arr[i], col_arr[i]] = 1
    return False


@register_jit(cache=True)
def mapped_coverage_map_nb(col_arr: tp.Array1d, idx_arr: tp.Array1d, target_shape: tp.Shape) -> tp.Array2d:
    """Get the coverage map of a mapped array.

    Each element corresponds to the number of times it was referenced (= duplicates of `col_arr` and `idx_arr`).
    More than one depicts a positional conflict."""
    out = np.zeros(target_shape, dtype=np.int_)

    for i in range(len(col_arr)):
        out[idx_arr[i], col_arr[i]] += 1
    return out


# ############# Unstacking ############# #


@register_generated_jit(cache=True)
def unstack_mapped_nb(mapped_arr: tp.Array1d, col_arr: tp.Array1d, idx_arr: tp.Array1d,
                      target_shape: tp.Shape, fill_value: float) -> tp.Array2d:
    """Unstack mapped array using index data."""
    nb_enabled = not isinstance(mapped_arr, np.ndarray)
    if nb_enabled:
        mapped_arr_dtype = as_dtype(mapped_arr.dtype)
        fill_value_dtype = as_dtype(fill_value)
    else:
        mapped_arr_dtype = mapped_arr.dtype
        fill_value_dtype = np.array(fill_value).dtype
    dtype = np.promote_types(mapped_arr_dtype, fill_value_dtype)

    def _unstack_mapped_nb(mapped_arr, col_arr, idx_arr, target_shape, fill_value):
        out = np.full(target_shape, fill_value, dtype=dtype)

        for r in range(mapped_arr.shape[0]):
            out[idx_arr[r], col_arr[r]] = mapped_arr[r]
        return out

    if not nb_enabled:
        return _unstack_mapped_nb(mapped_arr, col_arr, idx_arr, target_shape, fill_value)

    return _unstack_mapped_nb


@register_generated_jit(cache=True)
def ignore_unstack_mapped_nb(mapped_arr: tp.Array1d, col_map: tp.ColMap, fill_value: float) -> tp.Array2d:
    """Unstack mapped array by ignoring index data."""
    nb_enabled = not isinstance(mapped_arr, np.ndarray)
    if nb_enabled:
        mapped_arr_dtype = as_dtype(mapped_arr.dtype)
        fill_value_dtype = as_dtype(fill_value)
    else:
        mapped_arr_dtype = mapped_arr.dtype
        fill_value_dtype = np.array(fill_value).dtype
    dtype = np.promote_types(mapped_arr_dtype, fill_value_dtype)

    def _ignore_unstack_mapped_nb(mapped_arr, col_map, fill_value):
        col_idxs, col_lens = col_map
        col_start_idxs = np.cumsum(col_lens) - col_lens
        out = np.full((np.max(col_lens), col_lens.shape[0]), fill_value, dtype=dtype)

        for col in range(col_lens.shape[0]):
            col_len = col_lens[col]
            if col_len == 0:
                continue
            col_start_idx = col_start_idxs[col]
            idxs = col_idxs[col_start_idx:col_start_idx + col_len]
            out[:col_len, col] = mapped_arr[idxs]

        return out

    if not nb_enabled:
        return _ignore_unstack_mapped_nb(mapped_arr, col_map, fill_value)

    return _ignore_unstack_mapped_nb


@register_jit(cache=True)
def unstack_index_nb(repeat_cnt_arr: tp.Array1d) -> tp.Array1d:
    """Unstack index using the number of times each element must repeat.

    `repeat_cnt_arr` can be created from the coverage map."""
    out = np.empty(np.sum(repeat_cnt_arr), dtype=np.int_)

    k = 0
    for i in range(len(repeat_cnt_arr)):
        out[k:k + repeat_cnt_arr[i]] = i
        k += repeat_cnt_arr[i]
    return out


@register_generated_jit(cache=True)
def repeat_unstack_mapped_nb(mapped_arr: tp.Array1d, col_arr: tp.Array1d, idx_arr: tp.Array1d,
                             repeat_cnt_arr: tp.Array1d, n_cols: int, fill_value: float) -> tp.Array2d:
    """Unstack mapped array using repeated index data."""
    nb_enabled = not isinstance(mapped_arr, np.ndarray)
    if nb_enabled:
        mapped_arr_dtype = as_dtype(mapped_arr.dtype)
        fill_value_dtype = as_dtype(fill_value)
    else:
        mapped_arr_dtype = mapped_arr.dtype
        fill_value_dtype = np.array(fill_value).dtype
    dtype = np.promote_types(mapped_arr_dtype, fill_value_dtype)

    def _repeat_unstack_mapped_nb(mapped_arr, col_arr, idx_arr, repeat_cnt_arr, n_cols, fill_value):
        index_start_arr = np.cumsum(repeat_cnt_arr) - repeat_cnt_arr
        out = np.full((np.sum(repeat_cnt_arr), n_cols), fill_value, dtype=dtype)
        temp = np.zeros((len(repeat_cnt_arr), n_cols), dtype=np.int_)

        for i in range(len(col_arr)):
            out[index_start_arr[idx_arr[i]] + temp[idx_arr[i], col_arr[i]], col_arr[i]] = mapped_arr[i]
            temp[idx_arr[i], col_arr[i]] += 1
        return out

    if not nb_enabled:
        return _repeat_unstack_mapped_nb(mapped_arr, col_arr, idx_arr, repeat_cnt_arr, n_cols, fill_value)

    return _repeat_unstack_mapped_nb


# ############# Reducing ############# #

@register_jit(tags={'can_parallel'})
def reduce_mapped_nb(mapped_arr: tp.Array1d, col_map: tp.ColMap, fill_value: float,
                     reduce_func_nb: tp.MappedReduceFunc, *args) -> tp.Array1d:
    """Reduce mapped array by column to a single value.

    Faster than `unstack_mapped_nb` and `vbt.*` used together, and also
    requires less memory. But does not take advantage of caching.

    `reduce_func_nb` must accept the mapped array and `*args`.
    Must return a single value."""
    col_idxs, col_lens = col_map
    col_start_idxs = np.cumsum(col_lens) - col_lens
    out = np.full(col_lens.shape[0], fill_value, dtype=np.float_)

    for col in prange(col_lens.shape[0]):
        col_len = col_lens[col]
        if col_len == 0:
            continue
        col_start_idx = col_start_idxs[col]
        idxs = col_idxs[col_start_idx:col_start_idx + col_len]
        out[col] = reduce_func_nb(mapped_arr[idxs], *args)
    return out


@register_jit(tags={'can_parallel'})
def reduce_mapped_meta_nb(col_map: tp.ColMap, fill_value: float,
                          reduce_func_nb: tp.MappedReduceMetaFunc, *args) -> tp.Array1d:
    """Meta version of `reduce_mapped_nb`.

    `reduce_func_nb` must accept the mapped indices, the column index, and `*args`.
    Must return a single value."""
    col_idxs, col_lens = col_map
    col_start_idxs = np.cumsum(col_lens) - col_lens
    out = np.full(col_lens.shape[0], fill_value, dtype=np.float_)

    for col in prange(col_lens.shape[0]):
        col_len = col_lens[col]
        if col_len == 0:
            continue
        col_start_idx = col_start_idxs[col]
        idxs = col_idxs[col_start_idx:col_start_idx + col_len]
        out[col] = reduce_func_nb(idxs, col, *args)
    return out


@register_jit(tags={'can_parallel'})
def reduce_mapped_to_idx_nb(mapped_arr: tp.Array1d, col_map: tp.ColMap, idx_arr: tp.Array1d, fill_value: float,
                            reduce_func_nb: tp.MappedReduceFunc, *args) -> tp.Array1d:
    """Reduce mapped array by column to an index.

    Same as `reduce_mapped_nb` except `idx_arr` must be passed.

    !!! note
        Must return integers or raise an exception."""
    col_idxs, col_lens = col_map
    col_start_idxs = np.cumsum(col_lens) - col_lens
    out = np.full(col_lens.shape[0], fill_value, dtype=np.float_)

    for col in prange(col_lens.shape[0]):
        col_len = col_lens[col]
        if col_len == 0:
            continue
        col_start_idx = col_start_idxs[col]
        idxs = col_idxs[col_start_idx:col_start_idx + col_len]
        col_out = reduce_func_nb(mapped_arr[idxs], *args)
        out[col] = idx_arr[idxs][col_out]
    return out


@register_jit(tags={'can_parallel'})
def reduce_mapped_to_idx_meta_nb(col_map: tp.ColMap, idx_arr: tp.Array1d, fill_value: float,
                                 reduce_func_nb: tp.MappedReduceMetaFunc, *args) -> tp.Array1d:
    """Meta version of `reduce_mapped_to_idx_nb`.

    `reduce_func_nb` is the same as in `reduce_mapped_meta_nb`."""
    col_idxs, col_lens = col_map
    col_start_idxs = np.cumsum(col_lens) - col_lens
    out = np.full(col_lens.shape[0], fill_value, dtype=np.float_)

    for col in prange(col_lens.shape[0]):
        col_len = col_lens[col]
        if col_len == 0:
            continue
        col_start_idx = col_start_idxs[col]
        idxs = col_idxs[col_start_idx:col_start_idx + col_len]
        col_out = reduce_func_nb(idxs, col, *args)
        out[col] = idx_arr[idxs][col_out]
    return out


@register_jit(tags={'can_parallel'})
def reduce_mapped_to_array_nb(mapped_arr: tp.Array1d, col_map: tp.ColMap, fill_value: float,
                              reduce_func_nb: tp.MappedReduceToArrayFunc, *args) -> tp.Array2d:
    """Reduce mapped array by column to an array.

    `reduce_func_nb` same as for `reduce_mapped_nb` but must return an array."""
    col_idxs, col_lens = col_map
    col_start_idxs = np.cumsum(col_lens) - col_lens
    for col in range(col_lens.shape[0]):
        col_len = col_lens[col]
        if col_len > 0:
            col_start_idx = col_start_idxs[col]
            col0, midxs0 = col, col_idxs[col_start_idx:col_start_idx + col_len]
            break

    col_0_out = reduce_func_nb(mapped_arr[midxs0], *args)
    out = np.full((col_0_out.shape[0], col_lens.shape[0]), fill_value, dtype=np.float_)
    for i in range(col_0_out.shape[0]):
        out[i, col0] = col_0_out[i]

    for col in prange(col0 + 1, col_lens.shape[0]):
        col_len = col_lens[col]
        if col_len == 0:
            continue
        col_start_idx = col_start_idxs[col]
        idxs = col_idxs[col_start_idx:col_start_idx + col_len]
        col_out = reduce_func_nb(mapped_arr[idxs], *args)
        for i in range(col_out.shape[0]):
            out[i, col] = col_out[i]
    return out


@register_jit(tags={'can_parallel'})
def reduce_mapped_to_array_meta_nb(col_map: tp.ColMap, fill_value: float,
                                   reduce_func_nb: tp.MappedReduceToArrayMetaFunc, *args) -> tp.Array2d:
    """Meta version of `reduce_mapped_to_array_nb`.

    `reduce_func_nb` is the same as in `reduce_mapped_meta_nb`."""
    col_idxs, col_lens = col_map
    col_start_idxs = np.cumsum(col_lens) - col_lens
    for col in range(col_lens.shape[0]):
        col_len = col_lens[col]
        if col_len > 0:
            col_start_idx = col_start_idxs[col]
            col0, midxs0 = col, col_idxs[col_start_idx:col_start_idx + col_len]
            break

    col_0_out = reduce_func_nb(midxs0, col0, *args)
    out = np.full((col_0_out.shape[0], col_lens.shape[0]), fill_value, dtype=np.float_)
    for i in range(col_0_out.shape[0]):
        out[i, col0] = col_0_out[i]

    for col in prange(col0 + 1, col_lens.shape[0]):
        col_len = col_lens[col]
        if col_len == 0:
            continue
        col_start_idx = col_start_idxs[col]
        idxs = col_idxs[col_start_idx:col_start_idx + col_len]
        col_out = reduce_func_nb(idxs, col, *args)
        for i in range(col_out.shape[0]):
            out[i, col] = col_out[i]
    return out


@register_jit(tags={'can_parallel'})
def reduce_mapped_to_idx_array_nb(mapped_arr: tp.Array1d, col_map: tp.ColMap, idx_arr: tp.Array1d, fill_value: float,
                                  reduce_func_nb: tp.MappedReduceToArrayFunc, *args) -> tp.Array2d:
    """Reduce mapped array by column to an index array.

    Same as `reduce_mapped_to_array_nb` except `idx_arr` must be passed.

    !!! note
        Must return integers or raise an exception."""
    col_idxs, col_lens = col_map
    col_start_idxs = np.cumsum(col_lens) - col_lens
    for col in range(col_lens.shape[0]):
        col_len = col_lens[col]
        if col_len > 0:
            col_start_idx = col_start_idxs[col]
            col0, midxs0 = col, col_idxs[col_start_idx:col_start_idx + col_len]
            break

    col_0_out = reduce_func_nb(mapped_arr[midxs0], *args)
    out = np.full((col_0_out.shape[0], col_lens.shape[0]), fill_value, dtype=np.float_)
    for i in range(col_0_out.shape[0]):
        out[i, col0] = idx_arr[midxs0[col_0_out[i]]]

    for col in prange(col0 + 1, col_lens.shape[0]):
        col_len = col_lens[col]
        if col_len == 0:
            continue
        col_start_idx = col_start_idxs[col]
        idxs = col_idxs[col_start_idx:col_start_idx + col_len]
        col_out = reduce_func_nb(mapped_arr[idxs], *args)
        for i in range(col_0_out.shape[0]):
            out[i, col] = idx_arr[idxs[col_out[i]]]
    return out


@register_jit(tags={'can_parallel'})
def reduce_mapped_to_idx_array_meta_nb(col_map: tp.ColMap, idx_arr: tp.Array1d, fill_value: float,
                                       reduce_func_nb: tp.MappedReduceToArrayMetaFunc, *args) -> tp.Array2d:
    """Meta version of `reduce_mapped_to_idx_array_nb`.

    `reduce_func_nb` is the same as in `reduce_mapped_meta_nb`."""
    col_idxs, col_lens = col_map
    col_start_idxs = np.cumsum(col_lens) - col_lens
    for col in range(col_lens.shape[0]):
        col_len = col_lens[col]
        if col_len > 0:
            col_start_idx = col_start_idxs[col]
            col0, midxs0 = col, col_idxs[col_start_idx:col_start_idx + col_len]
            break

    col_0_out = reduce_func_nb(midxs0, col0, *args)
    out = np.full((col_0_out.shape[0], col_lens.shape[0]), fill_value, dtype=np.float_)
    for i in range(col_0_out.shape[0]):
        out[i, col0] = idx_arr[midxs0[col_0_out[i]]]

    for col in prange(col0 + 1, col_lens.shape[0]):
        col_len = col_lens[col]
        if col_len == 0:
            continue
        col_start_idx = col_start_idxs[col]
        idxs = col_idxs[col_start_idx:col_start_idx + col_len]
        col_out = reduce_func_nb(idxs, col, *args)
        for i in range(col_0_out.shape[0]):
            out[i, col] = idx_arr[idxs[col_out[i]]]
    return out


# ############# Value counts ############# #


@register_jit(cache=True, tags={'can_parallel'})
def mapped_value_counts_per_col_nb(codes: tp.Array1d, n_uniques: int, col_map: tp.ColMap) -> tp.Array2d:
    """Get value counts per column/group of an already factorized mapped array."""
    col_idxs, col_lens = col_map
    col_start_idxs = np.cumsum(col_lens) - col_lens
    out = np.full((n_uniques, col_lens.shape[0]), 0, dtype=np.int_)

    for col in prange(col_lens.shape[0]):
        col_len = col_lens[col]
        if col_len == 0:
            continue
        col_start_idx = col_start_idxs[col]
        idxs = col_idxs[col_start_idx:col_start_idx + col_len]
        out[:, col] = generic_nb.value_counts_1d_nb(codes[idxs], n_uniques)
    return out


@register_jit(cache=True)
def mapped_value_counts_per_row_nb(mapped_arr: tp.Array1d, n_uniques: int,
                                   idx_arr: tp.Array1d, n_rows: int) -> tp.Array2d:
    """Get value counts per row of an already factorized mapped array."""
    out = np.full((n_uniques, n_rows), 0, dtype=np.int_)

    for c in range(mapped_arr.shape[0]):
        out[mapped_arr[c], idx_arr[c]] += 1
    return out


@register_jit(cache=True)
def mapped_value_counts_nb(mapped_arr: tp.Array1d, n_uniques: int) -> tp.Array2d:
    """Get value counts globally of an already factorized mapped array."""
    out = np.full(n_uniques, 0, dtype=np.int_)

    for c in range(mapped_arr.shape[0]):
        out[mapped_arr[c]] += 1
    return out
