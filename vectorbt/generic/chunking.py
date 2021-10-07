# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""Extensions to `vectorbt.utils.chunking` for chunking generic functions."""

import numpy as np
from numba.typed import List

from vectorbt import _typing as tp
from vectorbt.utils.config import Config
from vectorbt.utils.chunking import (
    ArgGetterMixin,
    ChunkMeta,
    ArgSizer,
    ArraySizer,
    ShapeSizer,
    CountAdapter,
    ShapeSlicer,
    ArraySlicer,
    ChunkMapper
)

n_cols_config = Config(
    dict(
        size=ArgSizer('n_cols'),
        arg_take_spec={'n_cols': CountAdapter()}
    )
)
"""Config for adapting the number of columns."""

target_shape_ax0_config = Config(
    dict(
        size=ShapeSizer('target_shape', 0),
        arg_take_spec={'target_shape': ShapeSlicer(0)}
    )
)
"""Config for slicing a target shape along the first axis (rows)."""

target_shape_ax1_config = Config(
    dict(
        size=ShapeSizer('target_shape', 1),
        arg_take_spec={'target_shape': ShapeSlicer(1)}
    )
)
"""Config for slicing a target shape along the second axis (columns)."""

arr_ax0_config = Config(
    dict(
        size=ArraySizer('arr', 0),
        arg_take_spec={'arr': ArraySlicer(0)}
    )
)
"""Config for slicing an array along the first axis (rows)."""

arr_ax1_config = Config(
    dict(
        size=ArraySizer('arr', 1),
        arg_take_spec={'arr': ArraySlicer(1)}
    )
)
"""Config for slicing an array along the second axis (columns)."""


def get_group_lens_slice(group_lens: tp.Array1d, chunk_meta: ChunkMeta) -> slice:
    """Get slice of each chunk in group lengths."""
    group_lens_cumsum = np.cumsum(group_lens[:chunk_meta.end])
    start = group_lens_cumsum[chunk_meta.start] - group_lens[chunk_meta.start]
    end = group_lens_cumsum[-1]
    return slice(start, end)


class GroupLensMapper(ChunkMapper, ArgGetterMixin):
    """Class for mapping chunk metadata using group lengths."""

    def __init__(self, arg_query: tp.AnnArgQuery) -> None:
        ChunkMapper.__init__(self)
        ArgGetterMixin.__init__(self, arg_query)

    def map(self, chunk_meta: ChunkMeta, ann_args: tp.Optional[tp.AnnArgs] = None, **kwargs) -> ChunkMeta:
        group_lens = self.get_arg(ann_args)
        group_lens_slice = get_group_lens_slice(group_lens, chunk_meta)
        return ChunkMeta(
            idx=chunk_meta.idx,
            start=group_lens_slice.start,
            end=group_lens_slice.stop,
            indices=None
        )


group_lens_config = Config(
    dict(
        size=ArraySizer('group_lens', 0),
        arg_take_spec={'group_lens': ArraySlicer(0)}
    )
)
"""Config for slicing group lengths."""

arr_group_lens_config = Config(
    dict(arg_take_spec={'arr': ArraySlicer(1, mapper=GroupLensMapper('group_lens'))})
)
"""Config for slicing an array based on group lengths."""


def column_stack(results: list) -> tp.MaybeTuple[list]:
    """Stack columns from each array in results. Supports multiple arrays per result."""
    if isinstance(results[0], (tuple, list, List)):
        return tuple(map(np.column_stack, zip(*results)))
    return np.column_stack(results)


column_stack_config = Config(
    dict(merge_func=column_stack)
)
"""Config for merging arrays through horizontal stacking (column-wise)."""


def row_stack(results: list) -> tp.MaybeTuple[list]:
    """Stack rows from each array in results. Supports multiple arrays per result."""
    if isinstance(results[0], (tuple, list, List)):
        return tuple(map(np.row_stack, zip(*results)))
    return np.row_stack(results)


row_stack_config = Config(
    dict(merge_func=row_stack)
)
"""Config for merging arrays through vertical stacking (row-wise)."""


def concat(results: list) -> tp.MaybeTuple[list]:
    """Concatenate elements from each array in results. Supports multiple arrays per result."""
    if isinstance(results[0], (tuple, list, List)):
        return tuple(map(np.concatenate, zip(*results)))
    return np.concatenate(results)


concat_config = Config(
    dict(merge_func=concat)
)
"""Config for merging arrays through concatenating."""
