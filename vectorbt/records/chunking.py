# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""Extensions to `vectorbt.utils.chunking` for records and mapped arrays."""

import numpy as np

from vectorbt import _typing as tp
from vectorbt.utils.config import Config
from vectorbt.utils.chunking import (
    ArgGetterMixin,
    ChunkMeta,
    ArgSizer,
    ChunkSlicer,
    ArraySlicer,
    CountAdapter,
    ChunkMapper,
    ann_argsT
)
from vectorbt.generic.chunking import get_group_lens_slice


class ColMapSizer(ArgSizer):
    """Class for getting the size from a column map."""

    def get_size(self, ann_args: ann_argsT) -> int:
        return len(self.get_arg(ann_args)[1])


class ColMapSlicer(ChunkSlicer):
    """Class for slicing multiple elements from a column map based on the chunk range."""

    def take(self, obj: tp.ColMap, chunk_meta: ChunkMeta, **kwargs) -> tp.ColMap:
        col_idxs, col_lens = obj
        col_lens = col_lens[chunk_meta.start:chunk_meta.end]
        return np.arange(np.sum(col_lens)), col_lens


col_map_config = Config(
    dict(
        size=ColMapSizer('col_map'),
        arg_take_spec={'col_map': ColMapSlicer()}
    )
)
"""Config for slicing a column map."""


class ColIdxsMapper(ChunkMapper, ArgGetterMixin):
    """Class for mapping chunk metadata to column indices using column lengths."""

    def __init__(self, arg: tp.Union[str, int]) -> None:
        ChunkMapper.__init__(self)
        ArgGetterMixin.__init__(self, arg)

    def map(self, chunk_meta: ChunkMeta, ann_args: tp.Optional[ann_argsT] = None, **kwargs) -> ChunkMeta:
        col_map = self.get_arg(ann_args)
        col_idxs, col_lens = col_map
        col_lens_slice = get_group_lens_slice(col_lens, chunk_meta)
        return ChunkMeta(
            idx=chunk_meta.idx,
            start=None,
            end=None,
            indices=col_idxs[col_lens_slice]
        )


arr_config = Config(
    dict(
        arg_take_spec={
            'mapped_arr': ArraySlicer(0, mapper=ColIdxsMapper('col_map')),
            'col_arr': ArraySlicer(0, mapper=ColIdxsMapper('col_map')),
            'idx_arr': ArraySlicer(0, mapper=ColIdxsMapper('col_map')),
            'id_arr': ArraySlicer(0, mapper=ColIdxsMapper('col_map'))
        }
    )
)
"""Config for slicing any array based on a column map."""


arr_len_config = Config(
    dict(
        arg_take_spec={
            'n_mapped': CountAdapter(mapper=ColIdxsMapper('col_map')),
            'n_records': CountAdapter(mapper=ColIdxsMapper('col_map'))
        }
    )
)
"""Config for adapting the length of any array based on a column map."""
