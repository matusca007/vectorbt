# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""Extensions to `vectorbt.utils.chunking`."""

import numpy as np
import uuid

from vectorbt import _typing as tp
from vectorbt.utils.chunking import (
    ArgGetterMixin,
    ChunkMeta,
    ArgSizer,
    ChunkSlicer,
    ChunkMapper
)
from vectorbt.base.chunking import get_group_lens_slice


class ColLensSizer(ArgSizer):
    """Class for getting the size from column lengths.

    Argument can be either a column map tuple or a column lengths array."""

    def get_size(self, ann_args: tp.AnnArgs) -> int:
        arg = self.get_arg(ann_args)
        if isinstance(arg, tuple):
            return len(arg[1])
        return len(arg)


class ColLensSlicer(ChunkSlicer):
    """Class for slicing multiple elements from column lengths based on the chunk range."""

    def take(self, obj: tp.Optional[tp.Union[tp.ColLens, tp.ColMap]],
             chunk_meta: ChunkMeta, **kwargs) -> tp.Optional[tp.ColMap]:
        if obj is None:
            return None
        if isinstance(obj, tuple):
            return obj[1][chunk_meta.start:chunk_meta.end]
        return obj[chunk_meta.start:chunk_meta.end]


class ColLensMapper(ChunkMapper, ArgGetterMixin):
    """Class for mapping chunk metadata to per-column record lengths.

    Argument can be either a column map tuple or a column lengths array."""

    def __init__(self, arg_query: tp.AnnArgQuery) -> None:
        ChunkMapper.__init__(self)
        ArgGetterMixin.__init__(self, arg_query)

    def map(self, chunk_meta: ChunkMeta, ann_args: tp.Optional[tp.AnnArgs] = None, **kwargs) -> ChunkMeta:
        col_lens = self.get_arg(ann_args)
        if isinstance(col_lens, tuple):
            col_lens = col_lens[1]
        col_lens_slice = get_group_lens_slice(col_lens, chunk_meta)
        return ChunkMeta(
            uuid=str(uuid.uuid4()),
            idx=chunk_meta.idx,
            start=col_lens_slice.start,
            end=col_lens_slice.stop,
            indices=None
        )


col_lens_mapper = ColLensMapper(r'(col_lens|col_map)')
"""Default instance of `ColLensMapper`."""


class ColMapSlicer(ChunkSlicer):
    """Class for slicing multiple elements from a column map based on the chunk range."""

    def take(self, obj: tp.Optional[tp.ColMap], chunk_meta: ChunkMeta, **kwargs) -> tp.Optional[tp.ColMap]:
        if obj is None:
            return None
        col_idxs, col_lens = obj
        col_lens = col_lens[chunk_meta.start:chunk_meta.end]
        return np.arange(np.sum(col_lens)), col_lens


class ColIdxsMapper(ChunkMapper, ArgGetterMixin):
    """Class for mapping chunk metadata to per-column record indices.

    Argument must be a column map tuple."""

    def __init__(self, arg_query: tp.AnnArgQuery) -> None:
        ChunkMapper.__init__(self)
        ArgGetterMixin.__init__(self, arg_query)

    def map(self, chunk_meta: ChunkMeta, ann_args: tp.Optional[tp.AnnArgs] = None, **kwargs) -> ChunkMeta:
        col_map = self.get_arg(ann_args)
        col_idxs, col_lens = col_map
        col_lens_slice = get_group_lens_slice(col_lens, chunk_meta)
        return ChunkMeta(
            uuid=str(uuid.uuid4()),
            idx=chunk_meta.idx,
            start=None,
            end=None,
            indices=col_idxs[col_lens_slice]
        )


col_idxs_mapper = ColIdxsMapper('col_map')
"""Default instance of `ColIdxsMapper`."""


def merge_records(results: tp.List[tp.RecordArray], chunk_meta: ChunkMeta) -> tp.RecordArray:
    """Merge chunks of record arrays."""
    for _chunk_meta in chunk_meta:
        results[_chunk_meta.idx]['col'] += _chunk_meta.start
    return np.concatenate(results)
