# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""Classes and specifications for parallelization of generic functions."""

import numpy as np
import pandas as pd

from vectorbt import _typing as tp
from vectorbt.utils.chunking import ChunkMeta, ChunkTaker, ChunkSelector, ChunkSlicer


class FlexChunkTaker(ChunkTaker):
    """Class for flexible taking one or more elements.

    Accepts `flex_2d`."""

    def __init__(self, flex_2d: tp.Optional[bool] = None) -> None:
        self._flex_2d = flex_2d

    @property
    def flex_2d(self) -> tp.Optional[bool]:
        """See `vectorbt.base.indexing.flex_select_auto_nb`."""
        return self._flex_2d


class ColSelector(FlexChunkTaker, ChunkSelector):
    """Class for flexible selecting one column based on the chunk index."""

    def take(self, obj: tp.AnyArray, chunk_meta: ChunkMeta, **kwargs) -> tp.ArrayLike:
        if self.flex_2d is not None:
            if isinstance(obj, pd.Series):
                return obj
        if isinstance(obj, pd.DataFrame):
            return obj.iloc[:, chunk_meta.idx]
        if isinstance(obj, np.ndarray):
            if self.flex_2d is not None:
                if obj.ndim == 0:
                    return obj
                if obj.ndim == 1:
                    if self.flex_2d:
                        return obj[chunk_meta.idx]
                    return obj
            if obj.ndim == 2:
                return obj[:, chunk_meta.idx]
        raise ValueError(f"ColSelector accepts Series, DataFrame, or NumPy array, not {type(obj)}")


class ColSlicer(FlexChunkTaker, ChunkSlicer):
    """Class for flexible slicing multiple columns based on the chunk range."""

    def take(self, obj: tp.AnyArray, chunk_meta: ChunkMeta, **kwargs) -> tp.AnyArray:
        if self.flex_2d is not None:
            if isinstance(obj, pd.Series):
                return obj
        if isinstance(obj, pd.DataFrame):
            return obj.iloc[:, chunk_meta.range_start:chunk_meta.range_end]
        if isinstance(obj, np.ndarray):
            if self.flex_2d is not None:
                if obj.ndim == 0:
                    return obj
                if obj.ndim == 1:
                    if self.flex_2d:
                        return obj[chunk_meta.range_start:chunk_meta.range_end]
                    return obj
            if obj.ndim == 2:
                return obj[:, chunk_meta.range_start:chunk_meta.range_end]
        raise ValueError(f"ColSlicer accepts Series, DataFrame, or NumPy array, not {type(obj)}")
