# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""Classes and specifications for parallelization of generic functions."""

import numpy as np
import pandas as pd

from vectorbt import _typing as tp
from vectorbt.utils.parallel import ChunkTaker, chunk_metaT, ChunkIndexSelector, RangeSlicer


class FlexChunkTaker(ChunkTaker):
    """Class for flexible taking one or more elements.

    Accepts `flex_2d`."""

    def __init__(self, flex_2d: tp.Optional[bool] = None) -> None:
        self._flex_2d = flex_2d

    @property
    def flex_2d(self) -> tp.Optional[bool]:
        """See `vectorbt.base.reshape_fns.flex_select_auto_nb`."""
        return self._flex_2d


class ColSelector(FlexChunkTaker, ChunkIndexSelector):
    """Class for flexible selecting one column based on the chunk index."""

    def take(self, obj: tp.AnyArray, chunk_meta: chunk_metaT, **kwargs) -> tp.ArrayLike:
        if self.flex_2d is not None:
            if isinstance(obj, pd.Series):
                return obj
        if isinstance(obj, pd.DataFrame):
            return obj.iloc[:, chunk_meta[0]]
        if isinstance(obj, np.ndarray):
            if self.flex_2d is not None:
                if obj.ndim == 0:
                    return obj
                if obj.ndim == 1:
                    if self.flex_2d:
                        return obj[chunk_meta[0]]
                    return obj
            if obj.ndim == 2:
                return obj[:, chunk_meta[0]]
        raise ValueError(f"ColSelector accepts Series, DataFrame, or NumPy array, not {type(obj)}")


class ColSlicer(FlexChunkTaker, RangeSlicer):
    """Class for flexible slicing multiple columns based on the index range."""

    def take(self, obj: tp.AnyArray, chunk_meta: chunk_metaT, **kwargs) -> tp.AnyArray:
        if self.flex_2d is not None:
            if isinstance(obj, pd.Series):
                return obj
        if isinstance(obj, pd.DataFrame):
            return obj.iloc[:, chunk_meta[1]:chunk_meta[2]]
        if isinstance(obj, np.ndarray):
            if self.flex_2d is not None:
                if obj.ndim == 0:
                    return obj
                if obj.ndim == 1:
                    if self.flex_2d:
                        return obj[chunk_meta[1]:chunk_meta[2]]
                    return obj
            if obj.ndim == 2:
                return obj[:, chunk_meta[1]:chunk_meta[2]]
        raise ValueError(f"ColSlicer accepts Series, DataFrame, or NumPy array, not {type(obj)}")
