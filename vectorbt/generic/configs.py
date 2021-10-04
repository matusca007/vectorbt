# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""Configs."""

import numpy as np

from vectorbt.utils.config import Config
from vectorbt.utils.chunking import (
    ArgSizer,
    ArraySizer,
    ShapeSizer,
    SizeAdapter,
    ShapeSlicer,
    ArraySlicer
)
from vectorbt.base.grouping import GroupLensMapper

chunked_size_config = Config(
    dict(
        size=ArgSizer(0),
        arg_take_spec={0: SizeAdapter()}
    )
)
"""Config for adapting a constant holding the size."""

chunked_shape_ax1_config = Config(
    dict(
        size=ShapeSizer(0, 1),
        arg_take_spec={0: ShapeSlicer(1)}
    )
)
"""Config for slicing a shape along the second axis (columns)."""

chunked_shape_ax0_config = Config(
    dict(
        size=ShapeSizer(0, 0),
        arg_take_spec={0: ShapeSlicer(0)}
    )
)
"""Config for slicing a shape along the first axis (rows)."""

chunked_arr_ax1_config = Config(
    dict(
        size=ArraySizer(0, 1),
        arg_take_spec={0: ArraySlicer(1)}
    )
)
"""Config for slicing an array along the second axis (columns)."""

chunked_arr_ax0_config = Config(
    dict(
        size=ArraySizer(0, 0),
        arg_take_spec={0: ArraySlicer(0)}
    )
)
"""Config for slicing an array along the first axis (rows)."""

chunked_none_group_lens_config = Config(
    dict(
        size=ArraySizer(1, 0),
        arg_take_spec={1: ArraySlicer(0)}
    )
)
"""Config for slicing group lengths (second argument)."""

chunked_arr_group_lens_config = Config(
    dict(
        size=ArraySizer(1, 0),
        arg_take_spec={0: ArraySlicer(1, mapper=GroupLensMapper(1)), 1: ArraySlicer(0)}
    )
)
"""Config for slicing an array (first argument) based on group lengths (second argument)."""

chunked_arr_none_group_lens_config = Config(
    dict(
        size=ArraySizer(2, 0),
        arg_take_spec={0: ArraySlicer(1, mapper=GroupLensMapper(2)), 2: ArraySlicer(0)}
    )
)
"""Config for slicing an array (first argument) based on group lengths (third argument)."""

chunked_hstack_config = Config(
    dict(
        merge_func=np.hstack
    )
)
"""Config for merging arrays through horizontal stacking (column-wise)."""

chunked_vstack_config = Config(
    dict(
        merge_func=np.vstack
    )
)
"""Config for merging arrays through vertical stacking (row-wise)."""

chunked_concat_config = Config(
    dict(
        merge_func=np.concatenate
    )
)
"""Config for merging arrays through concatenating."""
