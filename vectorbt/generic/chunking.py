# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""Extensions to `vectorbt.utils.chunking`."""

from vectorbt.utils.config import Config
from vectorbt.utils.chunking import (
    ArgSizer,
    ArraySizer,
    ShapeSizer,
    CountAdapter,
    ShapeSlicer,
    ArraySlicer
)
from vectorbt.base.chunking import GroupLensMapper

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

arr_group_lens_config = Config(
    dict(arg_take_spec={'arr': ArraySlicer(1, mapper=GroupLensMapper('group_lens'))})
)
"""Config for slicing an array based on group lengths."""
