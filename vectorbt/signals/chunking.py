# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""Extensions to `vectorbt.utils.chunking`."""

from vectorbt.utils.config import Config
from vectorbt.utils.chunking import ArraySizer, ArraySlicer
from vectorbt.base.chunking import GroupLensMapper

mask_config = Config(
    dict(
        size=ArraySizer('(entries|exits|.*mask)', 1),
        arg_take_spec={'(entries|exits|.*mask)': ArraySlicer(1)}
    )
)
"""Config for slicing a mask along the second axis (columns)."""

mask_group_lens_config = Config(
    dict(arg_take_spec={'(entries|exits|.*mask)': ArraySlicer(1, mapper=GroupLensMapper('group_lens'))})
)
"""Config for slicing a mask based on group lengths."""
