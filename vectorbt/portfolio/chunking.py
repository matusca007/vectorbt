# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""Extensions to `vectorbt.utils.chunking` for chunking portfolio and its attributes."""

from vectorbt.utils.config import Config
from vectorbt.utils.chunking import ArraySlicer

close_config = Config(
    dict(arg_take_spec={'close': ArraySlicer(1)})
)
"""Config for slicing close along the second axis (columns)."""
