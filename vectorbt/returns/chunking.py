# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""Extensions to `vectorbt.utils.chunking`."""

from vectorbt.utils.config import Config
from vectorbt.utils.chunking import (
    ArraySizer,
    ArraySlicer,
    ArgsTaker
)

returns_config = Config(
    dict(
        size=ArraySizer('(rets|adj_rets|benchmark_rets)', 1),
        arg_take_spec={'(rets|adj_rets|benchmark_rets)': ArraySlicer(1)}
    )
)
"""Config for slicing returns along the second axis (columns)."""


args_rets_benchmark_rets_config = Config(
    dict(
        arg_take_spec={'args': ArgsTaker(ArraySlicer(1), ArraySlicer(1))}
    )
)
"""Config for slicing returns and benchmark returns along the second axis (columns) in variable arguments."""
