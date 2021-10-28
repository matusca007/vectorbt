# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""Numba-compiled functions.

Provides an arsenal of Numba-compiled functions that are used to generate data.
These only accept NumPy arrays and other Numba-compatible types."""

import numpy as np

from vectorbt import _typing as tp
from vectorbt.nb_registry import register_jit


@register_jit(cache=True)
def generate_random_data_nb(shape: tp.Shape, start_value: float, mean: float, std: float) -> tp.Array2d:
    """Generate data using returns drawn from normal (Gaussian) distribution."""
    out = np.empty(shape, dtype=np.float_)
    for col in range(shape[1]):
        for i in range(shape[0]):
            if i == 0:
                prev_value = start_value
            else:
                prev_value = out[i - 1, col]
            out[i, col] = prev_value * (1 + np.random.normal(mean, std))
    return out
