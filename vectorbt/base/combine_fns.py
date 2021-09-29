# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""Functions for combining arrays.

Combine functions combine two or more NumPy arrays using a custom function. The emphasis here is
done upon stacking the results into one NumPy array - since vectorbt is all about brute-forcing
large spaces of hyperparameters, concatenating the results of each hyperparameter combination into
a single DataFrame is important. All functions are available in both Python and Numba-compiled form."""

import numpy as np
import warnings
from numba.typed import List

from vectorbt import _typing as tp
from vectorbt.nb_registry import register_jit
from vectorbt.utils import checks
from vectorbt.utils.execution import execute
from vectorbt.base import reshape_fns


@register_jit
def apply_and_concat_none_nb(ntimes: int, apply_func_nb: tp.Callable, *args) -> None:
    """Run `apply_func_nb` that returns nothing. Meant for in-place outputs."""
    for i in range(ntimes):
        apply_func_nb(i, *args)


@register_jit
def to_2d_one_nb(a: tp.Array) -> tp.Array2d:
    """Expand the dimensions of the array along the axis 1."""
    if a.ndim > 1:
        return a
    return np.expand_dims(a, axis=1)


@register_jit
def apply_and_concat_one_nb(ntimes: int, apply_func_nb: tp.Callable, *args) -> tp.Array2d:
    """Run `apply_func_nb` that returns one array."""
    output_0 = to_2d_one_nb(apply_func_nb(0, *args))
    output = np.empty((output_0.shape[0], ntimes * output_0.shape[1]), dtype=output_0.dtype)
    for i in range(ntimes):
        if i == 0:
            outputs_i = output_0
        else:
            outputs_i = to_2d_one_nb(apply_func_nb(i, *args))
        output[:, i * outputs_i.shape[1]:(i + 1) * outputs_i.shape[1]] = outputs_i
    return output


@register_jit
def to_2d_multiple_nb(a: tp.Iterable[tp.Array]) -> tp.List[tp.Array2d]:
    """Expand the dimensions of each array in `a` along axis 1."""
    lst = list()
    for _a in a:
        lst.append(to_2d_one_nb(_a))
    return lst


@register_jit
def apply_and_concat_multiple_nb(ntimes: int, apply_func_nb: tp.Callable, *args) -> tp.List[tp.Array2d]:
    """Run `apply_func_nb` that returns multiple arrays."""
    outputs = list()
    outputs_0 = to_2d_multiple_nb(apply_func_nb(0, *args))
    for j in range(len(outputs_0)):
        outputs.append(np.empty((outputs_0[j].shape[0], ntimes * outputs_0[j].shape[1]), dtype=outputs_0[j].dtype))
    for i in range(ntimes):
        if i == 0:
            outputs_i = outputs_0
        else:
            outputs_i = to_2d_multiple_nb(apply_func_nb(i, *args))
        for j in range(len(outputs_i)):
            outputs[j][:, i * outputs_i[j].shape[1]:(i + 1) * outputs_i[j].shape[1]] = outputs_i[j]
    return outputs


def apply_and_concat(ntimes: int,
                     apply_func: tp.Callable, *args,
                     n_outputs: tp.Optional[int] = None,
                     numba_loop: bool = False,
                     execute_kwargs: tp.KwargsLike = None,
                     silence_warnings: bool = False,
                     **kwargs) -> tp.Union[None, tp.Array2d, tp.List[tp.Array2d]]:
    """Run `apply_func` function `ntimes` times and concatenate the results depending upon how
    many array-like objects it generates.

    `apply_func` must accept arguments `i`, `*args`, and `**kwargs`.

    Executes the function using `vectorbt.utils.execution.execute`.

    Set `numba_loop` to True to use the Numba-compiled version.

    !!! note
        `n_outputs` must be set when `numba_loop` is True.

        Numba doesn't support variable keyword arguments."""
    if execute_kwargs is None:
        execute_kwargs = {}
    if numba_loop:
        if len(kwargs) > 0 and not silence_warnings:
            warnings.warn("Numba doesn't support **kwargs", stacklevel=2)
        if len(execute_kwargs) > 0 and not silence_warnings:
            warnings.warn("Numba doesn't support execute_kwargs", stacklevel=2)

    if numba_loop:
        if n_outputs is None:
            raise ValueError("Iteration with Numba requires n_outputs")
        if n_outputs == 0:
            return apply_and_concat_none_nb(ntimes, apply_func, *args)
        if n_outputs == 1:
            return apply_and_concat_one_nb(ntimes, apply_func, *args)
        return apply_and_concat_multiple_nb(ntimes, apply_func, *args)

    funcs_args = [(apply_func, (i, *args), kwargs) for i in range(ntimes)]
    out = execute(funcs_args, **execute_kwargs)
    if n_outputs is None:
        if out[0] is None:
            n_outputs = 0
        elif isinstance(out[0], (tuple, list, List)):
            n_outputs = len(out[0])
        else:
            n_outputs = 1
    if n_outputs == 0:
        return None
    if n_outputs == 1:
        return np.column_stack(tuple(map(reshape_fns.to_2d, out)))
    return list(map(np.column_stack, zip(*tuple(map(lambda x: tuple(map(reshape_fns.to_2d, x)), out)))))


@register_jit
def select_and_combine_nb(i: int,
                          obj: tp.Any,
                          others: tp.Sequence,
                          combine_func_nb: tp.Callable, *args) -> tp.AnyArray:
    """Numba-compiled version of `select_and_combine`."""
    return combine_func_nb(obj, others[i], *args)


@register_jit
def combine_and_concat_nb(obj: tp.Any,
                          others: tp.Sequence,
                          combine_func_nb: tp.Callable, *args) -> tp.Array2d:
    """Numba-compiled version of `combine_and_concat`."""
    return apply_and_concat_one_nb(len(others), select_and_combine_nb, obj, others, combine_func_nb, *args)


def select_and_combine(i: int,
                       obj: tp.Any,
                       others: tp.Sequence,
                       combine_func: tp.Callable, *args, **kwargs) -> tp.AnyArray:
    """Combine `obj` with an array at position `i` in `others` using `combine_func`."""
    return combine_func(obj, others[i], *args, **kwargs)


def combine_and_concat(obj: tp.Any,
                       others: tp.Sequence,
                       combine_func: tp.Callable, *args,
                       numba_loop: bool = False,
                       **kwargs) -> tp.Array2d:
    """Combine `obj` with each in `others` using `combine_func` and concatenate."""
    if numba_loop:
        apply_func = select_and_combine_nb
    else:
        apply_func = select_and_combine
    return apply_and_concat(
        len(others),
        apply_func, obj, others, combine_func, *args,
        n_outputs=1,
        numba_loop=numba_loop,
        **kwargs
    )


@register_jit
def combine_multiple_nb(objs: tp.Sequence, combine_func_nb: tp.Callable, *args) -> tp.Any:
    """Numba-compiled version of `combine_multiple`."""
    result = objs[0]
    for i in range(1, len(objs)):
        result = combine_func_nb(result, objs[i], *args)
    return result


def combine_multiple(objs: tp.Sequence,
                     combine_func: tp.Callable, *args,
                     numba_loop: bool = False,
                     silence_warnings: bool = False,
                     **kwargs) -> tp.Any:
    """Combine `objs` pairwise into a single object.

    Set `numba_loop` to True to use the Numba-compiled version.

    !!! note
        Numba doesn't support variable keyword arguments."""
    if numba_loop:
        if len(kwargs) > 0 and not silence_warnings:
            warnings.warn("Numba doesn't support variable keyword arguments", stacklevel=2)
        return combine_multiple_nb(objs, combine_func, *args)
    result = objs[0]
    for i in range(1, len(objs)):
        result = combine_func(result, objs[i], *args, **kwargs)
    return result
