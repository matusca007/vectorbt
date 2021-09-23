# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""Utilities for parallelization."""

from numba.core.registry import CPUDispatcher

from vectorbt import _typing as tp


def ray_apply(n: int,
              apply_func: tp.Callable, *args,
              ray_force_init: bool = False,
              ray_func_kwargs: tp.KwargsLike = None,
              ray_init_kwargs: tp.KwargsLike = None,
              ray_clear_store: bool = True,
              ray_shutdown: bool = False,
              **kwargs) -> tp.List[tp.Any]:
    """Run `apply_func` in a distributed manner using Ray.

    Set `ray_force_init` to True to terminate the Ray runtime and initialize a new one.
    `ray_func_kwargs` will be passed to `ray.remote` and `ray_init_kwargs` to `ray.init`.
    Set `ray_shutdown` to True to terminate the Ray runtime upon the job end.

    !!! note
        Ray spawns multiple processes as opposed to threads, so any argument and keyword argument must first
        be put into an object store to be shared. Make sure that the computation with `apply_func` takes
        a considerable amount of time compared to this copying operation, otherwise there will be
        a little to no speedup.

    !!! warning
        Passing callables and other objects with type annotations may load vectorbt and other packages
        that these annotations depend on, causing memory pollution and `RayOutOfMemoryError`.
    """
    import ray

    if ray_init_kwargs is None:
        ray_init_kwargs = {}
    if ray_func_kwargs is None:
        ray_func_kwargs = {}
    if ray_force_init:
        if ray.is_initialized():
            ray.shutdown()
    if not ray.is_initialized():
        ray.init(**ray_init_kwargs)
    if isinstance(apply_func, CPUDispatcher):
        # Numba-wrapped function is not recognized as a function
        _apply_func = lambda *_args, **_kwargs: apply_func(*_args, **_kwargs)
    else:
        _apply_func = apply_func
    if len(ray_func_kwargs) > 0:
        apply_func_remote = ray.remote(**ray_func_kwargs)(_apply_func)
    else:
        apply_func_remote = ray.remote(_apply_func)
    # args and kwargs don't change -> put to object store
    arg_refs = ()
    for v in args:
        arg_refs += (ray.put(v),)
    kwarg_refs = {}
    for k, v in kwargs.items():
        kwarg_refs[k] = ray.put(v)
    result_refs = [apply_func_remote.remote(i, *arg_refs, **kwarg_refs) for i in range(n)]
    try:
        results = ray.get(result_refs)
    finally:
        if ray_clear_store:
            # clear object store
            del arg_refs
            del kwarg_refs
            del result_refs
        if ray_shutdown:
            ray.shutdown()
    return results
