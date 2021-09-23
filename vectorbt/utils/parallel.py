# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""Utilities for parallelization."""

from numba.core.registry import CPUDispatcher

from vectorbt import _typing as tp
from vectorbt.utils.config import merge_dicts


def ray_apply(n: int,
              apply_func: tp.Callable, *args,
              ray_force_init: tp.Optional[bool] = None,
              ray_clear_store: tp.Optional[bool] = None,
              ray_shutdown: tp.Optional[bool] = None,
              ray_init_kwargs: tp.KwargsLike = None,
              ray_remote_kwargs: tp.KwargsLike = None,
              **kwargs) -> tp.List[tp.Any]:
    """Run `apply_func` in a distributed manner using Ray.

    Set `ray_force_init` to True to terminate the Ray runtime and initialize a new one.
    `ray_remote_kwargs` will be passed to `ray.remote` and `ray_init_kwargs` to `ray.init`.
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

    from vectorbt._settings import settings
    parallel_ray_cfg = settings['parallel']['ray']

    if ray_force_init is None:
        ray_force_init = parallel_ray_cfg['force_init']
    if ray_clear_store is None:
        ray_clear_store = parallel_ray_cfg['clear_store']
    if ray_shutdown is None:
        ray_shutdown = parallel_ray_cfg['shutdown']
    ray_init_kwargs = merge_dicts(ray_init_kwargs, parallel_ray_cfg['init_kwargs'])
    ray_remote_kwargs = merge_dicts(ray_remote_kwargs, parallel_ray_cfg['remote_kwargs'])

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
    if len(ray_remote_kwargs) > 0:
        apply_func_remote = ray.remote(**ray_remote_kwargs)(_apply_func)
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
