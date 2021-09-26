# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""Utilities for parallelization."""

from numba.core.registry import CPUDispatcher
import inspect
import multiprocessing

from vectorbt import _typing as tp
from vectorbt.utils import checks
from vectorbt.utils.config import merge_dicts
from vectorbt.utils.parsing import annotate_args

try:
    from ray.remote_function import RemoteFunction as RemoteFunctionT
    from ray import ObjectRef as ObjectRefT
except ImportError:
    RemoteFunctionT = tp.Any
    ObjectRefT = tp.Any

chunk_metaT = tp.Tuple[int, int, int]
TakeSpecT = tp.Union[None, "ChunkTaker"]
MappingTakeSpecT = tp.Mapping[str, TakeSpecT]
SequenceTakeSpecT = tp.Sequence[TakeSpecT]
ContainerTakeSpecT = tp.Union[MappingTakeSpecT, SequenceTakeSpecT]
ArgTakeSpecT = MappingTakeSpecT


class ChunkTaker:
    """Base class for taking one or more elements based on the chunk index or the index range.

    Implements the abstract `ChunkTaker.take` method."""

    def take(self, obj: tp.Any, chunk_meta: chunk_metaT, **kwargs) -> tp.Any:
        """Abstract method for taking subset of data.

        Takes the argument object, the chunk meta (tuple out of the chunk index, start index,
        and end index), and other keyword arguments passed down the stack, at least the entire
        argument specification `arg_take_spec`."""
        raise NotImplementedError


class ChunkIndexSelector(ChunkTaker):
    """Class for selecting one element based on the chunk index."""

    def take(self, obj: tp.Sequence, chunk_meta: chunk_metaT, **kwargs) -> tp.Any:
        return obj[chunk_meta[0]]


class ChunkIndexSlicer(ChunkTaker):
    """Class for slicing multiple elements based on the chunk index."""

    def take(self, obj: tp.Sequence, chunk_meta: chunk_metaT, **kwargs) -> tp.Sequence:
        return obj[chunk_meta[0]:chunk_meta[0] + 1]


class RangeStartSelector(ChunkTaker):
    """Class for selecting one element based on the start of the index range."""

    def take(self, obj: tp.Sequence, chunk_meta: chunk_metaT, **kwargs) -> tp.Any:
        return obj[chunk_meta[1]]


class RangeEndSelector(ChunkTaker):
    """Class for selecting one element based on the end of the index range."""

    def take(self, obj: tp.Sequence, chunk_meta: chunk_metaT, **kwargs) -> tp.Any:
        return obj[chunk_meta[2]]


class RangeSlicer(ChunkTaker):
    """Class for slicing multiple elements based on the index range."""

    def take(self, obj: tp.Sequence, chunk_meta: chunk_metaT, **kwargs) -> tp.Sequence:
        return obj[chunk_meta[1]:chunk_meta[2]]


class ContainerTaker(ChunkTaker):
    """Class for evaluating containers with other chunk takers.

    Accepts the specification of the container."""

    def __init__(self, cont_take_spec: ContainerTakeSpecT) -> None:
        self._cont_take_spec = cont_take_spec

    @property
    def cont_take_spec(self) -> ContainerTakeSpecT:
        """Specification of the container."""
        return self._cont_take_spec

    def take(self, obj: tp.Any, chunk_meta: chunk_metaT, **kwargs) -> tp.Any:
        raise NotImplementedError


class MappingTaker(ContainerTaker):
    """Class for evaluating mappings.

    Calls `take_from_arg` on each element."""

    def take(self, obj: tp.Mapping, chunk_meta: chunk_metaT, **kwargs) -> tp.Mapping:
        new_obj = {}
        for k, v in obj.items():
            new_obj[k] = take_from_arg(v, self.cont_take_spec.get(k, None), chunk_meta, **kwargs)
        return type(obj)(new_obj)


class SequenceTaker(ContainerTaker):
    """Class for evaluating sequences.

    Calls `take_from_arg` on each element."""

    def take(self, obj: tp.Sequence, chunk_meta: chunk_metaT, **kwargs) -> tp.Sequence:
        new_obj = []
        for k, v in enumerate(obj):
            new_obj.append(take_from_arg(v, self.cont_take_spec[k], chunk_meta, **kwargs))
        if checks.is_namedtuple(obj):
            return type(obj)(*new_obj)
        return type(obj)(new_obj)


def take_from_arg(arg: tp.Any, take_spec: TakeSpecT, chunk_meta: chunk_metaT, **kwargs) -> tp.Any:
    """Take from the argument given the specification `take_spec`.

    If `take_spec` is None, returns the original object. Otherwise, must be an instance of `ChunkTaker`."""
    if take_spec is None:
        return arg
    if isinstance(take_spec, ChunkTaker):
        return take_spec.take(arg, chunk_meta, **kwargs)
    raise TypeError(f"Specification of type {type(take_spec)} is not supported")


def take_from_args(annotated_args: tp.Tuple[dict, ...],
                   arg_take_spec: ArgTakeSpecT,
                   chunk_meta: chunk_metaT) -> tp.Tuple[tp.Args, tp.Kwargs]:
    """Take from each in the annotated arguments given the specification using `take_from_arg`.

    Returns arguments and keyword arguments that can be directly passed to the function
    using `func(*args, **kwargs)`."""
    new_args = ()
    new_kwargs = dict()
    for annotated_arg in annotated_args:
        arg_name = annotated_arg['name']
        if annotated_arg['kind'] == inspect.Parameter.VAR_POSITIONAL:
            for var_arg_i, var_arg in enumerate(annotated_arg['arg']):
                new_var_arg = take_from_arg(
                    var_arg,
                    arg_take_spec[arg_name][var_arg_i],
                    chunk_meta,
                    arg_take_spec=arg_take_spec
                )
                new_args += (new_var_arg,)
        elif annotated_arg['kind'] == inspect.Parameter.VAR_KEYWORD:
            for var_kwarg_name, var_kwarg in annotated_arg['arg'].items():
                new_var_kwarg = take_from_arg(
                    var_kwarg,
                    arg_take_spec[arg_name].get(var_kwarg_name, None),
                    chunk_meta,
                    arg_take_spec=arg_take_spec
                )
                new_kwargs[var_kwarg_name] = new_var_kwarg
        elif annotated_arg['kind'] == inspect.Parameter.KEYWORD_ONLY:
            kwarg = annotated_arg.get('arg', annotated_arg['default'])
            new_kwarg = take_from_arg(
                kwarg,
                arg_take_spec.get(arg_name, None),
                chunk_meta,
                arg_take_spec=arg_take_spec
            )
            new_kwargs[arg_name] = new_kwarg
        else:
            arg = annotated_arg['arg']
            new_arg = take_from_arg(
                arg,
                arg_take_spec.get(arg_name, None),
                chunk_meta,
                arg_take_spec=arg_take_spec
            )
            new_args += (new_arg,)
    return new_args, new_kwargs


def yield_chunk_meta(size: int,
                     n_chunks: tp.Optional[int] = None,
                     chunk_len: tp.Optional[int] = None) -> tp.Generator[chunk_metaT, None, None]:
    """Yield meta of each successive chunk from a sequence with a number of elements.

    If both `n_chunks` and `chunk_len` are None (after resolving them from settings),
    sets `n_chunks` to the number of cores.

    For defaults, see `parallel` in `vectorbt._settings.settings`."""
    from vectorbt._settings import settings
    parallel_cfg = settings['parallel']

    if n_chunks is None:
        n_chunks = parallel_cfg['n_chunks']
    if chunk_len is None:
        chunk_len = parallel_cfg['chunk_len']

    if n_chunks is None and chunk_len is None:
        n_chunks = multiprocessing.cpu_count()
    if n_chunks is not None:
        if n_chunks == 0:
            raise ValueError("Chunk count cannot be zero")
        if n_chunks > size:
            raise ValueError("Chunk count cannot exceed element count")
        d, r = divmod(size, n_chunks)
        for i in range(n_chunks):
            si = (d + 1) * (i if i < r else r) + d * (0 if i < r else i - r)
            yield i, si, si + (d + 1 if i < r else d)
    if chunk_len is not None:
        if chunk_len == 0:
            raise ValueError("Chunk length cannot be zero")
        for chunk_i, i in enumerate(range(0, size, chunk_len)):
            yield chunk_i, i, min(i + chunk_len, size)


funcs_argsT = tp.Tuple[tp.Callable, tp.Args, tp.Kwargs]
funcs_args_refsT = tp.Tuple[RemoteFunctionT, tp.Tuple[ObjectRefT, ...], tp.Dict[str, ObjectRefT]]


def get_ray_refs(funcs_args: tp.Iterable[funcs_argsT],
                 reuse_refs: bool = True,
                 remote_kwargs: tp.KwargsLike = None) -> tp.List[funcs_args_refsT]:
    """Get result references by putting each argument and keyword argument into the object store
    and invoking the remote decorator on each function using Ray."""
    import ray
    from ray.remote_function import RemoteFunction
    from ray import ObjectRef

    func_id_remotes = {}
    obj_id_refs = {}
    funcs_args_refs = []
    for func, args, kwargs in funcs_args:
        # Get remote function
        if isinstance(func, RemoteFunction):
            func_remote = func
        else:
            if not reuse_refs or id(func) not in func_id_remotes:
                if isinstance(func, CPUDispatcher):
                    # Numba-wrapped function is not recognized by ray as a function
                    _func = lambda *_args, **_kwargs: func(*_args, **_kwargs)
                else:
                    _func = func
                if len(remote_kwargs) > 0:
                    func_remote = ray.remote(**remote_kwargs)(_func)
                else:
                    func_remote = ray.remote(_func)
                if reuse_refs:
                    func_id_remotes[id(func)] = func_remote
            else:
                func_remote = func_id_remotes[id(func)]

        # Get id of each arg
        arg_refs = ()
        for arg in args:
            if isinstance(arg, ObjectRef):
                arg_ref = arg
            else:
                if not reuse_refs or id(arg) not in obj_id_refs:
                    arg_ref = ray.put(arg)
                    obj_id_refs[id(arg)] = arg_ref
                else:
                    arg_ref = obj_id_refs[id(arg)]
            arg_refs += (arg_ref,)

        # Get id of each kwarg
        kwarg_refs = {}
        for kwarg_name, kwarg in kwargs.items():
            if isinstance(kwarg, ObjectRef):
                kwarg_ref = kwarg
            else:
                if not reuse_refs or id(kwarg) not in obj_id_refs:
                    kwarg_ref = ray.put(kwarg)
                    obj_id_refs[id(kwarg)] = kwarg_ref
                else:
                    kwarg_ref = obj_id_refs[id(kwarg)]
            kwarg_refs[kwarg_name] = kwarg_ref

        funcs_args_refs.append((func_remote, arg_refs, kwarg_refs))
    return funcs_args_refs


def run_using_ray(funcs_args: tp.Iterable[funcs_argsT],
                  restart: tp.Optional[bool] = None,
                  reuse_refs: tp.Optional[bool] = None,
                  del_refs: tp.Optional[bool] = None,
                  shutdown: tp.Optional[bool] = None,
                  init_kwargs: tp.KwargsLike = None,
                  remote_kwargs: tp.KwargsLike = None) -> tp.List[tp.Any]:
    """Run a sequence of functions in a distributed manner using Ray.

    Args:
        funcs_args (sequence of tuples): Sequence of tuples, each composed of the
            function to be called, and arguments and keyword arguments passed to this function.
        restart (bool): Whether to terminate the Ray runtime and initialize a new one.
        reuse_refs (bool): Whether to re-use function and object references, such that each
            unique object will be copied only once.
        del_refs (bool): Whether to explicitly delete the result object references.
        shutdown (bool): Whether to True to terminate the Ray runtime upon the job end.
        init_kwargs (dict): Keyword arguments passed to `ray.init`.
        remote_kwargs (dict): Keyword arguments passed to `ray.remote`.

    For defaults, see `parallel.ray` in `vectorbt._settings.settings`.

    !!! note
        Ray spawns multiple processes as opposed to threads, so any argument and keyword argument must first
        be put into an object store to be shared. Make sure that the computation with `func` takes
        a considerable amount of time compared to this copying operation, otherwise there will be
        a little to no speedup.

    !!! warning
        Passing callables and other objects with type annotations may load vectorbt and other packages
        that these annotations depend on, causing memory pollution and `RayOutOfMemoryError`.
    """
    import ray

    from vectorbt._settings import settings
    parallel_ray_cfg = settings['parallel']['ray']

    if restart is None:
        restart = parallel_ray_cfg['restart']
    if reuse_refs is None:
        reuse_refs = parallel_ray_cfg['reuse_refs']
    if del_refs is None:
        del_refs = parallel_ray_cfg['del_refs']
    if shutdown is None:
        shutdown = parallel_ray_cfg['shutdown']
    init_kwargs = merge_dicts(init_kwargs, parallel_ray_cfg['init_kwargs'])
    remote_kwargs = merge_dicts(remote_kwargs, parallel_ray_cfg['remote_kwargs'])

    if restart:
        if ray.is_initialized():
            ray.shutdown()
    if not ray.is_initialized():
        ray.init(**init_kwargs)
    funcs_args_refs = get_ray_refs(funcs_args, reuse_refs=reuse_refs, remote_kwargs=remote_kwargs)
    result_refs = []
    for func_remote, arg_refs, kwarg_refs in funcs_args_refs:
        result_refs.append(func_remote.remote(*arg_refs, **kwarg_refs))
    try:
        results = ray.get(result_refs)
    finally:
        if del_refs:
            # clear object store
            del result_refs
        if shutdown:
            ray.shutdown()
    return results


def run_ntimes_using_ray(n: int, func: tp.Callable, *args,
                         ray_kwargs: tp.KwargsLike = None, **kwargs) -> tp.List[tp.Any]:
    """Run a function a number of times using `run_using_ray`.

    `func` must accept the (incrementing) index of the run, `*args`, and `**kwargs`.
    Use the run index, for example, to select a portion of the data to analyze."""
    if ray_kwargs is None:
        ray_kwargs = {}

    return run_using_ray([(func, (i, *args), kwargs) for i in range(n)], **ray_kwargs)


class SizeFromArg:
    """Class that tells `resolve_size` to derive the size from an argument.

    `arg` can be an integer indicating the position, or the name of the argument.

    If `axis` is None, uses the argument as the size. Otherwise, derives the size from
    the argument's axis. Supported are both arrays and shape tuples."""

    def __init__(self, arg: tp.Union[str, int], axis: tp.Optional[int] = None) -> None:
        self._arg = arg
        self._axis = axis

    @property
    def arg(self) -> tp.Union[str, int]:
        """Argument to derive the size from."""
        return self._arg

    @property
    def axis(self) -> tp.Optional[int]:
        """Axis of the argument to derive the size from."""
        return self._axis


def resolve_size(annotated_args: tp.Tuple[dict, ...],
                 size: tp.Optional[int, SizeFromArg, tp.Callable]) -> int:
    """Resolve the size to chunk.

    `size` can be an integer, an instance of `SizeFromArg`, or a callable that takes annotated arguments
    and returns an integer."""
    if isinstance(size, SizeFromArg):
        if isinstance(size.arg, int):
            annotated_arg = annotated_args[size.arg]
        else:
            annotated_arg = list(map(lambda x: x['name'], annotated_args)).index(size.arg)
        arg = annotated_arg.get('arg', annotated_arg.get('default'), None)
        if size.axis is None:
            return arg
        if not isinstance(arg, tuple):
            arg = arg.shape
        if size.axis <= len(arg) - 1:
            return arg[size.axis]
        return 0
    if callable(size):
        return size(annotated_args)
    raise TypeError(f"Size type {type(size)} is not supported")


def parallelize(func: tp.Callable,
                arg_take_spec: ArgTakeSpecT,
                size: tp.Optional[int, SizeFromArg, tp.Callable],
                chunk_meta_iter: tp.Optional[tp.Iterable[chunk_metaT]] = None,
                n_chunks: tp.Optional[int] = None,
                chunk_len: tp.Optional[int] = None,
                merge_func: tp.Optional[tp.Callable] = None,
                backend: tp.Optional[tp.Union[str, tp.Callable]] = None,
                **backend_kwargs) -> tp.Callable:
    """Parallelize the function `func` given the specification `arg_take_spec` and the size to chunk `size`.

    Returns a wrapper that accepts the same arguments as `func`, splits them into chunks,
    runs those chunks in parallel using `backend`, and optionally, merges the results using `merge_func`.

    `size` is resolved using `resolve_size`.

    Chunk meta is generated either by iterating `chunk_meta_iter` or using `yield_chunk_meta`.
    
    `backend_kwargs` are passed to the backend function.

    For defaults, see `parallel` in `vectorbt._settings.settings`."""
    from vectorbt._settings import settings
    parallel_cfg = settings['parallel']

    if backend is None:
        backend = parallel_cfg['backend']

    def wrapper(*args, _chunk_meta_iter=chunk_meta_iter, **kwargs):
        annotated_args = annotate_args(func, *args, **kwargs)
        resolved_size = resolve_size(annotated_args, size)
        if _chunk_meta_iter is None:
            _chunk_meta_iter = yield_chunk_meta(resolved_size, n_chunks=n_chunks, chunk_len=chunk_len)
        funcs_args = []
        for chunk_meta in _chunk_meta_iter:
            chunk_args, chunk_kwargs = take_from_args(annotated_args, arg_take_spec, chunk_meta)
            funcs_args.append((func, chunk_args, chunk_kwargs))
        if isinstance(backend, str):
            if backend.lower() == 'ray':
                results = run_using_ray(funcs_args, **backend_kwargs)
            else:
                raise ValueError(f"Backend '{type(backend)}' is not supported")
        elif callable(backend):
            results = backend(funcs_args, **backend_kwargs)
        else:
            raise TypeError(f"Backend type {type(backend)} is not supported")
        if merge_func is not None:
            return merge_func(results)
        return results

    return wrapper
