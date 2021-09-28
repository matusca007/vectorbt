# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""Utilities for parallelization."""

from numba.core.registry import CPUDispatcher
import inspect
import multiprocessing
from functools import wraps

from vectorbt import _typing as tp
from vectorbt.utils import checks
from vectorbt.utils.config import merge_dicts, Config
from vectorbt.utils.parsing import annotate_args, ann_argsT, get_from_ann_args, get_func_arg_names
from vectorbt.utils.template import deep_substitute, Rep

try:
    from ray.remote_function import RemoteFunction as RemoteFunctionT
    from ray import ObjectRef as ObjectRefT
except ImportError:
    RemoteFunctionT = tp.Any
    ObjectRefT = tp.Any

__pdoc__ = {}


# ############# Chunk generation ############# #

class ChunkMeta(tp.NamedTuple):
    idx: int
    range_start: tp.Optional[int]
    range_end: tp.Optional[int]


__pdoc__['ChunkMeta'] = "A named tuple representing a chunk metadata."
__pdoc__['ChunkMeta.chunk_idx'] = "Chunk index."
__pdoc__['ChunkMeta.from_idx'] = "Start of the chunk range (including)."
__pdoc__['ChunkMeta.to_idx'] = "End of the chunk range (excluding)."


def yield_chunk_meta(n_chunks: tp.Optional[int] = None,
                     size: tp.Optional[int] = None,
                     chunk_len: tp.Optional[int] = None) -> tp.Generator[ChunkMeta, None, None]:
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
        if size is not None:
            if n_chunks > size:
                raise ValueError("Chunk count cannot exceed element count")
            d, r = divmod(size, n_chunks)
            for i in range(n_chunks):
                si = (d + 1) * (i if i < r else r) + d * (0 if i < r else i - r)
                yield ChunkMeta(
                    idx=i,
                    range_start=si,
                    range_end=si + (d + 1 if i < r else d)
                )
        else:
            for i in range(n_chunks):
                yield ChunkMeta(
                    idx=i,
                    range_start=None,
                    range_end=None
                )
    if chunk_len is not None:
        checks.assert_not_none(size)
        if chunk_len == 0:
            raise ValueError("Chunk length cannot be zero")
        for chunk_i, i in enumerate(range(0, size, chunk_len)):
            yield ChunkMeta(
                idx=chunk_i,
                range_start=i,
                range_end=min(i + chunk_len, size)
            )


class ChunkMetaGenerator:
    """Abstract class for generating chunk metadata from annotated arguments."""

    def get_chunk_meta(self, ann_args: ann_argsT) -> tp.Iterable[ChunkMeta]:
        """Get chunk metadata."""
        raise NotImplementedError


class ArgGetter:
    """Abstract class for getting an argument from annotated arguments."""

    def __init__(self, arg: tp.Union[str, int]) -> None:
        self._arg = arg

    @property
    def arg(self) -> tp.Union[str, int]:
        """Argument position or name to derive the size from."""
        return self._arg

    def get_arg(self, ann_args: ann_argsT) -> tp.Any:
        """Get argument using `vectorbt.utils.parsing.get_from_ann_args`."""
        if isinstance(self.arg, int):
            return get_from_ann_args(ann_args, i=self.arg)
        return get_from_ann_args(ann_args, name=self.arg)


class ArgChunkMeta(ChunkMetaGenerator, ArgGetter):
    """Class for generating chunk metadata from an argument."""

    def get_chunk_meta(self, ann_args: ann_argsT) -> tp.Iterable[ChunkMeta]:
        return self.get_arg(ann_args)


class LenChunkMeta(ArgChunkMeta):
    """Class for generating chunk metadata from a sequence of chunk lengths."""

    def get_chunk_meta(self, ann_args: ann_argsT) -> tp.Iterable[ChunkMeta]:
        arg = self.get_arg(ann_args)
        range_start = 0
        range_end = 0
        for i, chunk_len in enumerate(arg):
            range_end += chunk_len
            yield ChunkMeta(idx=i, range_start=range_start, range_end=range_end)
            range_start = range_end


# ############# Chunk sizing ############# #


class Sizer:
    """Abstract class for getting the size from annotated arguments."""

    def get_size(self, ann_args: ann_argsT) -> int:
        """Get the size given the annotated arguments."""
        raise NotImplementedError


class ArgSizer(Sizer, ArgGetter):
    """Class for getting the size from an argument."""

    def get_size(self, ann_args: ann_argsT) -> int:
        return self.get_arg(ann_args)


class LenSizer(ArgSizer):
    """Class for getting the size from the length of an argument."""

    def get_size(self, ann_args: ann_argsT) -> int:
        return len(self.get_arg(ann_args))


class AxisSizer(ArgSizer):
    """Abstract class for getting the size from the length of an axis in an argument."""

    def __init__(self, arg: tp.Union[str, int], axis: int) -> None:
        ArgSizer.__init__(self, arg)

        if axis < 0:
            raise ValueError("Axis cannot be negative")
        self._axis = axis

    @property
    def axis(self) -> int:
        """Axis of the argument to derive the size from."""
        return self._axis


class ShapeSizer(AxisSizer):
    """Class for getting the size from the length of an axis in a shape."""

    def get_size(self, ann_args: ann_argsT) -> int:
        arg = self.get_arg(ann_args)
        if self.axis <= len(arg) - 1:
            return arg[self.axis]
        return 0


class ArraySizer(AxisSizer):
    """Class for getting the size from the length of an axis in an array."""

    def get_size(self, ann_args: ann_argsT) -> int:
        arg = self.get_arg(ann_args)
        if self.axis <= len(arg.shape) - 1:
            return arg.shape[self.axis]
        return 0


# ############# Chunk taking ############# #

TakeSpecT = tp.Union[None, "ChunkTaker"]
MappingTakeSpecT = tp.Mapping[str, TakeSpecT]
SequenceTakeSpecT = tp.Sequence[TakeSpecT]
ContainerTakeSpecT = tp.Union[MappingTakeSpecT, SequenceTakeSpecT]
ArgTakeSpecT = MappingTakeSpecT


class ChunkTaker:
    """Base class for taking one or more elements based on the chunk index or range.

    Implements the abstract `ChunkTaker.take` method."""

    def take(self, obj: tp.Any, chunk_meta: ChunkMeta, **kwargs) -> tp.Any:
        """Abstract method for taking subset of data.

        Takes the argument object, the chunk meta (tuple out of the index, start index,
        and end index of the chunk), and other keyword arguments passed down the stack,
        at least the entire argument specification `arg_take_spec`."""
        raise NotImplementedError


class ChunkSelector(ChunkTaker):
    """Class for selecting one element based on the chunk index."""

    def take(self, obj: tp.Sequence, chunk_meta: ChunkMeta, **kwargs) -> tp.Any:
        return obj[chunk_meta.idx]


class ChunkSlicer(ChunkTaker):
    """Class for slicing multiple elements based on the chunk range."""

    def take(self, obj: tp.Sequence, chunk_meta: ChunkMeta, **kwargs) -> tp.Sequence:
        return obj[chunk_meta.range_start:chunk_meta.range_end]


class ContainerTaker(ChunkTaker):
    """Class for taking from a container with other chunk takers.

    Accepts the specification of the container."""

    def __init__(self, cont_take_spec: ContainerTakeSpecT) -> None:
        self._cont_take_spec = cont_take_spec

    @property
    def cont_take_spec(self) -> ContainerTakeSpecT:
        """Specification of the container."""
        return self._cont_take_spec

    def take(self, obj: tp.Any, chunk_meta: ChunkMeta, **kwargs) -> tp.Any:
        raise NotImplementedError


class SequenceTaker(ContainerTaker):
    """Class for taking from a sequence container.

    Calls `take_from_arg` on each element."""

    def take(self, obj: tp.Sequence, chunk_meta: ChunkMeta, **kwargs) -> tp.Sequence:
        new_obj = []
        for k, v in enumerate(obj):
            new_obj.append(take_from_arg(v, self.cont_take_spec[k], chunk_meta, **kwargs))
        if checks.is_namedtuple(obj):
            return type(obj)(*new_obj)
        return type(obj)(new_obj)


class MappingTaker(ContainerTaker):
    """Class for taking from a mapping container.

    Calls `take_from_arg` on each element."""

    def take(self, obj: tp.Mapping, chunk_meta: ChunkMeta, **kwargs) -> tp.Mapping:
        new_obj = {}
        for k, v in obj.items():
            new_obj[k] = take_from_arg(v, self.cont_take_spec.get(k, None), chunk_meta, **kwargs)
        return type(obj)(new_obj)


class ArgsTaker(SequenceTaker):
    """Class for taking from a variable arguments container."""

    def __init__(self, *args) -> None:
        SequenceTaker.__init__(self, args)


class KwargsTaker(MappingTaker):
    """Class for taking from a variable keyword arguments container."""

    def __init__(self, **kwargs) -> None:
        MappingTaker.__init__(self, kwargs)


def take_from_arg(arg: tp.Any, take_spec: TakeSpecT, chunk_meta: ChunkMeta, **kwargs) -> tp.Any:
    """Take from the argument given the specification `take_spec`.

    If `take_spec` is None, returns the original object. Otherwise, must be an instance of `ChunkTaker`."""
    if take_spec is None:
        return arg
    if isinstance(take_spec, ChunkTaker):
        return take_spec.take(arg, chunk_meta, **kwargs)
    raise TypeError(f"Specification of type {type(take_spec)} is not supported")


def take_from_args(ann_args: ann_argsT,
                   arg_take_spec: ArgTakeSpecT,
                   chunk_meta: ChunkMeta) -> tp.Tuple[tp.Args, tp.Kwargs]:
    """Take from each in the annotated arguments given the specification using `take_from_arg`.

    `arg_take_spec` must be a dictionary, with keys being argument names as generated by
    `vectorbt.utils.parsing.annotate_args`. For values, see `take_from_arg`.

    Returns arguments and keyword arguments that can be directly passed to the function
    using `func(*args, **kwargs)`."""
    new_args = ()
    new_kwargs = dict()
    for arg_name, ann_arg in ann_args.items():
        take_spec = arg_take_spec.get(arg_name, None)
        result = take_from_arg(ann_arg['value'], take_spec, chunk_meta, arg_take_spec=arg_take_spec)
        if ann_arg['kind'] == inspect.Parameter.VAR_POSITIONAL:
            for new_arg in result:
                new_args += (new_arg,)
        elif ann_arg['kind'] == inspect.Parameter.VAR_KEYWORD:
            for new_kwarg_name, new_kwarg in result.items():
                new_kwargs[new_kwarg_name] = new_kwarg
        elif ann_arg['kind'] == inspect.Parameter.KEYWORD_ONLY:
            new_kwargs[arg_name] = result
        else:
            new_args += (result,)
    return new_args, new_kwargs


# ############# Splitting ############# #

funcs_argsT = tp.Tuple[tp.Callable, tp.Args, tp.Kwargs]


def split_args(func: tp.Callable,
               args: tp.Args,
               kwargs: tp.Kwargs,
               n_chunks: tp.Optional[tp.Union[int, Sizer, tp.Callable]] = None,
               size: tp.Optional[tp.Union[int, Sizer, tp.Callable]] = None,
               chunk_len: tp.Optional[tp.Union[int, Sizer, tp.Callable]] = None,
               chunk_meta: tp.Optional[tp.Union[tp.Iterable[ChunkMeta], ChunkMetaGenerator, tp.Callable]] = None,
               arg_take_spec: tp.Optional[tp.Union[ArgTakeSpecT, tp.Callable]] = None) -> tp.List[funcs_argsT]:
    """Split arguments and keyword arguments.

    Annotates the arguments using `vectorbt.utils.parsing.annotate_args`, uses or generates chunk metadata,
    splits the arguments into chunks, and returns them.

    Args:
        func (callable): Function.
        args (tuple): Tuple of arguments passed to `func`.
        kwargs (dict): Dictionary with keyword arguments passed to `func`.
        n_chunks (int, Sizer, or callable): Number of chunks.

            Can be an integer, an instance of `Sizer`, or a callable taking the annotated arguments
            and returning an integer.
        size (int, Sizer, or callable): Size of the space to split.

            Can be an integer, an instance of `Sizer`, or a callable taking the annotated arguments
            and returning an integer.
        chunk_len (int, Sizer, or callable): Length of each chunk.

            Can be an integer, an instance of `Sizer`, or a callable taking the annotated arguments
            and returning an integer.
        chunk_meta (iterable of ChunkMeta, ChunkMetaGenerator, or callable): Chunk meta.

            Can be an iterable of `ChunkMeta`, an instance of `ChunkMetaGenerator`, or
            a callable taking the annotated arguments and returning an iterable.
        arg_take_spec (dict or callable): Chunk taking specification.

            Can be a callable taking the annotated arguments and returning new arguments and
            keyword arguments. Otherwise, see `take_from_args`.

    Returns a list of tuples. Each tuple corresponds to a chunk and contains `func`, the chunk's arguments,
    and the chunk's keyword arguments."""

    ann_args = annotate_args(func, *args, **kwargs)

    if chunk_meta is None:
        if n_chunks is not None:
            if isinstance(n_chunks, Sizer):
                n_chunks = n_chunks.get_size(ann_args)
            elif callable(n_chunks):
                n_chunks = n_chunks(ann_args)
            elif not isinstance(n_chunks, int):
                raise TypeError(f"Type {type(n_chunks)} for n_chunks is not supported")
        if size is not None:
            if isinstance(size, Sizer):
                size = size.get_size(ann_args)
            elif callable(size):
                size = size(ann_args)
            elif not isinstance(size, int):
                raise TypeError(f"Type {type(size)} for size is not supported")
        if chunk_len is not None:
            if isinstance(chunk_len, Sizer):
                chunk_len = chunk_len.get_size(ann_args)
            elif callable(chunk_len):
                chunk_len = chunk_len(ann_args)
            elif not isinstance(chunk_len, int):
                raise TypeError(f"Type {type(chunk_len)} for chunk_len is not supported")
        chunk_meta = yield_chunk_meta(
            n_chunks=n_chunks,
            size=size,
            chunk_len=chunk_len
        )
    elif isinstance(chunk_meta, ChunkMetaGenerator):
        chunk_meta = chunk_meta.get_chunk_meta(ann_args)
    elif callable(chunk_meta):
        chunk_meta = chunk_meta(ann_args)

    funcs_args = []
    for _chunk_meta in chunk_meta:
        mapping = {**ann_args, 'chunk_meta': _chunk_meta}
        chunk_ann_args = deep_substitute(ann_args, mapping=mapping)
        if arg_take_spec is None:
            arg_take_spec = {}
        if callable(arg_take_spec):
            chunk_args, chunk_kwargs = arg_take_spec(chunk_ann_args, _chunk_meta)
        else:
            chunk_args, chunk_kwargs = take_from_args(chunk_ann_args, arg_take_spec, _chunk_meta)
        funcs_args.append((func, chunk_args, chunk_kwargs))

    return funcs_args


# ############# Running ############# #


def run_using_sequence(funcs_args: tp.Iterable[funcs_argsT]) -> list:
    """Run a sequence of functions in a serial manner.

    This function is mostly used for testing purposes."""
    return [func(*args, **kwargs) for func, args, kwargs in funcs_args]


funcs_args_refsT = tp.Tuple[RemoteFunctionT, tp.Tuple[ObjectRefT, ...], tp.Dict[str, ObjectRefT]]


def get_ray_refs(funcs_args: tp.Iterable[funcs_argsT],
                 reuse_refs: bool = True,
                 remote_kwargs: tp.KwargsLike = None) -> tp.List[funcs_args_refsT]:
    """Get result references by putting each argument and keyword argument into the object store
    and invoking the remote decorator on each function using Ray.

    If `reuse_refs` is True, will generate one reference per unique object id."""
    import ray
    from ray.remote_function import RemoteFunction
    from ray import ObjectRef

    if remote_kwargs is None:
        remote_kwargs = {}

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

        # Get id of each (unique) arg
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

        # Get id of each (unique) kwarg
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
                  remote_kwargs: tp.KwargsLike = None) -> list:
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


def run_ntimes_using_ray(n: int, func: tp.Callable, *args, ray_kwargs: tp.KwargsLike = None, **kwargs) -> list:
    """Run a function a number of times using `run_using_ray`.

    `func` must accept the (incrementing) index of the run, `*args`, and `**kwargs`.
    Use the run index, for example, to select a portion of the data to analyze."""
    if ray_kwargs is None:
        ray_kwargs = {}

    return run_using_ray([(func, (i, *args), kwargs) for i in range(n)], **ray_kwargs)


def chunked(*args,
            prepend_chunk_meta: tp.Optional[bool] = None,
            engine: tp.Optional[tp.Union[str, tp.Callable]] = None,
            engine_kwargs: tp.KwargsLike = None,
            merge_func: tp.Optional[tp.Callable] = None,
            **split_kwargs) -> tp.Callable:
    """Make a function chunked. Engine-agnostic.

    Returns a function with the same signature as the passed function doing the following:

    * Splits the arguments using `split_args`.
        The keyword arguments `split_kwargs` are passed to this function.
    * Passes the list of chunks to the engine function for execution.
        The keyword arguments `engine_kwargs` are passed to this function.
    * Optionally, post-processes and merges the results using `merge_func`.

    For defaults, see `parallel` in `vectorbt._settings.settings`.

    The following engines are supported:

    * 'ray': See `run_using_ray`.
    * 'sequence': See `run_using_sequence`.

    Each parameter can be modified in the `options` attribute of the wrapper function or
    passed as a keyword argument with a leading underscore directly.

    !!! hint
        Run this function sequentially by passing `engine='sequence'` for testing.

    ## Example

    For testing purposes, let's divide the input array into 2 chunks and compute the mean in a sequential manner:

    ```python-repl
    >>> import vectorbt as vbt
    >>> import numpy as np

    >>> @vbt.chunked(
    ...     n_chunks=2,
    ...     size=vbt.LenSizer('a'),
    ...     arg_take_spec=dict(a=vbt.ChunkSlicer()),
    ...     engine='sequence')
    ... def f(a):
    ...     return np.mean(a)

    >>> f(np.arange(10))
    [2.0, 7.0]
    ```

    The `chunked` function is a decorator that takes `f` and creates a function that splits and
    runs all calculations in parallel. It has the same signature as the original function:

    ```python-repl
    >>> f
    <function __main__.f(a)>
    ```

    We can change any option at any time:

    ```python-repl
    >>> # Change the option directly on the function
    >>> f.options.n_chunks = 3

    >>> f(np.arange(10))
    [1.5, 5.0, 8.0]

    >>> # Pass a new option with a leading underscore
    >>> f(np.arange(10), _n_chunks=4)
    [1.0, 4.0, 6.5, 8.5]
    ```

    When we run the wrapped function, it first generates a list of chunk metadata of type `ChunkMeta`.
    Chunk metadata contains the chunk index that can be used to split any input:

    ```python-repl
    >>> from vectorbt.utils.parallel import yield_chunk_meta

    >>> list(yield_chunk_meta(n_chunks=2))
    [ChunkMeta(idx=0, range_start=None, range_end=None),
     ChunkMeta(idx=1, range_start=None, range_end=None)]
    ```

    Additionally, it may contain the start and end index of the space we want to split.
    This space can be the length of an input array, for example. In our case:

    ```python-repl
    >>> list(yield_chunk_meta(n_chunks=2, size=10))
    [ChunkMeta(idx=0, range_start=0, range_end=5),
     ChunkMeta(idx=1, range_start=5, range_end=10)]
    ```

    If we know the size of the space in advance, we can pass it as an integer constant.
    Otherwise, we need to tell `chunked` to derive the size from the inputs dynamically
    by passing any subclass of `Sizer`. In the example above, we instruct the wrapped function
    to derive the size from the length of the input array `a`.

    Once all chunks are generated, the wrapped function attempts to split inputs into chunks.
    The specification for this operation can be provided by the `arg_take_spec` argument, which
    in most cases is a dictionary of `ChunkTaker` instances keyed by the input name.
    Here's an example of a complex specification:

    ```python-repl
    >>> arg_take_spec = dict(
    ...     a=vbt.ChunkSelector(),
    ...     args=vbt.ArgsTaker(
    ...         None,
    ...         vbt.ChunkSelector()
    ...     ),
    ...     b=vbt.SequenceTaker([
    ...         None,
    ...         vbt.ChunkSelector()
    ...     ]),
    ...     kwargs=vbt.KwargsTaker(
    ...         c=vbt.MappingTaker(dict(
    ...             d=vbt.ChunkSelector()
    ...         ))
    ...     )
    ... )

    >>> @vbt.chunked(
    ...     n_chunks=vbt.LenSizer('a'),
    ...     arg_take_spec=arg_take_spec,
    ...     engine='sequence')
    ... def f(a, *args, b=None, **kwargs):
    ...     return a + sum(args) + sum(b) + sum(kwargs['c'].values())

    >>> f([1, 2, 3], 10, [1, 2, 3], b=(100, [1, 2, 3]), c=dict(d=[1, 2, 3], e=1000))
    [1114, 1118, 1122]
    ```

    After splitting all inputs into chunks, the wrapped function forwards them to the engine function.
    The engine argument can be either the name of a supported engine, or a callable. Once the engine
    has finished all tasks and returned a list of results, we can merge them back using `merge_func`:

    ```python-repl
    >>> @vbt.chunked(
    ...     n_chunks=2,
    ...     size=vbt.LenSizer('a'),
    ...     arg_take_spec=dict(a=vbt.ChunkSlicer()),
    ...     engine='sequence',
    ...     merge_func=np.concatenate)
    ... def f(a):
    ...     return a

    >>> f(np.arange(10))
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    ```

    Instead of (or in addition to) specifying `arg_take_spec`, we can define our function with the
    first argument being `chunk_meta` to be able to split the arguments during the execution.
    The `chunked` decorator will automatically recognize and replace it with the actual `ChunkMeta` object:

    ```python-repl
    >>> @vbt.chunked(
    ...     n_chunks=2,
    ...     size=vbt.LenSizer('a'),
    ...     engine='sequence',
    ...     merge_func=np.concatenate)
    ... def f(chunk_meta, a):
    ...     return a[chunk_meta.range_start:chunk_meta.range_end]

    >>> f(np.arange(10))
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    ```

    This may be a good idea in multi-threading, but a bad idea in multi-processing.

    The same can be accomplished by using templates (here we tell `chunked` to not replace
    the first argument by setting `prepend_chunk_meta` to False):

    ```python-repl
    >>> @vbt.chunked(
    ...     n_chunks=2,
    ...     size=vbt.LenSizer('a'),
    ...     engine='sequence',
    ...     merge_func=np.concatenate,
    ...     prepend_chunk_meta=False)
    ... def f(chunk_meta, a):
    ...     return a[chunk_meta.range_start:chunk_meta.range_end]

    >>> f(vbt.Rep('chunk_meta'), np.arange(10))
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    ```

    Templates in arguments are substituted right before taking a chunk from them.
    """

    def decorator(func: tp.Callable) -> tp.Callable:
        nonlocal prepend_chunk_meta

        if prepend_chunk_meta is None:
            prepend_chunk_meta = False
            func_arg_names = get_func_arg_names(func)
            if len(func_arg_names) > 0:
                if func_arg_names[0] == 'chunk_meta':
                    prepend_chunk_meta = True

        @wraps(func)
        def wrapper(*args, **kwargs):
            from vectorbt._settings import settings
            parallel_cfg = settings['parallel']

            n_chunks = kwargs.pop('_n_chunks', wrapper.options['n_chunks'])
            size = kwargs.pop('_size', wrapper.options['size'])
            chunk_len = kwargs.pop('_chunk_len', wrapper.options['chunk_len'])
            chunk_meta = kwargs.pop('_chunk_meta', wrapper.options['chunk_meta'])
            arg_take_spec = kwargs.pop('_arg_take_spec', wrapper.options['arg_take_spec'])
            engine = kwargs.pop('_engine', wrapper.options['engine'])
            engine_kwargs = merge_dicts(wrapper.options['engine_kwargs'], kwargs.pop('_engine_kwargs', {}))
            merge_func = kwargs.pop('_merge_func', wrapper.options['merge_func'])

            if prepend_chunk_meta:
                args = (Rep('chunk_meta'), *args)

            funcs_args = split_args(
                func,
                args,
                kwargs,
                n_chunks=n_chunks,
                size=size,
                chunk_len=chunk_len,
                chunk_meta=chunk_meta,
                arg_take_spec=arg_take_spec
            )
            if engine is None:
                engine = parallel_cfg['engine']
            if isinstance(engine, str):
                if engine.lower() == 'ray':
                    results = run_using_ray(funcs_args, **engine_kwargs)
                elif engine.lower() == 'sequence':
                    results = run_using_sequence(funcs_args)
                else:
                    raise ValueError(f"Engine '{type(engine)}' is not supported")
            elif callable(engine):
                results = engine(funcs_args, **engine_kwargs)
            else:
                raise TypeError(f"Engine type {type(engine)} is not supported")
            if merge_func is not None:
                return merge_func(results)
            return results

        wrapper.options = Config(
            dict(
                n_chunks=split_kwargs.pop('n_chunks', None),
                size=split_kwargs.pop('size', None),
                chunk_len=split_kwargs.pop('chunk_len', None),
                chunk_meta=split_kwargs.pop('chunk_meta', None),
                arg_take_spec=split_kwargs.pop('arg_take_spec', None),
                engine=engine,
                engine_kwargs=engine_kwargs,
                merge_func=merge_func
            ),
            frozen_keys=True
        )

        if len(split_kwargs) > 0:
            for k in split_kwargs:
                raise TypeError(f"chunked() got an unexpected keyword argument '{k}'")

        if prepend_chunk_meta:
            signature = inspect.signature(wrapper)
            wrapper.__signature__ = signature.replace(parameters=tuple(signature.parameters.values())[1:])

        return wrapper

    if len(args) == 0:
        return decorator
    elif len(args) == 1:
        return decorator(args[0])
    raise ValueError("Either function or keyword arguments must be passed")
