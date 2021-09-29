# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""Utilities for chunking."""

import pandas as pd
import inspect
import multiprocessing
from functools import wraps

from vectorbt import _typing as tp
from vectorbt.utils import checks
from vectorbt.utils.config import merge_dicts, Config
from vectorbt.utils.parsing import annotate_args, ann_argsT, get_from_ann_args, get_func_arg_names
from vectorbt.utils.template import deep_substitute, Rep
from vectorbt.utils.execution import funcs_argsT, ExecutionEngine, SequenceEngine, DaskEngine, RayEngine

__pdoc__ = {}


# ############# Chunk sizing ############# #


class Sizer:
    """Abstract class for getting the size from annotated arguments."""

    def get_size(self, ann_args: ann_argsT) -> int:
        """Get the size given the annotated arguments."""
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

    For defaults, see `chunking` in `vectorbt._settings.settings`."""
    from vectorbt._settings import settings
    chunking_cfg = settings['chunking']

    if n_chunks is None:
        n_chunks = chunking_cfg['n_chunks']
    if chunk_len is None:
        chunk_len = chunking_cfg['chunk_len']

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


any_sizeT = tp.Union[int, Sizer, tp.Callable]
any_chunk_metaT = tp.Union[tp.Iterable[ChunkMeta], ChunkMetaGenerator, tp.Callable]


def get_chunk_meta_from_args(ann_args: ann_argsT,
                             n_chunks: tp.Optional[any_sizeT] = None,
                             size: tp.Optional[any_sizeT] = None,
                             chunk_len: tp.Optional[any_sizeT] = None,
                             chunk_meta: tp.Optional[any_chunk_metaT] = None) -> tp.Iterable[ChunkMeta]:
    """Get chunk metadata from annotated arguments.

    Args:
        ann_args (dict): Arguments annotated with `vectorbt.utils.parsing.annotate_args`.
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
            a callable taking the annotated arguments and returning an iterable."""
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
        return yield_chunk_meta(
            n_chunks=n_chunks,
            size=size,
            chunk_len=chunk_len
        )
    if isinstance(chunk_meta, ChunkMetaGenerator):
        return chunk_meta.get_chunk_meta(ann_args)
    if callable(chunk_meta):
        return chunk_meta(ann_args)
    return chunk_meta


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


class AxisTaker(ChunkTaker):
    """Abstract class for taking one or more elements from an axis."""

    def __init__(self, axis: int) -> None:
        ChunkTaker.__init__(self)

        if axis < 0:
            raise ValueError("Axis cannot be negative")
        self._axis = axis

    @property
    def axis(self) -> int:
        """Axis of the argument to take from."""
        return self._axis


class ArraySelector(AxisTaker):
    """Class for selecting one element from an axis based on the chunk index."""

    def take(self, obj: tp.AnyArray, chunk_meta: ChunkMeta, **kwargs) -> tp.ArrayLike:
        if self.axis >= len(obj.shape):
            raise IndexError(f"Array is {obj.ndim}-dimensional, but axis {self.axis} was indexed")
        slc = [slice(None)] * len(obj.shape)
        slc[self.axis] = chunk_meta.idx
        if isinstance(obj, (pd.Series, pd.DataFrame)):
            return obj.iloc[tuple(slc)]
        return obj[tuple(slc)]


class ArraySlicer(AxisTaker):
    """Class for slicing multiple elements from an axis based on the chunk range."""

    def take(self, obj: tp.AnyArray, chunk_meta: ChunkMeta, **kwargs) -> tp.AnyArray:
        if self.axis >= len(obj.shape):
            raise IndexError(f"Array is {obj.ndim}-dimensional, but axis {self.axis} was indexed")
        slc = [slice(None)] * len(obj.shape)
        slc[self.axis] = slice(chunk_meta.range_start, chunk_meta.range_end)
        if isinstance(obj, (pd.Series, pd.DataFrame)):
            return obj.iloc[tuple(slc)]
        return obj[tuple(slc)]


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


def yield_arg_chunks(func: tp.Callable,
                     ann_args: ann_argsT,
                     chunk_meta: tp.Iterable[ChunkMeta],
                     arg_take_spec: tp.Optional[tp.Union[ArgTakeSpecT, tp.Callable]] = None,
                     template_mapping: tp.Optional[tp.Mapping] = None) -> tp.Generator[funcs_argsT, None, None]:
    """Split annotated arguments into chunks and yield each chunk.

    Args:
        func (callable): Callable.
        ann_args (dict): Arguments annotated with `vectorbt.utils.parsing.annotate_args`.
        chunk_meta (iterable of ChunkMeta): Chunk metadata.
        arg_take_spec (dict or callable): Chunk taking specification.

            Can be a callable taking the annotated arguments and returning new arguments and
            keyword arguments. Otherwise, see `take_from_args`.
        template_mapping (mapping): Mapping to replace templates in arguments.

    For defaults, see `chunking` in `vectorbt._settings.settings`."""

    from vectorbt._settings import settings
    chunking_cfg = settings['chunking']

    template_mapping = merge_dicts(chunking_cfg['template_mapping'], template_mapping)

    for _chunk_meta in chunk_meta:
        mapping = merge_dicts(ann_args, dict(chunk_meta=_chunk_meta), template_mapping)
        chunk_ann_args = deep_substitute(ann_args, mapping=mapping)
        if arg_take_spec is None:
            arg_take_spec = {}
        if callable(arg_take_spec):
            chunk_args, chunk_kwargs = arg_take_spec(chunk_ann_args, _chunk_meta)
        else:
            chunk_args, chunk_kwargs = take_from_args(chunk_ann_args, arg_take_spec, _chunk_meta)
        yield func, chunk_args, chunk_kwargs


def chunked(*args,
            prepend_chunk_meta: tp.Optional[bool] = None,
            engine: tp.Optional[tp.Union[str, type, ExecutionEngine, tp.Callable]] = None,
            engine_kwargs: tp.KwargsLike = None,
            merge_func: tp.Optional[tp.Callable] = None,
            **kwargs) -> tp.Callable:
    """Decorator for chunking a function. Engine-agnostic.

    Args:
        prepend_chunk_meta (bool): Whether to prepend an instance of `ChunkMeta` to the arguments.

            If None, prepends automatically if the first argument is named 'chunk_meta'.
        engine (str, type, ExecutionEngine, or callable): Engine for executing chunks.

            Supported values:

            * Name of the engine (see supported engines)
            * Subclass of `vectorbt.utils.execution.ExecutionEngine` -
                will initialize with `**engine_kwargs`
            * Instance of `vectorbt.utils.execution.ExecutionEngine` -
                will call `vectorbt.utils.execution.ExecutionEngine.run`
            * Callable - will pass an iterable of function and argument tuples,
                chunk metadata, and `**engine_kwargs`

            Supported engines:

            * 'sequence' (default): See `vectorbt.utils.execution.SequenceEngine`.
            * 'dask': See `vectorbt.utils.execution.DaskEngine`.
            * 'ray': See `vectorbt.utils.execution.RayEngine`.
        engine_kwargs (dict): Keyword arguments passed to `engine`.
        merge_func (callable): Function to post-process and merge the results.
        **kwargs: Keyword arguments for chunking (see `get_chunk_meta_from_args` and `yield_arg_chunks`).

    For defaults, see `chunking` in `vectorbt._settings.settings`.

    For example, to switch the engine globally:

    ```python-repl
    >>> import vectorbt as vbt

    >>> vbt.settings.chunking['engine'] = 'dask'
    ```

    Returns a function with the same signature as the passed function.

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
    ...     arg_take_spec=dict(a=vbt.ChunkSlicer()))
    ... def f(a):
    ...     return np.mean(a)

    >>> f(np.arange(10))
    [2.0, 7.0]
    ```

    The `chunked` function is a decorator that takes `f` and creates a function that splits
    passed arguments, runs each chunk using an engine, and optionally, merges the results.
    It has the same signature as the original function:

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
    >>> from vectorbt.utils.chunking import yield_chunk_meta

    >>> list(yield_chunk_meta(n_chunks=2))
    [ChunkMeta(idx=0, range_start=None, range_end=None),
     ChunkMeta(idx=1, range_start=None, range_end=None)]
    ```

    Additionally, it may contain the start and end index of the space we want to split.
    The space can be defined by the length of an input array, for example. In our case:

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
    ...     arg_take_spec=arg_take_spec)
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
    ...     merge_func=np.concatenate,
    ...     prepend_chunk_meta=False)
    ... def f(chunk_meta, a):
    ...     return a[chunk_meta.range_start:chunk_meta.range_end]

    >>> f(vbt.Rep('chunk_meta'), np.arange(10))
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    ```

    Templates in arguments are substituted right before taking a chunk from them.

    Keyword arguments to the engine can be provided using `engine_kwargs`:

    ```python-repl
    >>> @vbt.chunked(
    ...     n_chunks=2,
    ...     size=vbt.LenSizer('a'),
    ...     arg_take_spec=dict(a=vbt.ChunkSlicer()),
    ...     engine_kwargs=dict(show_progress=True))  # see run_using_sequence
    ... def f(a):
    ...     return np.mean(a)

    >>> f(np.arange(10))
    100% |█████████████████████████████████| 2/2 [00:00<00:00, 81.11it/s]
    [2.0, 7.0]
    ```
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
            chunking_cfg = settings['chunking']

            n_chunks = kwargs.pop('_n_chunks', wrapper.options['n_chunks'])
            size = kwargs.pop('_size', wrapper.options['size'])
            chunk_len = kwargs.pop('_chunk_len', wrapper.options['chunk_len'])
            chunk_meta = kwargs.pop('_chunk_meta', wrapper.options['chunk_meta'])
            arg_take_spec = kwargs.pop('_arg_take_spec', wrapper.options['arg_take_spec'])
            template_mapping = merge_dicts(wrapper.options['template_mapping'], kwargs.pop('_template_mapping', {}))
            engine = kwargs.pop('_engine', wrapper.options['engine'])
            engine_kwargs = merge_dicts(wrapper.options['engine_kwargs'], kwargs.pop('_engine_kwargs', {}))
            merge_func = kwargs.pop('_merge_func', wrapper.options['merge_func'])

            if prepend_chunk_meta:
                args = (Rep('chunk_meta'), *args)

            ann_args = annotate_args(func, *args, **kwargs)
            chunk_meta = list(get_chunk_meta_from_args(
                ann_args,
                n_chunks=n_chunks,
                size=size,
                chunk_len=chunk_len,
                chunk_meta=chunk_meta
            ))
            funcs_args = yield_arg_chunks(
                func,
                ann_args,
                chunk_meta=chunk_meta,
                arg_take_spec=arg_take_spec,
                template_mapping=template_mapping
            )
            if engine is None:
                engine = chunking_cfg['engine']
            if isinstance(engine, str):
                if engine.lower() == 'sequence':
                    engine = SequenceEngine
                elif engine.lower() == 'ray':
                    engine = RayEngine
                elif engine.lower() == 'dask':
                    engine = DaskEngine
                else:
                    raise ValueError(f"Engine '{type(engine)}' is not supported")
            if isinstance(engine, type) and issubclass(engine, ExecutionEngine):
                engine = engine(**engine_kwargs)
            if isinstance(engine, ExecutionEngine):
                results = engine.run(funcs_args, n_calls=len(chunk_meta))
            elif callable(engine):
                results = engine(funcs_args, chunk_meta, **engine_kwargs)
            else:
                raise TypeError(f"Engine type {type(engine)} is not supported")
            if merge_func is not None:
                return merge_func(results)
            return results

        wrapper.options = Config(
            dict(
                n_chunks=kwargs.pop('n_chunks', None),
                size=kwargs.pop('size', None),
                chunk_len=kwargs.pop('chunk_len', None),
                chunk_meta=kwargs.pop('chunk_meta', None),
                arg_take_spec=kwargs.pop('arg_take_spec', None),
                template_mapping=kwargs.pop('template_mapping', None),
                engine=engine,
                engine_kwargs=engine_kwargs,
                merge_func=merge_func
            ),
            frozen_keys=True
        )

        if len(kwargs) > 0:
            for k in kwargs:
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
