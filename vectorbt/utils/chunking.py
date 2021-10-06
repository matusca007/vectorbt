# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""Utilities for chunking."""

import numpy as np
import pandas as pd
import inspect
import multiprocessing
from functools import wraps

from vectorbt import _typing as tp
from vectorbt.utils import checks
from vectorbt.utils.config import merge_dicts, Config
from vectorbt.utils.parsing import annotate_args, ann_argsT, get_from_ann_args, get_func_arg_names
from vectorbt.utils.template import deep_substitute, Rep
from vectorbt.utils.execution import funcs_argsT, engineT, execute

__pdoc__ = {}


# ############# Named tuples ############# #

class ChunkMeta(tp.NamedTuple):
    idx: int
    start: tp.Optional[int]
    end: tp.Optional[int]
    indices: tp.Optional[tp.Sequence[int]]


__pdoc__['ChunkMeta'] = "A named tuple representing a chunk metadata."
__pdoc__['ChunkMeta.idx'] = "Chunk index."
__pdoc__['ChunkMeta.start'] = "Start of the chunk range (including). Can be None."
__pdoc__['ChunkMeta.end'] = "End of the chunk range (excluding). Can be None."
__pdoc__['ChunkMeta.indices'] = """Indices included in the chunk range. Can be None.

Has priority over `ChunkMeta.start` and `ChunkMeta.end`."""


# ############# Mixins ############# #


class ArgGetterMixin:
    """Class for getting an argument from annotated arguments."""

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


class AxisMixin:
    """Mixin class with an attribute for specifying an axis."""

    def __init__(self, axis: int) -> None:
        if axis < 0:
            raise ValueError("Axis cannot be negative")
        self._axis = axis

    @property
    def axis(self) -> int:
        """Axis of the argument to take from."""
        return self._axis


class DimRetainerMixin:
    """Mixin class with an attribute for retaining dimensions."""

    def __init__(self, retain_dim: bool) -> None:
        self._retain_dim = retain_dim

    @property
    def retain_dim(self) -> bool:
        """Whether to retain dimensions."""
        return self._retain_dim


# ############# Chunk sizing ############# #


class Sizer:
    """Abstract class for getting the size from annotated arguments."""

    def get_size(self, ann_args: ann_argsT) -> int:
        """Get the size given the annotated arguments."""
        raise NotImplementedError


class ArgSizer(Sizer, ArgGetterMixin):
    """Class for getting the size from an argument."""

    def __init__(self, arg: tp.Union[str, int]) -> None:
        Sizer.__init__(self)
        ArgGetterMixin.__init__(self, arg)

    def get_size(self, ann_args: ann_argsT) -> int:
        return self.get_arg(ann_args)


class LenSizer(ArgSizer):
    """Class for getting the size from the length of an argument."""

    def get_size(self, ann_args: ann_argsT) -> int:
        return len(self.get_arg(ann_args))


class ShapeSizer(ArgSizer, AxisMixin):
    """Class for getting the size from the length of an axis in a shape."""

    def __init__(self, arg: tp.Union[str, int], axis: int) -> None:
        ArgSizer.__init__(self, arg)
        AxisMixin.__init__(self, axis)

    def get_size(self, ann_args: ann_argsT) -> int:
        arg = self.get_arg(ann_args)
        if self.axis <= len(arg) - 1:
            return arg[self.axis]
        return 0


class ArraySizer(ShapeSizer):
    """Class for getting the size from the length of an axis in an array."""

    def get_size(self, ann_args: ann_argsT) -> int:
        arg = self.get_arg(ann_args)
        if self.axis <= len(arg.shape) - 1:
            return arg.shape[self.axis]
        return 0


# ############# Chunk generation ############# #


def yield_chunk_meta(n_chunks: tp.Optional[int] = None,
                     size: tp.Optional[int] = None,
                     min_size: tp.Optional[int] = None,
                     chunk_len: tp.Optional[int] = None) -> tp.Generator[ChunkMeta, None, None]:
    """Yield meta of each successive chunk from a sequence with a number of elements.

    If both `n_chunks` and `chunk_len` are None (after resolving them from settings),
    sets `n_chunks` to the number of cores.

    For defaults, see `chunking` in `vectorbt._settings.settings`."""
    from vectorbt._settings import settings
    chunking_cfg = settings['chunking']

    if n_chunks is None:
        n_chunks = chunking_cfg['n_chunks']
    if min_size is None:
        min_size = chunking_cfg['min_size']
    if chunk_len is None:
        chunk_len = chunking_cfg['chunk_len']

    if size is not None and min_size is not None and size < min_size:
        yield ChunkMeta(
            idx=0,
            start=0,
            end=size,
            indices=None
        )
    else:
        if n_chunks is None and chunk_len is None:
            n_chunks = multiprocessing.cpu_count()
        if n_chunks is not None and chunk_len is not None:
            raise ValueError("Either n_chunks or chunk_len must be set, not both")
        if n_chunks is not None:
            if n_chunks == 0:
                raise ValueError("Chunk count cannot be zero")
            if size is not None:
                if n_chunks > size:
                    n_chunks = size
                d, r = divmod(size, n_chunks)
                for i in range(n_chunks):
                    si = (d + 1) * (i if i < r else r) + d * (0 if i < r else i - r)
                    yield ChunkMeta(
                        idx=i,
                        start=si,
                        end=si + (d + 1 if i < r else d),
                        indices=None
                    )
            else:
                for i in range(n_chunks):
                    yield ChunkMeta(
                        idx=i,
                        start=None,
                        end=None,
                        indices=None
                    )
        if chunk_len is not None:
            checks.assert_not_none(size)
            if chunk_len == 0:
                raise ValueError("Chunk length cannot be zero")
            for chunk_i, i in enumerate(range(0, size, chunk_len)):
                yield ChunkMeta(
                    idx=chunk_i,
                    start=i,
                    end=min(i + chunk_len, size),
                    indices=None
                )


class ChunkMetaGenerator:
    """Abstract class for generating chunk metadata from annotated arguments."""

    def get_chunk_meta(self, ann_args: ann_argsT) -> tp.Iterable[ChunkMeta]:
        """Get chunk metadata."""
        raise NotImplementedError


class ArgChunkMeta(ChunkMetaGenerator, ArgGetterMixin):
    """Class for generating chunk metadata from an argument."""

    def __init__(self, arg: tp.Union[str, int]) -> None:
        ChunkMetaGenerator.__init__(self)
        ArgGetterMixin.__init__(self, arg)

    def get_chunk_meta(self, ann_args: ann_argsT) -> tp.Iterable[ChunkMeta]:
        return self.get_arg(ann_args)


class LenChunkMeta(ArgChunkMeta):
    """Class for generating chunk metadata from a sequence of chunk lengths."""

    def get_chunk_meta(self, ann_args: ann_argsT) -> tp.Iterable[ChunkMeta]:
        arg = self.get_arg(ann_args)
        start = 0
        end = 0
        for i, chunk_len in enumerate(arg):
            end += chunk_len
            yield ChunkMeta(
                idx=i,
                start=start,
                end=end,
                indices=None
            )
            start = end


any_sizeT = tp.Union[int, Sizer, tp.Callable]
any_chunk_metaT = tp.Union[tp.Iterable[ChunkMeta], ChunkMetaGenerator, tp.Callable]


def get_chunk_meta_from_args(ann_args: ann_argsT,
                             n_chunks: tp.Optional[any_sizeT] = None,
                             size: tp.Optional[any_sizeT] = None,
                             min_size: tp.Optional[int] = None,
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
        min_size (int): If `size` is lower than this number, returns a single chunk.
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
            min_size=min_size,
            chunk_len=chunk_len
        )
    if isinstance(chunk_meta, ChunkMetaGenerator):
        return chunk_meta.get_chunk_meta(ann_args)
    if callable(chunk_meta):
        return chunk_meta(ann_args)
    return chunk_meta


# ############# Chunk mapping ############# #


class ChunkMapper:
    """Abstract class for mapping chunk metadata.

    Implements the abstract `ChunkMapper.map` method."""

    def map(self, chunk_meta: ChunkMeta, **kwargs) -> ChunkMeta:
        """Abstract method for mapping chunk metadata.

        Takes the chunk metadata of type `ChunkMeta` and returns a new chunk metadata of the same type."""
        raise NotImplementedError


# ############# Chunk taking ############# #


class ChunkTaker:
    """Abstract class for taking one or more elements based on the chunk index or range."""

    def __init__(self, mapper: tp.Optional[ChunkMapper] = None) -> None:
        self._mapper = mapper

    @property
    def mapper(self) -> tp.Optional[ChunkMapper]:
        """Chunk mapper of type `ChunkMapper`."""
        return self._mapper

    def map_and_take(self, obj: tp.Any, chunk_meta: ChunkMeta, **kwargs) -> tp.Any:
        """Map chunk metadata using `ChunkTaker.mapper` (if not None) and take the chunk using `ChunkTaker.take`."""
        if self.mapper is not None:
            chunk_meta = self.mapper.map(chunk_meta, **kwargs)
        return self.take(obj, chunk_meta, **kwargs)

    def take(self, obj: tp.Any, chunk_meta: ChunkMeta, **kwargs) -> tp.Any:
        """Abstract method for taking subset of data.

        Takes the argument object, the chunk meta (tuple out of the index, start index,
        and end index of the chunk), and other keyword arguments passed down the stack,
        at least the entire argument specification `arg_take_spec`."""
        raise NotImplementedError


class ChunkSelector(ChunkTaker, DimRetainerMixin):
    """Class for selecting one element based on the chunk index."""

    def __init__(self, mapper: tp.Optional[ChunkMapper] = None, retain_dim: bool = False) -> None:
        ChunkTaker.__init__(self, mapper=mapper)
        DimRetainerMixin.__init__(self, retain_dim)

    def take(self, obj: tp.Sequence, chunk_meta: ChunkMeta, **kwargs) -> tp.Any:
        if self.retain_dim:
            return obj[chunk_meta.idx:chunk_meta.idx + 1]
        return obj[chunk_meta.idx]


class ChunkSlicer(ChunkTaker):
    """Class for slicing multiple elements based on the chunk range."""

    def take(self, obj: tp.Sequence, chunk_meta: ChunkMeta, **kwargs) -> tp.Sequence:
        if chunk_meta.indices is not None:
            return obj[chunk_meta.indices]
        return obj[chunk_meta.start:chunk_meta.end]


class CountAdapter(ChunkSlicer):
    """Class for adapting a count based on the chunk range."""

    def take(self, obj: int, chunk_meta: ChunkMeta, **kwargs) -> int:
        checks.assert_instance_of(obj, int)
        if chunk_meta.indices is not None:
            indices = np.asarray(chunk_meta.indices)
            if np.any(indices >= obj):
                raise IndexError(f"Positional indexers are out-of-bounds")
            return len(indices)
        if chunk_meta.start >= obj:
            return 0
        return min(obj, chunk_meta.end) - chunk_meta.start


class ShapeSelector(ChunkSelector, AxisMixin):
    """Class for selecting one element from a shape's axis based on the chunk index."""

    def __init__(self, axis: int, mapper: tp.Optional[ChunkMapper] = None, retain_dim: bool = False) -> None:
        ChunkSelector.__init__(self, mapper=mapper, retain_dim=retain_dim)
        AxisMixin.__init__(self, axis)

    def take(self, obj: tp.Shape, chunk_meta: ChunkMeta, **kwargs) -> tp.Shape:
        checks.assert_instance_of(obj, tuple)
        if self.axis >= len(obj):
            raise IndexError(f"Shape is {len(obj)}-dimensional, but {self.axis} were indexed")
        if chunk_meta.idx >= obj[self.axis]:
            raise IndexError(f"Index {chunk_meta.idx} is out of bounds for axis {self.axis} with size {obj[self.axis]}")
        obj = list(obj)
        if self.retain_dim:
            obj[self.axis] = 1
        else:
            del obj[self.axis]
        return tuple(obj)


class ShapeSlicer(ChunkSlicer, AxisMixin):
    """Class for slicing multiple elements from a shape's axis based on the chunk range."""

    def __init__(self, axis: int, mapper: tp.Optional[ChunkMapper] = None) -> None:
        ChunkSlicer.__init__(self, mapper=mapper)
        AxisMixin.__init__(self, axis)

    def take(self, obj: tp.Shape, chunk_meta: ChunkMeta, **kwargs) -> tp.Shape:
        checks.assert_instance_of(obj, tuple)
        if self.axis >= len(obj):
            raise IndexError(f"Shape is {len(obj)}-dimensional, but {self.axis} were indexed")
        obj = list(obj)
        if chunk_meta.indices is not None:
            indices = np.asarray(chunk_meta.indices)
            if np.any(indices >= obj[self.axis]):
                raise IndexError(f"Positional indexers are out-of-bounds")
            obj[self.axis] = len(indices)
        else:
            if chunk_meta.start >= obj[self.axis]:
                del obj[self.axis]
            else:
                obj[self.axis] = min(obj[self.axis], chunk_meta.end) - chunk_meta.start
        return tuple(obj)


class ArraySelector(ShapeSelector):
    """Class for selecting one element from an array's axis based on the chunk index."""

    def take(self, obj: tp.AnyArray, chunk_meta: ChunkMeta, **kwargs) -> tp.ArrayLike:
        checks.assert_instance_of(obj, (pd.Series, pd.DataFrame, np.ndarray))
        if self.axis >= len(obj.shape):
            raise IndexError(f"Array is {obj.ndim}-dimensional, but {self.axis} were indexed")
        slc = [slice(None)] * len(obj.shape)
        if self.retain_dim:
            slc[self.axis] = slice(chunk_meta.idx, chunk_meta.idx + 1)
        else:
            slc[self.axis] = chunk_meta.idx
        if isinstance(obj, (pd.Series, pd.DataFrame)):
            return obj.iloc[tuple(slc)]
        return obj[tuple(slc)]


class ArraySlicer(ShapeSlicer):
    """Class for slicing multiple elements from an array's axis based on the chunk range."""

    def take(self, obj: tp.AnyArray, chunk_meta: ChunkMeta, **kwargs) -> tp.AnyArray:
        checks.assert_instance_of(obj, (pd.Series, pd.DataFrame, np.ndarray))
        if self.axis >= len(obj.shape):
            raise IndexError(f"Array is {obj.ndim}-dimensional, but {self.axis} were indexed")
        slc = [slice(None)] * len(obj.shape)
        if chunk_meta.indices is not None:
            slc[self.axis] = np.asarray(chunk_meta.indices)
        else:
            slc[self.axis] = slice(chunk_meta.start, chunk_meta.end)
        if isinstance(obj, (pd.Series, pd.DataFrame)):
            return obj.iloc[tuple(slc)]
        return obj[tuple(slc)]


TakeSpecT = tp.Union[None, ChunkTaker]
MappingTakeSpecT = tp.Mapping[tp.Union[str, int], TakeSpecT]
SequenceTakeSpecT = tp.Sequence[TakeSpecT]
ContainerTakeSpecT = tp.Union[MappingTakeSpecT, SequenceTakeSpecT]
ArgTakeSpecT = MappingTakeSpecT


class ContainerTaker(ChunkTaker):
    """Class for taking from a container with other chunk takers.

    Accepts the specification of the container."""

    def __init__(self, cont_take_spec: ContainerTakeSpecT, mapper: tp.Optional[ChunkMapper] = None) -> None:
        ChunkTaker.__init__(self, mapper=mapper)

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
        for i, v in enumerate(obj):
            if i < len(self.cont_take_spec):
                take_spec = self.cont_take_spec[i]
            else:
                take_spec = None
            new_obj.append(take_from_arg(v, take_spec, chunk_meta, **kwargs))
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

    If `take_spec` is None, returns the original object. Otherwise, must be an instance of `ChunkTaker`.

    `**kwargs` are passed to `ChunkTaker.map_and_take`."""
    if take_spec is None:
        return arg
    if isinstance(take_spec, ChunkTaker):
        return take_spec.map_and_take(arg, chunk_meta, **kwargs)
    raise TypeError(f"Specification of type {type(take_spec)} is not supported")


def take_from_args(ann_args: ann_argsT,
                   arg_take_spec: ArgTakeSpecT,
                   chunk_meta: ChunkMeta) -> tp.Tuple[tp.Args, tp.Kwargs]:
    """Take from each in the annotated arguments given the specification using `take_from_arg`.

    Additionally, passes to `take_from_arg` as keyword arguments `ann_args` and `arg_take_spec`.

    `arg_take_spec` must be a dictionary, with keys being argument positions or names as generated by
    `vectorbt.utils.parsing.annotate_args`. For values, see `take_from_arg`.

    Returns arguments and keyword arguments that can be directly passed to the function
    using `func(*args, **kwargs)`."""
    new_args = ()
    new_kwargs = dict()
    for i, (arg_name, ann_arg) in enumerate(ann_args.items()):
        if arg_name in arg_take_spec:
            take_spec = arg_take_spec[arg_name]
        elif i in arg_take_spec:
            take_spec = arg_take_spec[i]
        else:
            take_spec = None
        result = take_from_arg(
            ann_arg['value'],
            take_spec,
            chunk_meta,
            ann_args=ann_args,
            arg_take_spec=arg_take_spec
        )
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
            n_chunks: tp.Optional[any_sizeT] = None,
            size: tp.Optional[any_sizeT] = None,
            min_size: tp.Optional[int] = None,
            chunk_len: tp.Optional[any_sizeT] = None,
            chunk_meta: tp.Optional[any_chunk_metaT] = None,
            skip_one_chunk: tp.Optional[bool] = None,
            arg_take_spec: tp.Optional[tp.Union[ArgTakeSpecT, tp.Callable]] = None,
            template_mapping: tp.Optional[tp.Mapping] = None,
            prepend_chunk_meta: tp.Optional[bool] = None,
            merge_func: tp.Optional[tp.Callable] = None,
            merge_kwargs: tp.KwargsLike = None,
            engine: engineT = None,
            **engine_kwargs) -> tp.Callable:
    """Decorator that chunks the function. Engine-agnostic.
    Returns a new function with the same signature as the passed one.

    Does the following:

    1. Generates chunk metadata by passing `n_chunks`, `size`, `min_size`, `chunk_len`, and `chunk_meta`
        to `get_chunk_meta_from_args`.
    2. Splits arguments and keyword arguments by passing chunk metadata, `arg_take_spec`,
        and `template_mapping` to `yield_arg_chunks`, which yields one chunk at a time.
    3. Executes all chunks by passing `engine` and `**engine_kwargs` to `vectorbt.utils.execution.execute`.
    4. Optionally, post-processes and merges the results by passing them and `**merge_kwargs` to `merge_func`.

    Any template in both `engine_kwargs` and `merge_kwargs` will be substituted. You can use
    the keys `ann_args`, `chunk_meta`, `arg_take_spec`, and `funcs_args` to be replaced by the actual objects.

    Use `prepend_chunk_meta` to prepend an instance of `ChunkMeta` to the arguments.
    If None, prepends automatically if the first argument is named 'chunk_meta'.

    Each parameter can be modified in the `options` attribute of the wrapper function or
    directly passed as a keyword argument with a leading underscore.

    For defaults, see `chunking` in `vectorbt._settings.settings`.
    For example, to change the engine globally:

    ```python-repl
    >>> import vectorbt as vbt

    >>> vbt.settings.chunking['engine'] = 'dask'
    ```

    !!! note
        If less than two chunks were generated and `skip_one_chunk` is True,
        executes the function without chunking.

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
    [ChunkMeta(idx=0, start=None, end=None, indices=None),
     ChunkMeta(idx=1, start=None, end=None, indices=None)]
    ```

    Additionally, it may contain the start and end index of the space we want to split.
    The space can be defined by the length of an input array, for example. In our case:

    ```python-repl
    >>> list(yield_chunk_meta(n_chunks=2, size=10))
    [ChunkMeta(idx=0, start=0, end=5, indices=None),
     ChunkMeta(idx=1, start=5, end=10, indices=None)]
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
    ...     return a[chunk_meta.start:chunk_meta.end]

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
    ...     return a[chunk_meta.start:chunk_meta.end]

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
    ...     engine_kwargs=dict(show_progress=True))  # see SequenceEngine
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
            min_size = kwargs.pop('_min_size', wrapper.options['min_size'])
            chunk_len = kwargs.pop('_chunk_len', wrapper.options['chunk_len'])
            skip_one_chunk = kwargs.pop('_skip_one_chunk', wrapper.options['skip_one_chunk'])
            if skip_one_chunk is None:
                skip_one_chunk = chunking_cfg['skip_one_chunk']
            chunk_meta = kwargs.pop('_chunk_meta', wrapper.options['chunk_meta'])
            arg_take_spec = kwargs.pop('_arg_take_spec', wrapper.options['arg_take_spec'])
            template_mapping = merge_dicts(wrapper.options['template_mapping'], kwargs.pop('_template_mapping', {}))
            engine = kwargs.pop('_engine', wrapper.options['engine'])
            if engine is None:
                engine = chunking_cfg['engine']
            engine_kwargs = merge_dicts(wrapper.options['engine_kwargs'], kwargs.pop('_engine_kwargs', {}))
            merge_func = kwargs.pop('_merge_func', wrapper.options['merge_func'])
            merge_kwargs = merge_dicts(wrapper.options['merge_kwargs'], kwargs.pop('_merge_kwargs', {}))

            if prepend_chunk_meta:
                args = (Rep('chunk_meta'), *args)

            ann_args = annotate_args(func, *args, **kwargs)
            chunk_meta = list(get_chunk_meta_from_args(
                ann_args,
                n_chunks=n_chunks,
                size=size,
                min_size=min_size,
                chunk_len=chunk_len,
                chunk_meta=chunk_meta
            ))
            if len(chunk_meta) < 2 and skip_one_chunk:
                return func(*args, **kwargs)
            funcs_args = yield_arg_chunks(
                func,
                ann_args,
                chunk_meta=chunk_meta,
                arg_take_spec=arg_take_spec,
                template_mapping=template_mapping
            )
            mapping = merge_dicts(
                dict(
                    ann_args=ann_args,
                    chunk_meta=chunk_meta,
                    arg_take_spec=arg_take_spec,
                ),
                template_mapping
            )
            engine_kwargs = deep_substitute(engine_kwargs, mapping)
            results = execute(funcs_args, engine=engine, n_calls=len(chunk_meta), **engine_kwargs)
            if merge_func is not None:
                mapping['funcs_args'] = funcs_args
                merge_kwargs = deep_substitute(merge_kwargs, mapping)
                return merge_func(results, **merge_kwargs)
            return results

        wrapper.options = Config(
            dict(
                n_chunks=n_chunks,
                size=size,
                min_size=min_size,
                chunk_len=chunk_len,
                chunk_meta=chunk_meta,
                skip_one_chunk=skip_one_chunk,
                arg_take_spec=arg_take_spec,
                template_mapping=template_mapping,
                engine=engine,
                engine_kwargs=engine_kwargs,
                merge_func=merge_func,
                merge_kwargs=merge_kwargs
            ),
            frozen_keys=True
        )

        if prepend_chunk_meta:
            signature = inspect.signature(wrapper)
            wrapper.__signature__ = signature.replace(parameters=tuple(signature.parameters.values())[1:])

        return wrapper

    if len(args) == 0:
        return decorator
    elif len(args) == 1:
        return decorator(args[0])
    raise ValueError("Either function or keyword arguments must be passed")


def resolve_chunked(func: tp.Callable, option: tp.ChunkedOption, **kwargs) -> tp.Callable:
    """Decorate with `chunked` based on an option.

    `option` can be:

    * True: Decorate with default values
    * False: Return `func` without chunking
    * string: Use `option` as the name of an engine
    * dict: Use `option` as keyword arguments passed to `chunked`

    For defaults, see `chunking.option` in `vectorbt._settings.settings`."""
    from vectorbt._settings import settings
    chunking_cfg = settings['chunking']

    if option is None:
        option = chunking_cfg['option']

    if option is not None and not (isinstance(option, bool) and not option):
        if isinstance(option, dict):
            chunk_kwargs = option
        elif isinstance(option, str):
            chunk_kwargs = dict(engine=option)
        elif isinstance(option, bool):
            chunk_kwargs = dict()
        else:
            raise TypeError(f"Type {type(option)} is not supported for chunking")
        if isinstance(chunking_cfg['option'], dict):
            chunk_kwargs = merge_dicts(chunking_cfg['option'], kwargs, chunk_kwargs)
        else:
            chunk_kwargs = merge_dicts(kwargs, chunk_kwargs)
        return chunked(func, **chunk_kwargs)
    return func
