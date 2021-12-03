# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""Global registry for chunkable functions."""

from vectorbt import _typing as tp
from vectorbt.utils import checks
from vectorbt.utils.chunking import chunked, resolve_chunked
from vectorbt.utils.config import merge_dicts
from vectorbt.utils.docs import SafeToStr, prepare_for_doc
from vectorbt.utils.hashing import Hashable
from vectorbt.utils.template import RepEval


class ChunkedSetup(Hashable, SafeToStr):
    """Class that represents a chunkable setup.

    !!! note
        Hashed solely by `setup_id`."""

    @staticmethod
    def get_hash(setup_id: tp.Hashable) -> int:
        return hash((setup_id,))

    def __init__(self,
                 setup_id: tp.Hashable,
                 func: tp.Callable,
                 options: tp.DictLike = None,
                 tags: tp.SetLike = None) -> None:
        if options is None:
            options = {}
        if tags is None:
            tags = set()

        self._setup_id = setup_id
        self._func = func
        self._options = options
        self._tags = tags

    @property
    def setup_id(self) -> tp.Hashable:
        """Setup id."""
        return self._setup_id

    @property
    def func(self) -> tp.Callable:
        """Chunkable function."""
        return self._func

    @property
    def options(self) -> tp.DictLike:
        """Options dictionary."""
        return self._options

    @property
    def tags(self) -> set:
        """Set of tags."""
        return self._tags

    def to_dict(self) -> dict:
        """Convert this instance to a dict."""
        return dict(
            setup_id=self.setup_id,
            func=self.func,
            options=self.options,
            tags=self.tags
        )

    def __str__(self) -> str:
        return f"{type(self).__name__}(" \
               f"setup_id={self.setup_id}, " \
               f"func={self.func}, " \
               f"options={prepare_for_doc(self.options)}, " \
               f"tags={self.tags})"

    @property
    def hash_key(self) -> tuple:
        return (self.setup_id,)


class ChunkableRegistry:
    """Class that registers chunkable functions."""

    def __init__(self) -> None:
        self._setups = {}

    @property
    def setups(self) -> tp.Dict[tp.Hashable, ChunkedSetup]:
        """Dict of registered `ChunkedSetup` instances by `ChunkedSetup.setup_id`."""
        return self._setups

    def register(self,
                 func: tp.Callable,
                 setup_id: tp.Optional[tp.Hashable] = None,
                 options: tp.DictLike = None,
                 tags: tp.SetLike = None) -> None:
        """Register a new setup."""
        if setup_id is None:
            setup_id = func.__module__ + '.' + func.__name__
        if setup_id in self.setups:
            raise ValueError(f"Setup id '{str(setup_id)}' already registered")

        setup = ChunkedSetup(
            setup_id=setup_id,
            func=func,
            options=options,
            tags=tags
        )
        self.setups[setup_id] = setup

    def match_setups(self, expression: tp.Optional[str] = None,
                     mapping: tp.KwargsLike = None) -> tp.Set[ChunkedSetup]:
        """Match setups against an expression with each setup being a mapping."""
        matched_setups = set()
        for setup in self.setups.values():
            if expression is None:
                result = True
            else:
                result = RepEval(expression).substitute(mapping=merge_dicts(setup.to_dict(), mapping))
                checks.assert_instance_of(result, bool)

            if result:
                matched_setups.add(setup)
        return matched_setups

    def get_setup(self, setup_id_or_func: tp.Union[tp.Hashable, tp.Callable]) -> ChunkedSetup:
        """Get setup by its id or function.

        `setup_id_or_func` can be an identifier or a function.
        If it's a function, will build the identifier using its module and name."""
        if hasattr(setup_id_or_func, 'py_func'):
            nb_setup_id = setup_id_or_func.__module__ + '.' + setup_id_or_func.__name__
            if nb_setup_id in self.setups:
                setup_id = nb_setup_id
            else:
                setup_id = setup_id_or_func.py_func.__module__ + '.' + setup_id_or_func.py_func.__name__
        elif callable(setup_id_or_func):
            setup_id = setup_id_or_func.__module__ + '.' + setup_id_or_func.__name__
        else:
            setup_id = setup_id_or_func
        if setup_id not in self.setups:
            raise KeyError(f"Setup id '{str(setup_id)}' not registered")
        return self.setups[setup_id]

    def decorate(self,
                 setup_id_or_func: tp.Union[tp.Hashable, tp.Callable],
                 target_func: tp.Optional[tp.Callable] = None,
                 **kwargs) -> tp.Callable:
        """Decorate the setup's function using the `vectorbt.utils.chunking.chunked` decorator.

        Finds setup using `ChunkableRegistry.get_setup`.

        Merges setup's options with `options`.

        Specify `target_func` to apply the found setup on another function."""
        setup = self.get_setup(setup_id_or_func)

        if target_func is not None:
            func = target_func
        elif callable(setup_id_or_func):
            func = setup_id_or_func
        else:
            func = setup.func
        return chunked(func, **merge_dicts(setup.options, kwargs))

    def resolve_option(self,
                       setup_id_or_func: tp.Union[tp.Hashable, tp.Callable],
                       option: tp.ChunkedOption,
                       target_func: tp.Optional[tp.Callable] = None,
                       **kwargs) -> tp.Callable:
        """Same as `ChunkableRegistry.decorate` but using `vectorbt.utils.chunking.resolve_chunked`."""
        setup = self.get_setup(setup_id_or_func)

        if target_func is not None:
            func = target_func
        elif callable(setup_id_or_func):
            func = setup_id_or_func
        else:
            func = setup.func
        return resolve_chunked(func, option=option, **merge_dicts(setup.options, kwargs))


ch_registry = ChunkableRegistry()
"""Default registry of type `ChunkableRegistry`."""


def register_chunkable(func: tp.Optional[tp.Callable] = None,
                       setup_id: tp.Optional[tp.Hashable] = None,
                       registry: ChunkableRegistry = ch_registry,
                       tags: tp.SetLike = None,
                       return_wrapped: bool = False,
                       **options) -> tp.Callable:
    """Register a new chunkable function.

    If `return_wrapped` is True, wraps with the `vectorbt.utils.chunking.chunked` decorator.
    Otherwise, leaves the function as-is (preferred).

    Options are merged in the following order:

    * `chunking.options` in `vectorbt._settings.settings`
    * `chunking.setup_options.{setup_id}` in `vectorbt._settings.settings`
    * `options`
    * `chunking.override_options` in `vectorbt._settings.settings`
    * `chunking.override_setup_options.{setup_id}` in `vectorbt._settings.settings`

    !!! note
        Calling the `register_chunkable` decorator before (or below) the `vectorbt.jit_registry.register_jitted`
        decorator with `return_wrapped` set to True won't work. Doing the same after (or above)
        `vectorbt.jit_registry.register_jitted` will work for calling the function from Python but not from Numba.
        Generally, avoid wrapping right away and use `ChunkableRegistry.decorate` to perform decoration."""

    def decorator(_func: tp.Callable) -> tp.Callable:
        nonlocal setup_id, options

        from vectorbt._settings import settings
        chunking_cfg = settings['chunking']

        if setup_id is None:
            setup_id = _func.__module__ + '.' + _func.__name__
        options = merge_dicts(
            chunking_cfg.get('options', None),
            chunking_cfg.get('setup_options', {}).get(setup_id, None),
            options,
            chunking_cfg.get('override_options', None),
            chunking_cfg.get('override_setup_options', {}).get(setup_id, None)
        )

        registry.register(
            func=_func,
            setup_id=setup_id,
            options=options,
            tags=tags
        )
        if return_wrapped:
            return chunked(_func, **options)
        return _func

    if func is None:
        return decorator
    return decorator(func)
