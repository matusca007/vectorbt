# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""Global registry for chunkable functions."""

from vectorbt import _typing as tp
from vectorbt.utils import checks
from vectorbt.utils.config import merge_dicts
from vectorbt.utils.template import RepEval
from vectorbt.utils.chunking import chunked, resolve_chunked

__pdoc__ = {}


class ChunkableRegistry:
    """Class that registers chunkable functions."""

    def __init__(self) -> None:
        self._setups = {}

    @property
    def setups(self) -> tp.Dict[tp.Hashable, tp.Kwargs]:
        """Dictionary of registered setups by their id."""
        return self._setups

    def register(self,
                 func: tp.Callable,
                 setup_id: tp.Optional[tp.Hashable] = None,
                 options: tp.KwargsLike = None,
                 tags: tp.Optional[set] = None) -> None:
        """Register a new setup."""
        if setup_id is None:
            setup_id = func.__module__ + '.' + func.__name__
        if setup_id in self.setups:
            raise ValueError(f"Setup id '{str(setup_id)}' already registered")
        if options is None:
            options = {}
        if tags is None:
            tags = set()
        else:
            tags = set(tags)

        setup = dict(
            func=func,
            options=options,
            tags=tags
        )
        self.setups[setup_id] = setup

    def match_setups(self, expression: str, mapping: tp.KwargsLike = None) -> tp.List[tp.Kwargs]:
        """Match each setup by evaluating an expression with the setup being a mapping."""
        matched_setups = []
        for setup_id, setup in self.setups.items():
            result = RepEval(expression).substitute(mapping=merge_dicts(setup, mapping))
            checks.assert_instance_of(result, bool)

            if result:
                matched_setups.append(setup)
        return matched_setups

    def get_setup(self, setup_id_or_func: tp.Union[tp.Hashable, tp.Callable]) -> tp.Kwargs:
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
            func = setup['func']
        return chunked(func, **merge_dicts(setup['options'], kwargs))

    def resolve_chunked(self,
                        setup_id_or_func: tp.Union[tp.Hashable, tp.Callable],
                        option: tp.ChunkedOption = None,
                        target_func: tp.Optional[tp.Callable] = None,
                        **kwargs) -> tp.Callable:
        """Same as `ChunkableRegistry.decorate` but using `vectorbt.utils.chunking.resolve_chunked`."""
        setup = self.get_setup(setup_id_or_func)

        if target_func is not None:
            func = target_func
        elif callable(setup_id_or_func):
            func = setup_id_or_func
        else:
            func = setup['func']
        return resolve_chunked(func, option=option, **merge_dicts(setup['options'], kwargs))


ch_registry = ChunkableRegistry()
"""Registry of type `ChunkableRegistry`."""


def register_chunkable(func: tp.Optional[tp.Callable] = None,
                       setup_id: tp.Optional[tp.Hashable] = None,
                       registry: ChunkableRegistry = ch_registry,
                       tags: tp.Optional[set] = None,
                       wrap: bool = False,
                       **options) -> tp.Callable:
    """Register a new chunkable function.

    If `wrap` is True, wraps with the `vectorbt.utils.chunking.chunked` decorator.
    Otherwise, leaves the function as-is (preferred).

    !!! note
        Calling the `register_chunkable` decorator before (or below) the `vectorbt.nb_registry.register_jit`
        decorator with `wrap` set to True won't work. Doing the same after (or above)
        `vectorbt.nb_registry.register_jit` will work for calling the function from Python but not from Numba.
        Generally, avoid wrapping right away and use `ChunkableRegistry.decorate` to perform decoration."""

    def decorator(_func: tp.Callable) -> tp.Callable:
        registry.register(
            func=_func,
            setup_id=setup_id,
            options=options,
            tags=tags
        )
        if wrap:
            return chunked(_func, **options)
        return _func

    if func is None:
        return decorator
    return decorator(func)
