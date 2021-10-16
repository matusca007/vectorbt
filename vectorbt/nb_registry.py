# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""Global registry for Numba-compiled functions."""

import warnings
from numba import jit, generated_jit
from functools import partial

from vectorbt import _typing as tp
from vectorbt.utils import checks
from vectorbt.utils.config import merge_dicts
from vectorbt.utils.template import RepEval


class NumbaRegistry:
    """Class that registers Numba-compiled functions."""

    def __init__(self) -> None:
        self._setups = {}

    @property
    def setups(self) -> tp.Dict[tp.Hashable, tp.Kwargs]:
        """Dictionary of registered setups by their id."""
        return self._setups

    def register(self,
                 py_func: tp.Callable,
                 nb_func: tp.Callable,
                 setup_id: tp.Optional[tp.Hashable] = None,
                 nb_decorator: tp.Callable = jit,
                 options: tp.KwargsLike = None,
                 tags: tp.Optional[set] = None) -> None:
        """Register a new setup."""
        checks.assert_numba_func(nb_func)

        if setup_id is None:
            setup_id = py_func.__module__ + '.' + py_func.__name__
        if setup_id in self.setups:
            raise ValueError(f"Setup id '{str(setup_id)}' already registered")
        if options is None:
            options = {}
        if tags is None:
            tags = set()
        else:
            tags = set(tags)
        if 'parallel' in options:
            tags.add('can_parallel')

        setup = dict(
            py_func=py_func,
            nb_func=nb_func,
            nb_decorator=nb_decorator,
            options=options,
            tags=tags
        )
        self.setups[setup_id] = setup

    def match_setups(self, expression: str, mapping: tp.KwargsLike = None) -> tp.List[tp.Kwargs]:
        """Match each setup by evaluating an expression with the setup being a mapping.

        For example, to get all setups with the tag 'can_parallel', pass `expression="'can_parallel' in tags"`."""
        matched_setups = []
        for setup_id, setup in self.setups.items():
            result = RepEval(expression).substitute(mapping=merge_dicts(setup, mapping))
            checks.assert_instance_of(result, bool)

            if result:
                matched_setups.append(setup)
        return matched_setups

    def redecorate(self,
                   setup_id_or_func: tp.Union[tp.Hashable, tp.Callable],
                   new_setup_id: tp.Optional[tp.Hashable] = None,
                   options: tp.KwargsLike = None,
                   register: bool = True,
                   tags: tp.Optional[set] = None,
                   union_tags: bool = True) -> tp.Callable:
        """Redecorate the setup's Numba-compiled function.

        Merges setup's options with `options`. Extends tags as well if `union_tags` is True.

        `setup_id_or_func` can be an identifier or a function.
        If its a function, will build the identifier using its module and name."""
        if callable(setup_id_or_func):
            setup_id = setup_id_or_func.__module__ + '.' + setup_id_or_func.__name__
        else:
            setup_id = setup_id_or_func
        setup = self.setups[setup_id]
        if tags is None:
            tags = set()

        return decorate_py_func(
            py_func=setup['py_func'],
            nb_decorator=setup['nb_decorator'],
            options=merge_dicts(setup['options'], dict(cache=False), options),
            register=register,
            setup_id=new_setup_id,
            nb_registry=self,
            tags=setup['tags'] | tags if union_tags else tags
        )

    def redecorate_parallel(self,
                            setup_id_or_func: tp.Union[tp.Hashable, tp.Callable],
                            parallel: tp.Optional[bool] = None,
                            new_setup_id: tp.Optional[tp.Hashable] = None,
                            silence_warnings: tp.Optional[bool] = None,
                            **kwargs) -> tp.Callable:
        """Redecorate the setup's Numba-compiled function with the `parallel` option.

        The behavior depends upon `parallel`:

        * None: Returns the original function.
        * True: Appends the `parallel=True` option. Requires the setup to have 'can_parallel' tag.
        * False: Appends the `parallel=False` option.
        """
        if callable(setup_id_or_func):
            setup_id = setup_id_or_func.__module__ + '.' + setup_id_or_func.__name__
        else:
            setup_id = setup_id_or_func
        setup = self.setups[setup_id]

        from vectorbt._settings import settings
        numba_cfg = settings['numba']

        if parallel is None:
            parallel = numba_cfg['parallel']
        if silence_warnings is None:
            silence_warnings = numba_cfg['silence_warnings']
        if parallel is None:
            return setup['nb_func']
        checks.assert_instance_of(parallel, bool)
        if parallel:
            if setup['options'].get('parallel', False):
                return setup['nb_func']
            if new_setup_id is None:
                new_setup_id = (setup_id, 'parallel')
            if new_setup_id in self.setups:
                return self.setups[new_setup_id]['nb_func']
            if 'can_parallel' not in setup['tags']:
                if not silence_warnings:
                    warnings.warn("Function has no 'can_parallel' tag", stacklevel=2)
            return self.redecorate(
                setup_id,
                new_setup_id=new_setup_id,
                options=dict(parallel=True),
                **kwargs
            )
        else:
            if not setup['options'].get('parallel', False):
                return setup['nb_func']
            if new_setup_id is None:
                if isinstance(setup_id, tuple) \
                        and len(setup_id) == 2 \
                        and isinstance(setup_id, str) \
                        and setup_id[1] == 'parallel':
                    new_setup_id = setup_id[0]
                else:
                    new_setup_id = (setup_id, 'not_parallel')
            if new_setup_id in self.setups:
                return self.setups[new_setup_id]['nb_func']
            return self.redecorate(
                setup_id,
                new_setup_id=new_setup_id,
                options=dict(parallel=False),
                **kwargs
            )


nb_registry = NumbaRegistry()
"""Default registry of type `NumbaRegistry`."""


def decorate_py_func(py_func: tp.Optional[tp.Callable] = None,
                     nb_decorator: tp.Callable = jit,
                     options: tp.KwargsLike = None,
                     register: bool = True,
                     setup_id: tp.Optional[tp.Hashable] = None,
                     nb_registry: NumbaRegistry = nb_registry,
                     tags: tp.Optional[set] = None) -> tp.Callable:
    """Decorate a Python function using Numba and (optionally) register it."""
    nb_func = nb_decorator(**options)(py_func)
    if register:
        nb_registry.register(
            py_func=py_func,
            nb_func=nb_func,
            setup_id=setup_id,
            nb_decorator=nb_decorator,
            options=options,
            tags=tags
        )
    return nb_func


def register_jit(py_func: tp.Optional[tp.Callable] = None,
                 register: bool = True,
                 setup_id: tp.Optional[tp.Hashable] = None,
                 nb_registry: NumbaRegistry = nb_registry,
                 tags: tp.Optional[set] = None,
                 _nb_decorator: tp.Callable = jit,
                 **options) -> tp.Callable:
    """Pass keyword arguments to `numba.jit`, wrap `py_func`, and register it.
    
    Options are merged in the following order: 
    
    * `numba.options` in `vectorbt._settings.settings`
    * `**options`
    * `numba.setup_options` in `vectorbt._settings.settings` with key `setup_id`
    * `numba.override_options` in `vectorbt._settings.settings`"""

    def decorator(_py_func: tp.Callable) -> tp.Callable:
        nonlocal setup_id, options

        from vectorbt._settings import settings
        numba_options_cfg = settings['numba']['options']
        numba_setup_options_cfg = settings['numba']['setup_options']
        numba_override_options_cfg = settings['numba']['override_options']

        if setup_id is None:
            setup_id = _py_func.__module__ + '.' + _py_func.__name__
        setup_options = numba_setup_options_cfg.get(setup_id, {})
        options = merge_dicts(numba_options_cfg, options, setup_options, numba_override_options_cfg)

        return decorate_py_func(
            py_func=_py_func,
            nb_decorator=_nb_decorator,
            options=options,
            register=register,
            setup_id=setup_id,
            nb_registry=nb_registry,
            tags=tags
        )

    if py_func is None:
        return decorator
    return decorator(py_func)


register_generated_jit = partial(register_jit, _nb_decorator=generated_jit)
"""Pass keyword arguments to `numba.generated_jit`, wrap `py_func`, and register it."""
