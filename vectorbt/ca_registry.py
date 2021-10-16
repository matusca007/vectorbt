# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""Global registry for cacheable callables."""

import inspect

from vectorbt import _typing as tp
from vectorbt.utils import checks
from vectorbt.utils.docs import SafeToStr
from vectorbt.utils.decorators import cacheableT, cacheable_property, cachedproperty
from vectorbt.utils.parsing import hash_args
from vectorbt.utils.hashing import Hashable


class CASetup(Hashable, SafeToStr):
    """Class that represents a setup of a cacheable callable."""

    def __init__(self,
                 cacheable: cacheableT,
                 instance: tp.Optional[object] = None,
                 args: tp.Optional[tp.Args] = None,
                 kwargs: tp.Optional[tp.Kwargs] = None) -> None:
        self._cacheable = cacheable
        self._instance = instance
        if args is None:
            args = ()
        self._args = args
        if kwargs is None:
            kwargs = {}
        self._kwargs = kwargs

    @property
    def cacheable(self) -> cacheableT:
        """Cacheable callable.

        Must be either instance of `vectorbt.utils.decorators.cacheable_property`
        or `vectorbt.utils.decorators.cacheable_method`."""
        return self._cacheable

    @property
    def instance(self) -> tp.Optional[object]:
        """Class instance `CASetup.cacheable` is bound to."""
        return self._instance

    @property
    def args(self) -> tp.Args:
        """Arguments passed to `CASetup.cacheable`."""
        return self._args

    @property
    def kwargs(self) -> tp.Kwargs:
        """Keyword arguments passed to `CASetup.cacheable`."""
        return self._kwargs

    def run_func(self) -> tp.Any:
        """Run the function."""
        if self.instance is not None:
            return self.cacheable.func(self.instance, *self.args, **self.kwargs)
        return self.cacheable.func(*self.args, **self.kwargs)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(" \
               f"cacheable={self.cacheable}, " \
               f"instance={self.instance}, " \
               f"args={self.args}, " \
               f"kwargs={self.kwargs})"

    @property
    def hash_key(self) -> tuple:
        return (
            self.cacheable,
            id(self.instance) if self.instance is not None else None,
            self.args,
            tuple(self.kwargs.items())
        )

    @cachedproperty
    def hash(self) -> int:
        return hash_args(
            self.cacheable.func,
            self.args if self.instance is None else (id(self.instance), *self.args),
            self.kwargs,
            ignore_args=self.cacheable.ignore_args
        )


class CAQuery(Hashable, SafeToStr):
    """Class that represents a query for matching and ranking setups."""

    def __init__(self,
                 cacheable: tp.Optional[tp.Union[tp.Callable, cacheableT, str]] = None,
                 instance: tp.Optional[object] = None,
                 cls: tp.Optional[tp.Union[type, str]] = None,
                 base_cls: tp.Optional[tp.TypeLike] = None,
                 options: tp.Optional[dict] = None) -> None:
        self._cacheable = cacheable
        self._instance = instance
        self._cls = cls
        self._base_cls = base_cls
        self._options = options

    @property
    def cacheable(self) -> tp.Optional[tp.Union[tp.Callable, cacheableT, str]]:
        """Cacheable callable or its name (case-sensitive)."""
        return self._cacheable

    @property
    def instance(self) -> tp.Optional[object]:
        """Class instance `CAQuery.cacheable` is bound to."""
        return self._instance

    @property
    def cls(self) -> tp.Optional[tp.Union[type, str]]:
        """Class of the instance or its name (case-sensitive) `CAQuery.cacheable` is bound to."""
        return self._cls

    @property
    def base_cls(self) -> tp.Optional[tp.TypeLike]:
        """Base class of the instance or its name (case-sensitive) `CAQuery.cacheable` is bound to."""
        return self._base_cls

    @property
    def options(self) -> tp.Optional[dict]:
        """Options to match."""
        return self._options

    def matches_setup(self, setup: CASetup) -> bool:
        """Return whether a setup of type `CASetup` matches this query.

        ## Example

        Let's evaluate various queries:

        ```python-repl
        >>> import vectorbt as vbt

        >>> class A:
        ...     @vbt.cached_property(my_option=True)
        ...     def f(self):
        ...         return None

        >>> class B(A):
        ...     @vbt.cached_method(my_option=False)
        ...     def f(self):
        ...         return None

        >>> a = A()
        >>> b = B()

        >>> def match_query(query):
        ...     matched = []
        ...     if query.matches_setup(vbt.CASetup(A.f, a)):
        ...         matched.append('a')
        ...     if query.matches_setup(vbt.CASetup(B.f, b)):
        ...         matched.append('b')
        ...     return matched

        >>> match_query(vbt.CAQuery(instance=a, cacheable='f'))
        ['a']
        >>> match_query(vbt.CAQuery(instance=b, cacheable='f'))
        ['b']
        >>> match_query(vbt.CAQuery(instance=a, options=dict(my_option=True)))
        ['a']
        >>> match_query(vbt.CAQuery(instance=a, options=dict(my_option=False)))
        []
        >>> match_query(vbt.CAQuery(instance=b, options=dict(my_option=False)))
        ['b']
        >>> match_query(vbt.CAQuery(instance=a))
        ['a']
        >>> match_query(vbt.CAQuery(instance=b))
        ['b']
        >>> match_query(vbt.CAQuery(cls=A))
        ['a']
        >>> match_query(vbt.CAQuery(cls=B))
        ['b']
        >>> match_query(vbt.CAQuery(base_cls=A))
        ['a', 'b']
        >>> match_query(vbt.CAQuery(base_cls=B))
        ['b']
        >>> match_query(vbt.CAQuery(base_cls=A, options=dict(my_option=False)))
        ['b']
        >>> match_query(vbt.CAQuery(cacheable=A.f))
        ['a']
        >>> match_query(vbt.CAQuery(cacheable=B.f))
        ['b']
        >>> match_query(vbt.CAQuery(cacheable=b.f))
        ['b']
        >>> match_query(vbt.CAQuery(cacheable='f'))
        ['a', 'b']
        >>> match_query(vbt.CAQuery(cacheable='f', options=dict(my_option=False)))
        ['b']
        >>> match_query(vbt.CAQuery(options=dict(my_option=True)))
        ['a']
        >>> match_query(vbt.CAQuery())
        ['a', 'b']
        ```"""
        if self.cacheable is not None:
            if isinstance(self.cacheable, cacheable_property):
                if setup.cacheable is not self.cacheable and setup.cacheable.func is not self.cacheable.func:
                    return False
            elif isinstance(self.cacheable, property):
                raise TypeError("Only cacheable properties are supported")
            elif callable(self.cacheable) and \
                    hasattr(self.cacheable, 'is_cacheable') and \
                    self.cacheable.is_cacheable:
                if setup.cacheable is not self.cacheable and setup.cacheable.func is not self.cacheable.func:
                    return False
            elif callable(self.cacheable) and \
                    hasattr(self.cacheable, 'is_cacheable') and \
                    not self.cacheable.is_cacheable:
                raise TypeError("Only cacheable functions are supported")
            elif callable(self.cacheable):
                if setup.cacheable.func is not self.cacheable:
                    return False
            elif isinstance(self.cacheable, str):
                if setup.cacheable.name != self.cacheable:
                    return False
            else:
                raise TypeError(f"cacheable must be either a callable or a string, not {type(self.cacheable)}")
        if self.instance is not None:
            if setup.instance is not self.instance:
                return False
        if self.cls is not None:
            if inspect.isclass(self.cls):
                if type(setup.instance) is not self.cls:
                    return False
            elif isinstance(self.cls, str):
                if type(setup.instance).__name__ != self.cls:
                    return False
            else:
                raise TypeError(f"cls must be either a class or a string, not {type(self.cls)}")
        if self.base_cls is not None:
            if not checks.is_instance_of(setup.instance, self.base_cls):
                return False
        if self.options is not None:
            if not isinstance(self.options, dict):
                raise TypeError("options must be a dict")
            for k, v in self.options.items():
                if k not in setup.cacheable.options or setup.cacheable.options[k] != v:
                    return False
        return True

    def rank_setup(self, setup: CASetup) -> tp.Optional[int]:
        """Rank a setup of type `CASetup`.

        The rank is based on how narrow or broad the query is: a narrower query has a lower rank
        than a broader query. Lower means more important:

        0) `instance` and `cacheable`
        1) `instance` and `options`
        2) `instance`
        3) `cls` and `cacheable`
        4) `cls` and `options`
        5) `cls`
        6) `base_cls` and `cacheable`
        7) `base_cls` and `options`
        8) `base_cls`
        9) `cacheable` and `options`
        10) `cacheable`
        11) `options`

        Returns None if the setup could not be matched."""
        if not self.matches_setup(setup):
            return None

        if self.instance is not None and self.cacheable is not None:
            return 0
        if self.instance is not None and self.options is not None:
            return 1
        if self.instance is not None:
            return 2

        if self.cls is not None and self.cacheable is not None:
            return 3
        if self.cls is not None and self.options is not None:
            return 4
        if self.cls is not None:
            return 5

        if self.base_cls is not None and self.cacheable is not None:
            return 6
        if self.base_cls is not None and self.options is not None:
            return 7
        if self.base_cls is not None:
            return 8

        if self.cacheable is not None and self.options is not None:
            return 9
        if self.cacheable is not None:
            return 10
        if self.options is not None:
            return 11

        return 12

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(" \
               f"cacheable={self.cacheable}, " \
               f"instance={self.instance}, " \
               f"cls={self.cls}, " \
               f"base_cls={self.base_cls}, " \
               f"options={self.options})"

    @property
    def hash_key(self) -> tuple:
        return (
            self.cacheable,
            id(self.instance) if self.instance is not None else None,
            self.cls,
            self.base_cls,
            tuple(self.options.items()) if self.options is not None else None
        )


class CADirective(Hashable, SafeToStr):
    """Class that represents a caching directive."""

    def __init__(self,
                 query: CAQuery,
                 cache: tp.Optional[bool] = None,
                 override_disabled: bool = False,
                 rank: tp.Optional[int] = None) -> None:
        self._query = query
        self._cache = cache
        self._override_disabled = override_disabled
        self._rank = rank

    @property
    def query(self) -> CAQuery:
        """See `CAQuery`."""
        return self._query

    @property
    def cache(self) -> tp.Optional[bool]:
        """Whether to enable caching in any cacheable method and property that matches this directive.

        If not None, overrides the local `cache` property."""
        return self._cache

    @property
    def override_disabled(self) -> bool:
        """Whether to override `enabled` set to False under `caching` in `vectorbt._settings.settings`."""
        return self._override_disabled

    @property
    def rank(self) -> tp.Optional[int]:
        """Rank to override the default rank."""
        return self._rank

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(" \
               f"query={str(self.query)}, " \
               f"cache={self.cache}, " \
               f"override_disabled={self.override_disabled}, " \
               f"rank={self.rank})"

    @property
    def hash_key(self) -> tuple:
        return (
            self.query,
            self.cache,
            self.override_disabled,
            self.rank
        )


class CacheableRegistry:
    """Class that registers cacheable callables."""

    def __init__(self) -> None:
        self._setup_cache = dict()

    @property
    def directives(self) -> tp.List[CADirective]:
        """Get the list of directives from `settings.caching` under `vectorbt._settings.settings`."""
        from vectorbt._settings import settings
        caching_cfg = settings['caching']

        return caching_cfg['directives']

    def add_directive(self, directive: CADirective) -> None:
        """Add a new directive to `CacheableRegistry.directives`.

        !!! note
            Will raise an error if there is already a directive with an identical query."""
        for c in self.directives:
            if directive.query == c.query:
                raise ValueError(f"Query '{str(c.query)}' already registered")
        self.directives.append(directive)

    def remove_directive(self, directive_or_query: tp.Union[CADirective, CAQuery]) -> None:
        """Remove a directive from `CacheableRegistry.directives` given its instance or query."""
        if isinstance(directive_or_query, CADirective):
            query = directive_or_query.query
        else:
            query = directive_or_query
        found_i = None
        for i, c in enumerate(self.directives):
            if query == c.query:
                found_i = i
                break
        if found_i is None:
            raise ValueError(f"Query '{str(query)}' not registered")
        del self.directives[found_i]

    @property
    def setup_cache(self) -> tp.Dict[CASetup, tp.Any]:
        """Results of running each setup."""
        return self._setup_cache

    def should_cache_setup(self, setup: CASetup) -> bool:
        """Whether should cache a setup."""
        from vectorbt._settings import settings
        caching_cfg = settings['caching']

        ranked_directives = []
        for directive in self.directives:
            rank = directive.query.rank_setup(setup)
            if rank is not None:
                ranked_directives.append((rank, directive))
        ranked_directives = sorted(ranked_directives, key=lambda x: x[0], reverse=True)
        cache = None
        for _, directive in ranked_directives:
            if directive.cache is not None:
                if caching_cfg['enabled'] or directive.override_disabled:
                    cache = directive.cache
        if cache is None:
            if not caching_cfg['enabled']:
                return False
            return setup.cacheable.cache
        return cache

    def run_setup(self, setup: CASetup) -> tp.Any:
        """Run a setup and cache it, or pull it from the cache if it has already been cached."""
        if checks.is_hashable(setup):
            if self.should_cache_setup(setup):
                if setup not in self.setup_cache:
                    self.setup_cache[setup] = setup.run_func()
                return self.setup_cache[setup]
        return setup.run_func()

    def clear_cache(self, directive_or_query: tp.Optional[tp.Union[CADirective, CAQuery]] = None) -> None:
        """Clear all setups.

        If `directive_or_query` is not None, clears only those that can be matched."""
        if directive_or_query is None:
            self.setup_cache.clear()
        else:
            if isinstance(directive_or_query, CADirective):
                query = directive_or_query.query
            else:
                query = directive_or_query
            for setup in list(self.setup_cache.keys()):
                if query.matches_setup(setup):
                    del self.setup_cache[setup]


ca_registry = CacheableRegistry()
"""Default registry of type `CacheableRegistry`."""
