# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""Global registry for cacheables."""

import inspect

from vectorbt import _typing as tp
from vectorbt.utils import checks
from vectorbt.utils.docs import SafeToStr
from vectorbt.utils.decorators import cacheableT, cacheable_property
from vectorbt.utils.parsing import Regex, hash_args
from vectorbt.utils.hashing import Hashable

CAQueryT = tp.TypeVar("CAQueryT", bound="CAQuery")


class CAQuery(Hashable, SafeToStr):
    """Class that represents a query for matching and ranking setups."""

    def __init__(self,
                 cacheable: tp.Optional[tp.Union[tp.Callable, cacheableT, str]] = None,
                 instance: tp.Optional[object] = None,
                 cls: tp.Optional[tp.TypeLike] = None,
                 base_cls: tp.Optional[tp.TypeLike] = None,
                 options: tp.Optional[dict] = None) -> None:
        self._cacheable = cacheable
        self._instance = instance
        self._cls = cls
        self._base_cls = base_cls
        self._options = options

    @classmethod
    def parse(cls: tp.Type[CAQueryT], query_like: tp.Any, use_base_cls: bool = True) -> CAQueryT:
        """Parse a query-like object.

        !!! note
            Not all attribute combinations can be safely parsed by this function.
            For example, you cannot combine cacheable together with options.

        ## Example

        ```python-repl
        >>> import vectorbt as vbt

        >>> vbt.CAQuery.parse(func)
        <=> vbt.CAQuery(cacheable=func)

        >>> vbt.CAQuery.parse("a")
        <=> vbt.CAQuery(cacheable='a')

        >>> vbt.CAQuery.parse("A.a")
        <=> vbt.CAQuery(base_cls='A', cacheable='a')

        >>> vbt.CAQuery.parse("A")
        <=> vbt.CAQuery(base_cls='A')

        >>> vbt.CAQuery.parse("A", use_base_cls=False)
        <=> vbt.CAQuery(cls='A')

        >>> vbt.CAQuery.parse(vbt.Regex("[A-B]"))
        <=> vbt.CAQuery(base_cls=vbt.Regex("[A-B]"))

        >>> vbt.CAQuery.parse(dict(my_option=100))
        <=> vbt.CAQuery(options=dict(my_option=100))

        >>> vbt.CAQuery.parse(123)
        <=> vbt.CAQuery(instance=123)
        ```"""
        if query_like is None:
            return CAQuery()
        if isinstance(query_like, CASetup):
            return cls(
                cacheable=query_like.cacheable,
                instance=query_like.instance
            )
        if isinstance(query_like, cacheable_property):
            return cls(cacheable=query_like)
        if inspect.isfunction(query_like) or inspect.ismethod(query_like):
            return cls(cacheable=query_like)
        if isinstance(query_like, str) and query_like[0].islower():
            return cls(cacheable=query_like)
        if isinstance(query_like, str) and query_like[0].isupper() and '.' in query_like:
            if use_base_cls:
                return cls(cacheable=query_like.split('.')[1], base_cls=query_like.split('.')[0])
            return cls(cacheable=query_like.split('.')[1], cls=query_like.split('.')[0])
        if isinstance(query_like, str) and query_like[0].isupper():
            if use_base_cls:
                return cls(base_cls=query_like)
            return cls(cls=query_like)
        if isinstance(query_like, Regex):
            if use_base_cls:
                return cls(base_cls=query_like)
            return cls(cls=query_like)
        if isinstance(query_like, type):
            if use_base_cls:
                return cls(base_cls=query_like)
            return cls(cls=query_like)
        if isinstance(query_like, tuple):
            if use_base_cls:
                return cls(base_cls=query_like)
            return cls(cls=query_like)
        if isinstance(query_like, dict):
            return cls(options=query_like)
        return cls(instance=query_like)

    @property
    def cacheable(self) -> tp.Optional[tp.Union[tp.Callable, cacheableT, str]]:
        """Cacheable callable or its name (case-sensitive)."""
        return self._cacheable

    @property
    def instance(self) -> tp.Optional[object]:
        """Class instance `CAQuery.cacheable` is bound to."""
        return self._instance

    @property
    def cls(self) -> tp.Optional[tp.TypeLike]:
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

    def rank_setup(self, setup: 'CASetup') -> tp.Optional[int]:
        """Rank a setup of type `CASetup`.

        The rank is based on how narrow or broad the query is: a narrower query has a higher
        (= more important) rank than a broader query. If the setup could not be matched, returns -1.

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

        >>> def rank_setups(query):
        ...     return (
        ...         query.rank_setup(A.f.get_setup(instance=a)),
        ...         query.rank_setup(B.f.get_setup(instance=b))
        ... )

        >>> rank_setups(vbt.CAQuery(cacheable=A.f))
        (32, -1)
        >>> rank_setups(vbt.CAQuery(cacheable=B.f))
        (-1, 32)
        >>> rank_setups(vbt.CAQuery(cacheable=b.f))
        (-1, 32)
        >>> rank_setups(vbt.CAQuery(instance=a, cacheable='f'))
        (27, -1)
        >>> rank_setups(vbt.CAQuery(instance=b, cacheable='f'))
        (-1, 27)
        >>> rank_setups(vbt.CAQuery(instance=a, options=dict(my_option=True)))
        (25, -1)
        >>> rank_setups(vbt.CAQuery(instance=a, options=dict(my_option=False)))
        (-1, -1)
        >>> rank_setups(vbt.CAQuery(instance=b, options=dict(my_option=False)))
        (-1, 25)
        >>> rank_setups(vbt.CAQuery(instance=a))
        (24, -1)
        >>> rank_setups(vbt.CAQuery(instance=b))
        (-1, 24)
        >>> rank_setups(vbt.CAQuery(cls=A))
        (16, -1)
        >>> rank_setups(vbt.CAQuery(cls=B))
        (-1, 16)
        >>> rank_setups(vbt.CAQuery(base_cls=A))
        (8, 8)
        >>> rank_setups(vbt.CAQuery(base_cls=B))
        (-1, 8)
        >>> rank_setups(vbt.CAQuery(base_cls=A, options=dict(my_option=False)))
        (-1, 9)
        >>> rank_setups(vbt.CAQuery(cacheable='f'))
        (3, 3)
        >>> rank_setups(vbt.CAQuery(cacheable='f', options=dict(my_option=False)))
        (-1, 6)
        >>> rank_setups(vbt.CAQuery(options=dict(my_option=True)))
        (1, -1)
        >>> rank_setups(vbt.CAQuery())
        (0, 0)
        ```"""

        cacheable_rank = 0
        if self.cacheable is not None:
            if CASetup.is_cacheable(self.cacheable):
                if setup.cacheable is self.cacheable or setup.cacheable.func is self.cacheable.func:
                    cacheable_rank = 4
                else:
                    return -1
            elif callable(self.cacheable):
                if setup.cacheable.func is self.cacheable:
                    cacheable_rank = 3
                else:
                    return -1
            elif isinstance(self.cacheable, str):
                if setup.cacheable.name == self.cacheable:
                    cacheable_rank = 2
                else:
                    return -1
            elif isinstance(self.cacheable, Regex):
                if self.cacheable.matches(setup.cacheable.name):
                    cacheable_rank = 1
                else:
                    return -1
            else:
                return -1

        instance_rank = 0
        if self.instance is not None:
            if setup.instance is self.instance:
                instance_rank = 1
            else:
                return -1

        cls_rank = 0
        if self.cls is not None:
            if checks.is_class(setup.instance.__class__, self.cls):
                cls_rank = 1
            else:
                return -1

        base_cls_rank = 0
        if self.base_cls is not None:
            if checks.is_instance_of(setup.instance, self.base_cls):
                base_cls_rank = 1
            else:
                return -1

        options_rank = 0
        if self.options is not None:
            options_matched = True
            for k, v in self.options.items():
                if k not in setup.cacheable.options or setup.cacheable.options[k] != v:
                    options_matched = False
            if options_matched:
                options_rank = 1
            else:
                return -1

        conditions = [
            instance_rank == 1 and options_rank == 1 and cacheable_rank == 4,
            instance_rank == 1 and options_rank == 1 and cacheable_rank == 3,
            instance_rank == 1 and options_rank == 1 and cacheable_rank == 2,
            instance_rank == 1 and options_rank == 1 and cacheable_rank == 1,
            instance_rank == 1 and cacheable_rank == 4,
            instance_rank == 1 and cacheable_rank == 3,
            instance_rank == 1 and cacheable_rank == 2,
            instance_rank == 1 and cacheable_rank == 1,
            instance_rank == 1 and options_rank == 1,
            instance_rank == 1,
            cls_rank == 1 and options_rank == 1 and cacheable_rank == 4,
            cls_rank == 1 and options_rank == 1 and cacheable_rank == 3,
            cls_rank == 1 and options_rank == 1 and cacheable_rank == 2,
            cls_rank == 1 and options_rank == 1 and cacheable_rank == 1,
            cls_rank == 1 and cacheable_rank == 4,
            cls_rank == 1 and cacheable_rank == 3,
            cls_rank == 1 and cacheable_rank == 2,
            cls_rank == 1 and cacheable_rank == 1,
            cls_rank == 1 and options_rank == 1,
            cls_rank == 1,
            base_cls_rank == 1 and options_rank == 1 and cacheable_rank == 4,
            base_cls_rank == 1 and options_rank == 1 and cacheable_rank == 3,
            base_cls_rank == 1 and options_rank == 1 and cacheable_rank == 2,
            base_cls_rank == 1 and options_rank == 1 and cacheable_rank == 1,
            base_cls_rank == 1 and cacheable_rank == 4,
            base_cls_rank == 1 and cacheable_rank == 3,
            base_cls_rank == 1 and cacheable_rank == 2,
            base_cls_rank == 1 and cacheable_rank == 1,
            base_cls_rank == 1 and options_rank == 1,
            base_cls_rank == 1,
            cacheable_rank == 4 and options_rank == 1,
            cacheable_rank == 3 and options_rank == 1,
            cacheable_rank == 2 and options_rank == 1,
            cacheable_rank == 1 and options_rank == 1,
            cacheable_rank == 4,
            cacheable_rank == 3,
            cacheable_rank == 2,
            cacheable_rank == 1,
            options_rank == 1
        ]

        for i, condition in enumerate(conditions):
            if condition:
                return len(conditions) - i
        return 0

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
    """Class that represents a caching directive.

    !!! note
        The hash of each `CADirective` instance is based solely on the hash of its query
        because `CacheableRegistry` allows only one `CADirective` instance per each `CAQuery` instance."""

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
        return (self.query,)


dir_queryT = tp.MaybeSequence[tp.Union[CAQuery, CADirective]]
setup_queryT = tp.MaybeSequence[tp.Union['CASetup', CAQuery, CADirective]]


class CacheableRegistry:
    """Class that registers directives and cacheables."""

    def __init__(self) -> None:
        self._setups = set()

    @property
    def setups(self) -> tp.Set['CASetup']:
        """Set of registered `CASetup` instances."""
        return self._setups

    @property
    def directives(self) -> tp.Set[CADirective]:
        """Set of directives pulled from `settings.caching` under `vectorbt._settings.settings`."""
        from vectorbt._settings import settings
        caching_cfg = settings['caching']

        return caching_cfg['directives']

    def add_directive(self, directive: CADirective, update: bool = False) -> None:
        """Add a new directive to `CacheableRegistry.directives`.

        !!! note
            Will raise an error if there is already a directive with an identical query."""
        checks.is_instance_of(directive, CADirective)
        if directive in self.directives:
            if not update:
                raise ValueError(f"Query '{str(directive.query)}' already registered")
            self.remove_directive(directive)
        self.directives.add(directive)

    def remove_directive(self, dir_query: dir_queryT) -> None:
        """Remove a directive from `CacheableRegistry.directives` that matches `query`.

        `query` can be either an instance of `CADirective` or `CAQuery`."""
        if isinstance(dir_query, CADirective):
            query = dir_query.query
        else:
            query = dir_query
        for directive in self.directives:
            if query == directive.query:
                self.directives.remove(directive)

    def should_cache_setup(self, setup: 'CASetup') -> bool:
        """Whether should cache a setup."""
        from vectorbt._settings import settings
        caching_cfg = settings['caching']

        ranked_directives = []
        for directive in self.directives:
            rank = directive.query.rank_setup(setup)
            if rank > -1:
                ranked_directives.append((rank, directive))
        ranked_directives = sorted(ranked_directives, key=lambda x: x[0])
        cache = None
        last_rank = -1
        for rank, directive in ranked_directives:
            if rank == last_rank:
                raise ValueError("Multiple directives return the same rank")
            else:
                last_rank = rank
            if directive.cache is not None:
                if caching_cfg['enabled'] or directive.override_disabled:
                    cache = directive.cache
        if cache is None:
            if not caching_cfg['enabled']:
                return False
            return setup.cacheable.cache
        return cache

    def register_setup(self, setup: 'CASetup') -> None:
        """Register a new setup of type `CASetup`."""
        if setup in self.setups:
            raise ValueError(f"Setup '{str(setup)}' already registered")
        self.setups.add(setup)

    def get_setup(self, cacheable: cacheableT, instance: tp.Optional[object] = None) -> tp.Optional['CASetup']:
        """Get a setup with the same cacheable and instance, or return None."""
        for setup in self.setups:
            if hash(setup) == CASetup.get_hash(cacheable, instance=instance):
                return setup
        return None

    def match_setups(self, setup_query: tp.Optional[setup_queryT] = None) -> tp.Set['CASetup']:
        """Match setups from `CacheableRegistry.setups` against `setup_query`.

        `setup_query` can be either one or more instances of `CASetup`, `CADirective`, and `CAQuery`."""
        matches = set()
        if setup_query is None:
            for setup in self.setups:
                matches.add(setup)
        else:
            if not checks.is_sequence(setup_query):
                setup_query = [setup_query]
            for s in setup_query:
                if isinstance(s, CASetup):
                    if s in self.setups:
                        matches.add(s)
                else:
                    if isinstance(s, CADirective):
                        query = s.query
                    else:
                        query = s
                    for setup in self.setups:
                        if query.rank_setup(setup) > -1:
                            matches.add(setup)
        return matches

    def clear_cache(self, setup_query: tp.Optional[setup_queryT] = None) -> None:
        """Clear all setups or those that match `setup_query`.

        See `CacheableRegistry.match_run_setups`."""
        matches = self.match_setups(setup_query=setup_query)
        for setup in matches:
            setup.clear_cache()


ca_registry = CacheableRegistry()
"""Default registry of type `CacheableRegistry`."""

CASetupT = tp.TypeVar("CASetupT", bound="CASetup")


class CASetup(Hashable, SafeToStr):
    """Class that represents a setup of a cacheable.

    Takes care of caching and communicating with `CacheableRegistry`.

    A setup is hashed by the callable and optionally the instance its bound to.
    This way, it can be uniquely identified.

    !!! note
        Only one instance per each unique combination of `cacheable` and `instance` can exist at a time.

        Use `CASetup.get` class method instead of `CASetup.__init__` to create a setup. The class method
        first checks whether a setup with the same hash has already been registered, and if so, returns it.
        Otherwise, creates and registers a new one. Using `CASetup.__init__` will throw an error if there
        is a setup with the same hash."""

    @classmethod
    def get(cls: tp.Type[CASetupT],
            cacheable: cacheableT,
            instance: tp.Optional[object] = None,
            registry: CacheableRegistry = ca_registry) -> CASetupT:
        """Get setup from `CacheableRegistry` or register a new one."""
        setup = registry.get_setup(cacheable, instance=instance)
        if setup is not None:
            return setup
        return cls(cacheable, instance=instance, registry=registry)

    def __init__(self,
                 cacheable: cacheableT,
                 instance: tp.Optional[object] = None,
                 registry: CacheableRegistry = ca_registry) -> None:
        if not self.is_cacheable(cacheable):
            raise TypeError("cacheable must be a cacheable function, method, or property")
        if instance is None:
            if isinstance(cacheable, cacheable_property):
                raise ValueError("CASetup requires an instance for cacheable_property")
            elif cacheable.is_method:
                raise ValueError("CASetup requires an instance for cacheable_method")

        self._cacheable = cacheable
        self._instance = instance
        self._registry = registry
        self._cache = {}
        self._hits = 0
        self._misses = 0

        self.registry.register_setup(self)

    @property
    def cacheable(self) -> cacheableT:
        """Cacheable callable.

        Must be either instance of `vectorbt.utils.decorators.cacheable_property`
        or `vectorbt.utils.decorators.cacheable_method`."""
        return self._cacheable

    @staticmethod
    def is_cacheable_property(cacheable: tp.Any) -> bool:
        """Check if `cacheable` is a cacheable property."""
        return isinstance(cacheable, cacheable_property)

    @staticmethod
    def is_cacheable_function(cacheable: tp.Any) -> bool:
        """Check if `cacheable` is a cacheable function."""
        return (inspect.isfunction(cacheable) or inspect.ismethod(cacheable)) and hasattr(cacheable, 'get_setup')

    @staticmethod
    def is_cacheable_method(cacheable: tp.Any) -> bool:
        """Check if `cacheable` is a cacheable method."""
        return CASetup.is_cacheable_function(cacheable) and cacheable.is_method

    @staticmethod
    def is_cacheable(cacheable: tp.Any) -> bool:
        """Check if `cacheable` is a cacheable."""
        return CASetup.is_cacheable_property(cacheable) or CASetup.is_cacheable_function(cacheable)

    @property
    def instance(self) -> tp.Optional[object]:
        """Class instance `CASetup.cacheable` is bound to."""
        return self._instance

    @property
    def registry(self) -> CacheableRegistry:
        """Registry of type `CacheableRegistry`."""
        return self._registry

    @property
    def cache(self) -> tp.Dict[int, tp.Any]:
        """Cache dictionary with the call results keyed by the hash of the passed arguments.

        !!! note
            Not included in the hash."""
        return self._cache

    @property
    def hits(self) -> int:
        """Number of hits.

        Gets updated by `CacheableRegistry`.

        !!! note
            Not included in the hash."""
        return self._hits

    @property
    def misses(self) -> int:
        """Number of misses.

        Gets updated by `CacheableRegistry`.

        !!! note
            Not included in the hash."""
        return self._misses

    def run_func(self, *args, **kwargs) -> tp.Any:
        """Run the setup's function without caching."""
        if self.instance is not None:
            return self.cacheable.func(self.instance, *args, **kwargs)
        return self.cacheable.func(*args, **kwargs)

    def get_args_hash(self, *args, **kwargs) -> int:
        """Get the hash of the passed arguments."""
        return hash_args(
            self.cacheable.func,
            args if self.instance is None else (id(self.instance), *args),
            kwargs,
            ignore_args=self.cacheable.ignore_args
        )

    def run_func_and_cache(self, *args, **kwargs) -> tp.Any:
        """Run the setup's function and caches the result."""
        args_hash = self.get_args_hash(*args, **kwargs)
        if args_hash in self.cache:
            self._hits += 1
            return self.cache[args_hash]
        self._misses += 1
        if self.is_cacheable_function(self.cacheable) and \
                self.cacheable.max_size is not None and \
                self.cacheable.max_size <= len(self.cache):
            del self.cache[list(self.cache.keys())[0]]
        self.cache[args_hash] = self.run_func(*args, **kwargs)
        return self.cache[args_hash]

    @property
    def should_cache(self) -> bool:
        """Whether this setup should be cached.

        See `CacheableRegistry.should_cache_setup`."""
        return self.registry.should_cache_setup(self)

    def run(self, *args, **kwargs) -> tp.Any:
        """Run the setup.

        Runs `CASetup.run_func` or `CASetup.run_func_and_cache` depending on `CASetup.should_cache`."""
        if self.should_cache:
            return self.run_func_and_cache(*args, **kwargs)
        self.run_func(*args, **kwargs)

    def enable_caching(self, override_disabled: bool = False, rank: tp.Optional[int] = None) -> None:
        pass

    def disable_caching(self, rank: tp.Optional[int] = None) -> None:
        pass

    def clear_cache(self) -> None:
        """Clear `CASetup.cache`."""
        self.cache.clear()
        self._hits = 0
        self._misses = 0

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(" \
               f"cacheable={self.cacheable}, " \
               f"instance={self.instance}, " \
               f"hits={self.hits}, " \
               f"misses={self.misses})"

    @property
    def hash_key(self) -> tuple:
        return self.cacheable, id(self.instance) if self.instance is not None else None

    @staticmethod
    def get_hash(cacheable: cacheableT, instance: tp.Optional[object] = None) -> int:
        """Static method to get hash of the cacheable and the instance."""
        return hash((cacheable, id(instance) if instance is not None else None))
