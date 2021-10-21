# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""Global registry for cacheables."""

import warnings
import inspect
import weakref

from vectorbt import _typing as tp
from vectorbt.utils import checks
from vectorbt.utils.docs import SafeToStr
from vectorbt.utils.decorators import cacheableT, cacheable_property
from vectorbt.utils.caching import Cacheable
from vectorbt.utils.parsing import Regex, hash_args, UnhashableArgsError
from vectorbt.utils.hashing import Hashable

__pdoc__ = {}

_NO_OBJ = object()


def is_cacheable_function(cacheable: tp.Any) -> bool:
    """Check if `cacheable` is a cacheable function."""
    return callable(cacheable) \
           and hasattr(cacheable, 'is_method') \
           and not cacheable.is_method \
           and hasattr(cacheable, 'is_cacheable') \
           and cacheable.is_cacheable


def is_cacheable_property(cacheable: tp.Any) -> bool:
    """Check if `cacheable` is a cacheable property."""
    return isinstance(cacheable, cacheable_property)


def is_cacheable_method(cacheable: tp.Any) -> bool:
    """Check if `cacheable` is a cacheable method."""
    return callable(cacheable) \
           and hasattr(cacheable, 'is_method') \
           and cacheable.is_method \
           and hasattr(cacheable, 'is_cacheable') \
           and cacheable.is_cacheable


def is_bindable_cacheable(cacheable: tp.Any) -> bool:
    """Check if `cacheable` is a cacheable that can be bound to an instance."""
    return is_cacheable_property(cacheable) or is_cacheable_method(cacheable)


def is_cacheable(cacheable: tp.Any) -> bool:
    """Check if `cacheable` is a cacheable."""
    return is_cacheable_function(cacheable) or is_bindable_cacheable(cacheable)


CAQueryT = tp.TypeVar("CAQueryT", bound="CAQuery")


class CAQuery(Hashable, SafeToStr):
    """Class that represents a query for matching and ranking setups."""

    @classmethod
    def parse(cls: tp.Type[CAQueryT], query_like: tp.Any, use_base_cls: bool = True) -> CAQueryT:
        """Parse a query-like object.

        !!! note
            Not all attribute combinations can be safely parsed by this function.
            For example, you cannot combine cacheable together with options.

        ## Example

        ```python-repl
        >>> import vectorbt as vbt

        >>> CAQuery.parse(func)
        <=> CAQuery(cacheable=func)

        >>> CAQuery.parse("a")
        <=> CAQuery(cacheable='a')

        >>> CAQuery.parse("A.a")
        <=> CAQuery(base_cls='A', cacheable='a')

        >>> CAQuery.parse("A")
        <=> CAQuery(base_cls='A')

        >>> CAQuery.parse("A", use_base_cls=False)
        <=> CAQuery(cls='A')

        >>> CAQuery.parse(vbt.Regex("[A-B]"))
        <=> CAQuery(base_cls=vbt.Regex("[A-B]"))

        >>> CAQuery.parse(dict(my_option=100))
        <=> CAQuery(options=dict(my_option=100))

        >>> CAQuery.parse(123)
        <=> CAQuery(instance=123)
        ```"""
        if query_like is None:
            return CAQuery()
        if isinstance(query_like, CAQuery):
            return query_like
        if isinstance(query_like, CABaseSetup):
            return query_like.query
        if isinstance(query_like, cacheable_property):
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
        if callable(query_like):
            return cls(cacheable=query_like)
        return cls(instance=query_like)

    def __init__(self,
                 cacheable: tp.Optional[tp.Union[tp.Callable, cacheableT, str, Regex]] = None,
                 instance: tp.Optional[Cacheable] = None,
                 cls: tp.Optional[tp.TypeLike] = None,
                 base_cls: tp.Optional[tp.TypeLike] = None,
                 options: tp.Optional[dict] = None) -> None:
        self._cacheable = cacheable
        if instance is not None:
            instance = weakref.ref(instance)
        self._instance = instance
        self._cls = cls
        self._base_cls = base_cls
        self._options = options

    @property
    def cacheable(self) -> tp.Optional[tp.Union[tp.Callable, cacheableT, str, Regex]]:
        """Cacheable object or its name (case-sensitive)."""
        return self._cacheable

    @property
    def instance(self) -> tp.Optional[tp.Union[Cacheable, object]]:
        """Weak reference to the instance `CAQuery.cacheable` is bound to."""
        if self._instance is not None and self._instance() is None:
            return _NO_OBJ
        return self._instance() if self._instance is not None else None

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

    def matches_setup(self, setup: 'CABaseSetup') -> bool:
        """Return whether the setup matches this query.

        ## Example

        Let's evaluate various queries:

        ```python-repl
        >>> import vectorbt as vbt

        >>> class A(vbt.Cacheable):
        ...     @vbt.cached_property(my_option=True)
        ...     def f(self):
        ...         return None

        >>> class B(A):
        ...     @vbt.cached_method(my_option=False)
        ...     def f(self):
        ...         return None

        >>> @vbt.cached
        ... def f(self):
        ...     return None

        >>> a = A()
        >>> b = B()

        >>> def match_query(query):
        ...     matched = []
        ...     if query.matches_setup(A.f.get_ca_setup(a)):
        ...         matched.append('A.f')
        ...     if query.matches_setup(B.f.get_ca_setup(b)):
        ...         matched.append('B.f')
        ...     if query.matches_setup(f.get_ca_setup()):
        ...         matched.append('f')
        ...     return matched

        >>> match_query(vbt.CAQuery(cacheable='f'))
        ['A.f', 'B.f', 'f']
        >>> match_query(vbt.CAQuery(cacheable=A.f))
        ['A.f']
        >>> match_query(vbt.CAQuery(cacheable=B.f))
        ['B.f']
        >>> match_query(vbt.CAQuery(cacheable=f))
        ['f']
        >>> match_query(vbt.CAQuery(cacheable=f.func))
        ['f']
        >>> match_query(vbt.CAQuery(instance=a, cacheable='f'))
        ['A.f']
        >>> match_query(vbt.CAQuery(instance=b, cacheable='f'))
        ['B.f']
        >>> match_query(vbt.CAQuery(instance=a))
        ['A.f']
        >>> match_query(vbt.CAQuery(cls=A))
        ['A.f']
        >>> match_query(vbt.CAQuery(base_cls=A))
        ['A.f', 'B.f']
        >>> match_query(vbt.CAQuery(cls=vbt.Regex('[A-B]')))
        ['A.f', 'B.f']
        >>> match_query(vbt.CAQuery(cacheable='f', options=dict(my_option=True)))
        ['A.f']
        >>> match_query(vbt.CAQuery(base_cls=A, options=dict(my_option=False)))
        ['B.f']
        >>> match_query(vbt.CAQuery())
        []
        ```"""

        if self.cacheable is not None:
            if not isinstance(setup, (CARunSetup, CAUnboundSetup)):
                return False
            if is_cacheable(self.cacheable):
                if setup.cacheable is not self.cacheable and setup.cacheable.func is not self.cacheable.func:
                    return False
            elif callable(self.cacheable):
                if setup.cacheable.func is not self.cacheable:
                    return False
            elif isinstance(self.cacheable, str):
                if setup.cacheable.name != self.cacheable:
                    return False
            elif isinstance(self.cacheable, Regex):
                if not self.cacheable.matches(setup.cacheable.name):
                    return False
            else:
                return False

        if self.instance is not None:
            if not isinstance(setup, (CARunSetup, CAInstanceSetup)):
                return False
            if setup.instance is _NO_OBJ:
                return False
            if setup.instance is not self.instance:
                return False

        if self.cls is not None:
            if not isinstance(setup, (CARunSetup, CAInstanceSetup, CAClassSetup)):
                return False
            if isinstance(setup, (CARunSetup, CAInstanceSetup)) \
                    and setup.instance is _NO_OBJ:
                return False
            if isinstance(setup, (CARunSetup, CAInstanceSetup)) \
                    and not checks.is_class(setup.instance.__class__, self.cls):
                return False
            if isinstance(setup, CAClassSetup) \
                    and not checks.is_class(setup.cls, self.cls):
                return False

        if self.base_cls is not None:
            if not isinstance(setup, (CARunSetup, CAInstanceSetup, CAClassSetup)):
                return False
            if isinstance(setup, (CARunSetup, CAInstanceSetup)) \
                    and setup.instance is _NO_OBJ:
                return False
            if isinstance(setup, (CARunSetup, CAInstanceSetup)) \
                    and not checks.is_subclass_of(setup.instance.__class__, self.base_cls):
                return False
            if isinstance(setup, CAClassSetup) \
                    and not checks.is_subclass_of(setup.cls, self.base_cls):
                return False

        if self.options is not None:
            if not isinstance(setup, (CARunSetup, CAUnboundSetup)):
                return False
            for k, v in self.options.items():
                if k not in setup.cacheable.options or setup.cacheable.options[k] != v:
                    return False

        if self.cacheable is None \
                and self.instance is None \
                and self.cls is None \
                and self.base_cls is None \
                and self.options is None:
            return False

        return True

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


class CacheableRegistry:
    """Class that registers cacheables."""

    def __init__(self) -> None:
        self._run_setups = dict()
        self._unbound_setups = dict()
        self._instance_setups = dict()
        self._class_setups = dict()

    @property
    def run_setups(self) -> tp.Dict[int, 'CARunSetup']:
        """Dict of registered `CARunSetup` instances by their hash."""
        return self._run_setups

    @property
    def unbound_setups(self) -> tp.Dict[int, 'CAUnboundSetup']:
        """Dict of registered `CAUnboundSetup` instances by their hash."""
        return self._unbound_setups

    @property
    def instance_setups(self) -> tp.Dict[int, 'CAInstanceSetup']:
        """Dict of registered `CAInstanceSetup` instances by their hash."""
        return self._instance_setups

    @property
    def class_setups(self) -> tp.Dict[int, 'CAClassSetup']:
        """Dict of registered `CAClassSetup` instances by their hash."""
        return self._class_setups

    def register_setup(self, setup: 'CABaseSetup') -> None:
        """Register a new setup of type `CABaseSetup`."""
        if isinstance(setup, CARunSetup):
            setups = self.run_setups
        elif isinstance(setup, CAUnboundSetup):
            setups = self.unbound_setups
        elif isinstance(setup, CAInstanceSetup):
            setups = self.instance_setups
        elif isinstance(setup, CAClassSetup):
            setups = self.class_setups
        else:
            raise TypeError(str(type(setup)))
        if hash(setup) in setups:
            raise ValueError(f"Setup '{str(setup)}' already registered")
        setups[hash(setup)] = setup

    def deregister_setup(self, setup: 'CABaseSetup') -> None:
        """Deregister a new setup of type `CABaseSetup`."""
        if isinstance(setup, CARunSetup):
            setups = self.run_setups
        elif isinstance(setup, CAUnboundSetup):
            setups = self.unbound_setups
        elif isinstance(setup, CAInstanceSetup):
            setups = self.instance_setups
        elif isinstance(setup, CAClassSetup):
            setups = self.class_setups
        else:
            raise TypeError(str(type(setup)))
        if hash(setup) not in setups:
            raise ValueError(f"Setup '{str(setup)}' not registered")
        del setups[hash(setup)]

    def get_run_setup(self, cacheable: cacheableT,
                      instance: tp.Optional[Cacheable] = None) -> tp.Optional['CARunSetup']:
        """Get a setup of type `CARunSetup` with this cacheable and instance, or return None."""
        return self.run_setups.get(CARunSetup.get_hash(cacheable, instance=instance), None)

    def get_unbound_setup(self, cacheable: cacheableT) -> tp.Optional['CAUnboundSetup']:
        """Get a setup of type `CAUnboundSetup` with this cacheable or return None."""
        return self.unbound_setups.get(CAUnboundSetup.get_hash(cacheable), None)

    def get_instance_setup(self, instance: Cacheable) -> tp.Optional['CAInstanceSetup']:
        """Get a setup of type `CAInstanceSetup` with this instance or return None."""
        return self.instance_setups.get(CAInstanceSetup.get_hash(instance), None)

    def get_class_setup(self, cls: tp.Type[Cacheable]) -> tp.Optional['CAClassSetup']:
        """Get a setup of type `CAInstanceSetup` with this class or return None."""
        return self.class_setups.get(CAClassSetup.get_hash(cls), None)

    def match_setups(self,
                     query_like: tp.MaybeIterable[tp.Any],
                     collapse: bool = False,
                     kind: tp.Optional[tp.MaybeIterable[str]] = None,
                     exclude: tp.Optional[tp.MaybeIterable['CABaseSetup']] = None,
                     exclude_children: bool = True) -> tp.Set['CABaseSetup']:
        """Match setups from `CacheableRegistry.setups` against `query_like`.

        `query_like` can be one or more query-like objects that will be parsed using `CAQuery.parse`.

        Set `collapse` to True to remove child setups that belong to a matched parent setup.

        `kind` can be one or multiple of the following:

        * 'runnable' to only return runnable setups (instances of `CARunSetup`)
        * 'unbound' to only return unbound setups (instances of `CAUnboundSetup`)
        * 'instance' to only return instance setups (instances of `CAInstanceSetup`)
        * 'class' to only return class setups (instances of `CAClassSetup`)

        Set `exclude` to one or multiple setups to exclude. To not exclude their children,
        set `exclude_children` to False.

        !!! note
            `exclude_children` is applied only when `collapse` is True."""
        if not checks.is_iterable(query_like) or isinstance(query_like, (str, tuple)):
            query_like = [query_like]
        query_like = list(map(CAQuery.parse, query_like))
        if kind is None:
            kind = {'runnable', 'unbound', 'instance', 'class'}
        if exclude is None:
            exclude = set()
        if isinstance(exclude, CABaseSetup):
            exclude = {exclude}
        else:
            exclude = set(exclude)

        matches = set()
        if not collapse:
            if isinstance(kind, str):
                if kind.lower() == 'runnable':
                    setups = set(self.run_setups.values())
                elif kind.lower() == 'unbound':
                    setups = set(self.unbound_setups.values())
                elif kind.lower() == 'instance':
                    setups = set(self.instance_setups.values())
                elif kind.lower() == 'class':
                    setups = set(self.class_setups.values())
                else:
                    raise ValueError(f"kind '{kind}' is not supported")
                for setup in setups:
                    if setup not in exclude:
                        for q in query_like:
                            if q.matches_setup(setup):
                                matches.add(setup)
                                break
            elif checks.is_iterable(kind):
                matches = set.union(*[self.match_setups(
                    query_like,
                    kind=k,
                    collapse=collapse,
                    exclude=exclude,
                    exclude_children=exclude_children
                ) for k in kind])
            else:
                raise TypeError(f"kind must be either a string or a sequence of strings, not {type(kind)}")
        else:
            collapse_setups = set()
            if isinstance(kind, str):
                kind = {kind}
            else:
                kind = set(kind)
            if 'class' in kind:
                class_matches = set()
                for class_setup in self.class_setups.values():
                    for q in query_like:
                        if q.matches_setup(class_setup):
                            if class_setup not in exclude:
                                class_matches.add(class_setup)
                            if class_setup not in exclude or exclude_children:
                                collapse_setups |= class_setup.child_setups
                            break
                for class_setup in class_matches:
                    if class_setup not in collapse_setups:
                        matches.add(class_setup)
            if 'instance' in kind:
                for instance_setup in self.instance_setups.values():
                    if instance_setup not in collapse_setups:
                        for q in query_like:
                            if q.matches_setup(instance_setup):
                                if instance_setup not in exclude:
                                    matches.add(instance_setup)
                                if instance_setup not in exclude or exclude_children:
                                    collapse_setups |= instance_setup.child_setups
                                break
            if 'unbound' in kind:
                for unbound_setup in self.unbound_setups.values():
                    if unbound_setup not in collapse_setups:
                        for q in query_like:
                            if q.matches_setup(unbound_setup):
                                if unbound_setup not in exclude:
                                    matches.add(unbound_setup)
                                if unbound_setup not in exclude or exclude_children:
                                    collapse_setups |= unbound_setup.child_setups
                                break
            if 'runnable' in kind:
                for run_setup in self.run_setups.values():
                    if run_setup not in collapse_setups:
                        for q in query_like:
                            if q.matches_setup(run_setup):
                                if run_setup not in exclude:
                                    matches.add(run_setup)
                                break
        return matches


ca_registry = CacheableRegistry()
"""Default registry of type `CacheableRegistry`."""


class CABaseSetup:
    """Base class that exposes properties and methods for cache management."""

    def __init__(self,
                 registry: CacheableRegistry = ca_registry,
                 use_cache: tp.Optional[bool] = None,
                 whitelist: tp.Optional[bool] = None) -> None:
        self._registry = registry
        self._use_cache = use_cache
        self._whitelist = whitelist

    @property
    def query(self) -> CAQuery:
        """Query to match this setup."""
        raise NotImplementedError

    @property
    def registry(self) -> CacheableRegistry:
        """Registry of type `CacheableRegistry`."""
        return self._registry

    @property
    def use_cache(self) -> tp.Optional[bool]:
        """Whether caching is enabled."""
        return self._use_cache

    @property
    def whitelist(self) -> tp.Optional[bool]:
        """Whether to cache even if caching was disabled globally."""
        return self._whitelist

    def enable_whitelist(self) -> None:
        """Enable whitelisting."""
        self._whitelist = True

    def disable_whitelist(self) -> None:
        """Disable whitelisting."""
        self._whitelist = False

    def enable_caching(self, force: bool = False, silence_warnings: bool = False) -> None:
        """Enable caching.

        Set `force` to True to whitelist this setup."""
        from vectorbt._settings import settings
        caching_cfg = settings['caching']

        self._use_cache = True
        if force:
            self._whitelist = True
        else:
            if not caching_cfg['enabled'] and not self.whitelist and not silence_warnings:
                warnings.warn("Caching is disabled globally and setup is not whitelisted", stacklevel=2)

    def disable_caching(self, clear_cache: bool = True) -> None:
        """Disable caching.

        Set `clear_cache` to True to also clear the cache."""
        self._use_cache = False
        if clear_cache:
            self.clear_cache()

    def clear_cache(self) -> None:
        """Clear the cache."""
        raise NotImplementedError


class CASetupDelegatorMixin:
    """Mixin class that delegates cache management to child setups."""

    @property
    def child_setups(self) -> tp.Set[CABaseSetup]:
        """Child setups."""
        raise NotImplementedError

    def enable_whitelist(self) -> None:
        """Calls `CABaseSetup.enable_whitelist` on each child setup."""
        for setup in self.child_setups:
            setup.enable_whitelist()

    def disable_whitelist(self) -> None:
        """Calls `CABaseSetup.disable_whitelist` on each child setup."""
        for setup in self.child_setups:
            setup.disable_whitelist()

    def enable_caching(self, force: bool = False, silence_warnings: bool = False) -> None:
        """Calls `CABaseSetup.enable_caching` on each child setup."""
        for setup in self.child_setups:
            setup.enable_caching(force=force, silence_warnings=silence_warnings)

    def disable_caching(self, clear_cache: bool = True) -> None:
        """Calls `CABaseSetup.disable_caching` on each child setup."""
        for setup in self.child_setups:
            setup.disable_caching(clear_cache=clear_cache)

    def clear_cache(self) -> None:
        """Calls `CABaseSetup.clear_cache` on each child setup."""
        for setup in self.child_setups:
            setup.clear_cache()

    @property
    def hits(self) -> int:
        """Number of hits across all child setups."""
        return sum([setup.hits for setup in self.child_setups if hasattr(setup, 'hits')])

    @property
    def misses(self) -> int:
        """Number of misses across all child setups."""
        return sum([setup.misses for setup in self.child_setups if hasattr(setup, 'misses')])


class CABaseParentSetup(CABaseSetup, CASetupDelegatorMixin):
    """Base class acting as a stateful parent setup that delegates cache management to child setups."""

    @property
    def child_setups(self) -> tp.Set[CABaseSetup]:
        """Get child setups that match `CABaseParentSetup.query`."""
        return self.registry.match_setups(self.query, kind='collapse')

    def enable_whitelist(self) -> None:
        """Calls `CABaseSetup.enable_whitelist` on each child setup."""
        CABaseSetup.enable_whitelist(self)
        CASetupDelegatorMixin.enable_whitelist(self)

    def disable_whitelist(self) -> None:
        """Calls `CABaseSetup.disable_whitelist` on each child setup."""
        CABaseSetup.disable_whitelist(self)
        CASetupDelegatorMixin.disable_whitelist(self)

    def enable_caching(self, force: bool = False, silence_warnings: bool = False) -> None:
        """Calls `CABaseSetup.enable_caching` on each child setup."""
        CABaseSetup.enable_caching(self, force=force, silence_warnings=silence_warnings)
        CASetupDelegatorMixin.enable_caching(self, force=force, silence_warnings=silence_warnings)

    def disable_caching(self, clear_cache: bool = True) -> None:
        """Calls `CABaseSetup.disable_caching` on each child setup."""
        CABaseSetup.disable_caching(self, clear_cache=False)
        CASetupDelegatorMixin.disable_caching(self, clear_cache=clear_cache)

    def clear_cache(self) -> None:
        """Calls `CABaseSetup.clear_cache` on each child setup."""
        CASetupDelegatorMixin.clear_cache(self)


CAClassSetupT = tp.TypeVar("CAClassSetupT", bound="CAClassSetup")


class CAClassSetup(CABaseParentSetup, Hashable, SafeToStr):
    """Class that represents a setup of a cacheable class.

    The provided class must subclass `vectorbt.utils.caching.Cacheable`.

    Delegates cache management to its child setups of type `CAClassSetup` and `CAInstanceSetup`."""

    @staticmethod
    def get_hash(cls: tp.Type[Cacheable]) -> int:
        """Static method to get the hash of the class."""
        return hash((cls,))

    @staticmethod
    def get_superclasses(cls: tp.Type[Cacheable]) -> tp.List[tp.Type[Cacheable]]:
        """Get the cacheable superclasses of a cacheable class."""
        superclasses = []
        for supercls in inspect.getmro(cls):
            if issubclass(supercls, Cacheable):
                if supercls is not cls:
                    superclasses.append(supercls)
        return superclasses

    @staticmethod
    def get_subclasses(cls: tp.Type[Cacheable]) -> tp.List[tp.Type[Cacheable]]:
        """Get the cacheable subclasses of a cacheable class."""
        subclasses = []
        for subcls in cls.__subclasses__():
            if issubclass(subcls, Cacheable):
                if subcls is not cls:
                    subclasses.append(subcls)
            subclasses.extend(CAClassSetup.get_subclasses(subcls))
        return subclasses

    @classmethod
    def get(cls: tp.Type[CAClassSetupT],
            cls_: tp.Type[Cacheable],
            registry: CacheableRegistry = ca_registry,
            **kwargs) -> CAClassSetupT:
        """Get setup from `CacheableRegistry` or register a new one.

        `**kwargs` are passed to `CAClassSetup.__init__`."""
        setup = registry.get_class_setup(cls_)
        if setup is not None:
            return setup
        return cls(cls_, registry=registry, **kwargs)

    def __init__(self,
                 cls_: tp.Type[Cacheable],
                 registry: CacheableRegistry = ca_registry,
                 use_cache: tp.Optional[bool] = None,
                 whitelist: tp.Optional[bool] = None) -> None:
        checks.assert_subclass_of(cls_, Cacheable)

        CABaseParentSetup.__init__(
            self,
            registry=registry,
            use_cache=use_cache,
            whitelist=whitelist
        )

        self._cls = cls_

        if self.use_cache is None or self.whitelist is None:
            superclass_setups = self.lazy_superclass_setups
            for setup in superclass_setups:
                if self.use_cache is None:
                    if setup.use_cache is not None:
                        self._use_cache = setup.use_cache
                if self.whitelist is None:
                    if setup.whitelist is not None:
                        self._whitelist = setup.whitelist

        self.registry.register_setup(self)

    @property
    def query(self) -> CAQuery:
        return CAQuery(base_cls=self.cls)

    @property
    def cls(self) -> tp.Type[Cacheable]:
        """Cacheable class."""
        return self._cls

    @property
    def lazy_superclass_setups(self) -> tp.Set['CAClassSetup']:
        """Already registered setups of type `CAClassSetup` of each in `CAClassSetup.get_superclasses`."""
        setups = set()
        for super_cls in self.get_superclasses(self.cls):
            if ca_registry.get_class_setup(super_cls) is not None:
                setups.add(super_cls.get_ca_setup())
        return setups

    @property
    def lazy_subclass_setups(self) -> tp.Set['CAClassSetup']:
        """Already registered setups of type `CAClassSetup` of each in `CAClassSetup.get_subclasses`."""
        setups = set()
        for base_cls in self.get_subclasses(self.cls):
            if ca_registry.get_class_setup(base_cls) is not None:
                setups.add(base_cls.get_ca_setup())
        return setups

    @property
    def unbound_setups(self) -> tp.Set['CAUnboundSetup']:
        """Setups of type `CAUnboundSetup` of cacheable members of the class."""
        members = inspect.getmembers(self.cls, is_bindable_cacheable)
        return {attr.get_ca_setup() for attr_name, attr in members}

    @property
    def instance_setups(self) -> tp.Set['CAInstanceSetup']:
        """Setups of type `CAInstanceSetup` of instances of the class."""
        matches = set()
        for instance_setup in ca_registry.instance_setups.values():
            if instance_setup.class_setup is self:
                matches.add(instance_setup)
        return matches

    @property
    def child_setups(self) -> tp.Set[tp.Union['CAClassSetup', 'CAInstanceSetup']]:
        return self.lazy_subclass_setups | self.instance_setups

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(" \
               f"cls={self.cls})"

    @property
    def hash_key(self) -> tuple:
        return (self.cls,)


CAInstanceSetupT = tp.TypeVar("CAInstanceSetupT", bound="CAInstanceSetup")


class CAInstanceSetup(CABaseParentSetup, Hashable, SafeToStr):
    """Class that represents a setup of an instance that has cacheables bound to it.

    The provided instance must be of `vectorbt.utils.caching.Cacheable`.

    Delegates cache management to its child setups of type `CARunSetup`."""

    @staticmethod
    def get_hash(instance: Cacheable) -> int:
        """Static method to get the hash of the instance."""
        return hash((id(instance),))

    @classmethod
    def get(cls: tp.Type[CAInstanceSetupT],
            instance: Cacheable,
            registry: CacheableRegistry = ca_registry,
            **kwargs) -> CAInstanceSetupT:
        """Get setup from `CacheableRegistry` or register a new one.

        `**kwargs` are passed to `CAInstanceSetup.__init__`."""
        setup = registry.get_instance_setup(instance)
        if setup is not None:
            return setup
        return cls(instance, registry=registry, **kwargs)

    def __init__(self,
                 instance: Cacheable,
                 registry: CacheableRegistry = ca_registry,
                 use_cache: tp.Optional[bool] = None,
                 whitelist: tp.Optional[bool] = None) -> None:
        checks.assert_instance_of(instance, Cacheable)

        CABaseParentSetup.__init__(
            self,
            registry=registry,
            use_cache=use_cache,
            whitelist=whitelist
        )

        self._instance = weakref.ref(instance)

        if self.use_cache is None or self.whitelist is None:
            class_setup = self.class_setup
            if class_setup is not None:
                if self.use_cache is None:
                    if class_setup.use_cache is not None:
                        self._use_cache = class_setup.use_cache
                if self.whitelist is None:
                    if class_setup.whitelist is not None:
                        self._whitelist = class_setup.whitelist

        self.registry.register_setup(self)

    @property
    def query(self) -> CAQuery:
        return CAQuery(instance=self.instance)

    @property
    def instance(self) -> tp.Union[Cacheable, object]:
        """Weak reference to the instance."""
        if self._instance() is None:
            return _NO_OBJ
        return self._instance()

    @property
    def class_setup(self) -> tp.Optional[CAClassSetup]:
        """Setup of type `CAClassSetup` of the cacheable class of the instance."""
        if self.instance is _NO_OBJ:
            return None
        return self.registry.get_class_setup(self.instance.__class__)

    @property
    def unbound_setups(self) -> tp.Set['CAUnboundSetup']:
        """Setups of type `CAUnboundSetup` of unbound cacheables declared in the class of the instance."""
        if self.instance is _NO_OBJ:
            return set()
        members = inspect.getmembers(self.instance.__class__, is_bindable_cacheable)
        return {attr.get_ca_setup() for attr_name, attr in members}

    @property
    def run_setups(self) -> tp.Set['CARunSetup']:
        """Setups of type `CARunSetup` of cacheables bound to the instance."""
        if self.instance is _NO_OBJ:
            return set()
        matches = set()
        for run_setup in ca_registry.run_setups.values():
            if run_setup.instance_setup is self:
                matches.add(run_setup)
        return matches

    @property
    def child_setups(self) -> tp.Set['CARunSetup']:
        return self.run_setups

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(" \
               f"instance={self.instance})"

    @property
    def hash_key(self) -> tuple:
        return (id(self.instance),)


CAUnboundSetupT = tp.TypeVar("CAUnboundSetupT", bound="CAUnboundSetup")


class CAUnboundSetup(CABaseParentSetup, Hashable, SafeToStr):
    """Class that represents a setup of an unbound cacheable property or method.

    One unbound cacheable property or method can be bound to multiple instances, thus there is
    one-to-many relationship between `CAUnboundSetup` and `CARunSetup` instances.

    Delegates cache management to its child setups of type `CARunSetup`."""

    @staticmethod
    def get_hash(cacheable: cacheableT) -> int:
        """Static method to get the hash of the cacheable and the instance."""
        return hash((cacheable,))

    @classmethod
    def get(cls: tp.Type[CAUnboundSetupT],
            cacheable: cacheableT,
            registry: CacheableRegistry = ca_registry,
            **kwargs) -> CAUnboundSetupT:
        """Get setup from `CacheableRegistry` or register a new one.

        `**kwargs` are passed to `CAUnboundSetup.__init__`."""
        setup = registry.get_unbound_setup(cacheable)
        if setup is not None:
            return setup
        return cls(cacheable, registry=registry, **kwargs)

    def __init__(self,
                 cacheable: cacheableT,
                 registry: CacheableRegistry = ca_registry,
                 use_cache: bool = False,
                 whitelist: bool = False) -> None:
        if not is_bindable_cacheable(cacheable):
            raise TypeError("cacheable must be either cacheable_property or cacheable_method")

        CABaseParentSetup.__init__(
            self,
            registry=registry,
            use_cache=use_cache,
            whitelist=whitelist
        )
        self._cacheable = cacheable

        self.registry.register_setup(self)

    @property
    def query(self) -> CAQuery:
        return CAQuery(cacheable=self.cacheable)

    @property
    def cacheable(self) -> cacheableT:
        """Cacheable object."""
        return self._cacheable

    @property
    def run_setups(self) -> tp.Set['CARunSetup']:
        """Setups of type `CARunSetup` of bound cacheables."""
        matches = set()
        for run_setup in ca_registry.run_setups.values():
            if run_setup.unbound_setup is self:
                matches.add(run_setup)
        return matches

    @property
    def child_setups(self) -> tp.Set['CARunSetup']:
        return self.run_setups

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(" \
               f"cacheable={self.cacheable})"

    @property
    def hash_key(self) -> tuple:
        return (self.cacheable,)


CARunSetupT = tp.TypeVar("CARunSetupT", bound="CARunSetup")


class CARunSetup(CABaseSetup, Hashable, SafeToStr):
    """Class that represents a runnable setup of `vectorbt.utils.decorators.cacheable_property`,
    `vectorbt.utils.decorators.cacheable_method`, or `vectorbt.utils.decorators.cacheable`.

    Takes care of running functions and caching the results.

    A setup is hashed by the callable and optionally the id of the instance its bound to.
    This way, it can be uniquely identified among all setups.

    !!! note
        Cacheable properties and methods must provide an instance.

        Only one instance per each unique combination of `cacheable` and `instance` can exist at a time.

        Use `CARunSetup.get` class method instead of `CARunSetup.__init__` to create a setup. The class method
        first checks whether a setup with the same hash has already been registered, and if so, returns it.
        Otherwise, creates and registers a new one. Using `CARunSetup.__init__` will throw an error if there
        is a setup with the same hash.

        Cacheables that are declared in classes that do not subclass `vectorbt.utils.caching.Cacheable`
        will only have their cacheable setups registered once the cacheable is run, not upon
        the instantiation of the class."""

    @staticmethod
    def get_hash(cacheable: cacheableT, instance: tp.Optional[Cacheable] = None) -> int:
        """Static method to get the hash of the cacheable and the instance."""
        return hash((cacheable, id(instance) if instance is not None else None))

    @classmethod
    def get(cls: tp.Type[CARunSetupT],
            cacheable: cacheableT,
            instance: tp.Optional[Cacheable] = None,
            registry: CacheableRegistry = ca_registry,
            **kwargs) -> CARunSetupT:
        """Get setup from `CacheableRegistry` or register a new one.

        `**kwargs` are passed to `CARunSetup.__init__`."""
        setup = registry.get_run_setup(cacheable, instance=instance)
        if setup is not None:
            return setup
        return cls(cacheable, instance=instance, registry=registry, **kwargs)

    def __init__(self,
                 cacheable: cacheableT,
                 instance: tp.Optional[Cacheable] = None,
                 registry: CacheableRegistry = ca_registry,
                 use_cache: tp.Optional[bool] = None,
                 whitelist: tp.Optional[bool] = None,
                 max_size: tp.Optional[int] = None,
                 ignore_args: tp.Optional[tp.Sequence[tp.AnnArgQuery]] = None) -> None:
        if not is_cacheable(cacheable):
            raise TypeError("cacheable must be either cacheable_property, cacheable_method, or cacheable")
        if instance is None:
            if is_cacheable_property(cacheable):
                raise ValueError("CARunSetup requires an instance for cacheable_property")
            elif is_cacheable_method(cacheable):
                raise ValueError("CARunSetup requires an instance for cacheable_method")
        else:
            checks.assert_instance_of(instance, Cacheable)
            if is_cacheable_function(cacheable):
                raise ValueError("Cacheable functions can't have an instance")

        CABaseSetup.__init__(
            self,
            registry=registry,
            use_cache=use_cache,
            whitelist=whitelist
        )

        self._cacheable = cacheable
        if instance is not None:
            instance = weakref.ref(instance)
        self._instance = instance
        self._max_size = max_size
        self._ignore_args = ignore_args
        self._cache = {}
        self._hits = 0
        self._misses = 0

        if self.use_cache is None or self.whitelist is None:
            instance_setup = self.instance_setup
            unbound_setup = self.unbound_setup
            if self.use_cache is None:
                if instance_setup is not None and instance_setup.use_cache is not None:
                    self._use_cache = instance_setup.use_cache
                else:
                    self._use_cache = unbound_setup.use_cache
            if self.whitelist is None:
                if instance_setup is not None and instance_setup.whitelist is not None:
                    self._whitelist = instance_setup.whitelist
                else:
                    self._whitelist = unbound_setup.whitelist

        self.registry.register_setup(self)

    @property
    def query(self) -> CAQuery:
        return CAQuery(cacheable=self.cacheable, instance=self.instance)

    @property
    def cacheable(self) -> cacheableT:
        """Cacheable object."""
        return self._cacheable

    @property
    def instance(self) -> tp.Optional[tp.Union[Cacheable, object]]:
        """Weak reference to the instance `CARunSetup.cacheable` is bound to."""
        if self._instance is not None and self._instance() is None:
            return _NO_OBJ
        return self._instance() if self._instance is not None else None

    @property
    def max_size(self) -> tp.Optional[int]:
        """Maximum number of entries in `CARunSetup.cache`."""
        return self._max_size

    @property
    def ignore_args(self) -> tp.Optional[tp.Sequence[tp.AnnArgQuery]]:
        """Arguments to ignore when hashing."""
        return self._ignore_args

    @property
    def cache(self) -> tp.Dict[int, tp.Any]:
        """Cache dictionary with the call results keyed by the hash of the passed arguments."""
        return self._cache

    @property
    def hits(self) -> int:
        """Number of hits."""
        return self._hits

    @property
    def misses(self) -> int:
        """Number of misses."""
        return self._misses

    @property
    def unbound_setup(self) -> tp.Optional[CAUnboundSetup]:
        """Setup of type `CAUnboundSetup` of the unbound cacheable."""
        return self.registry.get_unbound_setup(self.cacheable)

    @property
    def instance_setup(self) -> tp.Optional[CAInstanceSetup]:
        """Setup of type `CAInstanceSetup` of the instance this cacheable is bound to."""
        if self.instance is None or self.instance is _NO_OBJ:
            return None
        return self.registry.get_instance_setup(self.instance)

    def run_func(self, *args, **kwargs) -> tp.Any:
        """Run the setup's function without caching."""
        if self.instance is not None:
            return self.cacheable.func(self.instance, *args, **kwargs)
        return self.cacheable.func(*args, **kwargs)

    def get_args_hash(self, *args, **kwargs) -> int:
        """Get the hash of the passed arguments."""
        if len(args) == 0 and len(kwargs) == 0:
            return hash(())
        return hash_args(
            self.cacheable.func,
            args if self.instance is None else (id(self.instance), *args),
            kwargs,
            ignore_args=self.ignore_args
        )

    def run_func_and_cache(self, *args, **kwargs) -> tp.Any:
        """Run the setup's function and caches the result."""
        args_hash = self.get_args_hash(*args, **kwargs)
        if args_hash in self.cache:
            self._hits += 1
            return self.cache[args_hash]
        self._misses += 1
        if self.max_size is not None and self.max_size <= len(self.cache):
            del self.cache[list(self.cache.keys())[0]]
        self.cache[args_hash] = self.run_func(*args, **kwargs)
        return self.cache[args_hash]

    def run(self, *args, **kwargs) -> tp.Any:
        """Run the setup.

        Runs `CARunSetup.run_func` or `CARunSetup.run_func_and_cache` depending on whether caching is enabled.
        If the passed arguments are not hashable, runs `CARunSetup.run_func`.

        Enables caching if `CARunSetup.use_cache` is True. If caching is disabled globally, can still
        cache if `CARunSetup.whitelist` is True, but not if whitelisting is disabled globally.

        For defaults, see `caching` in `vectorbt._settings.settings`."""
        from vectorbt._settings import settings
        caching_cfg = settings['caching']

        if self.use_cache:
            if caching_cfg['enabled'] or (self.whitelist and not caching_cfg['override_whitelist']):
                try:
                    return self.run_func_and_cache(*args, **kwargs)
                except UnhashableArgsError:
                    pass
        return self.run_func(*args, **kwargs)

    def clear_cache(self) -> None:
        """Clear the cache and reset all metrics."""
        self.cache.clear()
        self._hits = 0
        self._misses = 0

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(" \
               f"cacheable={self.cacheable}, " \
               f"instance={self.instance})"

    @property
    def hash_key(self) -> tuple:
        return self.cacheable, id(self.instance) if self.instance is not None else None


class CAQueryDelegator(CASetupDelegatorMixin):
    """Class that delegates cache management to any setup that matches a query.

    `*args` and `**kwargs` are passed to `CacheableRegistry.match_setups`."""

    def __init__(self, *args, registry: CacheableRegistry = ca_registry, **kwargs) -> None:
        self._args = args
        self._kwargs = kwargs
        self._registry = registry

    @property
    def args(self) -> tp.Args:
        """Arguments."""
        return self._args

    @property
    def kwargs(self) -> tp.Kwargs:
        """Keyword arguments."""
        return self._kwargs

    @property
    def registry(self) -> CacheableRegistry:
        """Registry of type `CacheableRegistry`."""
        return self._registry

    @property
    def child_setups(self) -> tp.Set[CABaseSetup]:
        """Get child setups by matching them using `CacheableRegistry.match_setups`."""
        return self.registry.match_setups(*self.args, **self.kwargs)
