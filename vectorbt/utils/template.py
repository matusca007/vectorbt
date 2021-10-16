# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""Utilities for working with templates."""

from copy import copy
from string import Template

from vectorbt import _typing as tp
from vectorbt.utils import checks
from vectorbt.utils.config import set_dict_item, merge_dicts
from vectorbt.utils.parsing import get_func_arg_names
from vectorbt.utils.docs import SafeToStr, prepare_for_doc
from vectorbt.utils.hashing import Hashable


class CustomTemplate(Hashable, SafeToStr):
    """Base class for substituting templates."""

    def __init__(self, template: tp.Any, mapping: tp.Optional[tp.Mapping] = None) -> None:
        self._template = template
        self._mapping = mapping

    @property
    def template(self) -> tp.Any:
        """Template to be processed."""
        return self._template

    @property
    def mapping(self) -> tp.Mapping:
        """Mapping object passed to the initializer."""
        if self._mapping is None:
            return {}
        return self._mapping

    def substitute(self, mapping: tp.Optional[tp.Mapping] = None) -> tp.Any:
        """Abstract method to substitute the template `CustomTemplate.template` using
        the mapping from merging `CustomTemplate.mapping` and `mapping`."""
        raise NotImplementedError

    @property
    def hash_key(self) -> tuple:
        return (
            self.template,
            tuple(self.mapping.items()) if self.mapping is not None else None
        )

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(" \
               f"template=\"{self.template}\", " \
               f"mapping={prepare_for_doc(self.mapping)})"


class Sub(CustomTemplate):
    """Template string to substitute parts with the respective values from `mapping`.

    Always returns a string."""

    def substitute(self, mapping: tp.Optional[tp.Mapping] = None) -> str:
        """Substitute parts of `Sub.template` as a regular template."""
        mapping = merge_dicts(self.mapping, mapping)
        return Template(self.template).substitute(mapping)


class Rep(CustomTemplate):
    """Template string to be replaced with the respective value from `mapping`."""

    def substitute(self, mapping: tp.Optional[tp.Mapping] = None) -> tp.Any:
        """Replace `Rep.template` as a key."""
        mapping = merge_dicts(self.mapping, mapping)
        return mapping[self.template]


class RepEval(CustomTemplate):
    """Template expression to be evaluated with `mapping` used as locals."""

    def substitute(self, mapping: tp.Optional[tp.Mapping] = None) -> tp.Any:
        """Evaluate `RepEval.template` as an expression."""
        mapping = merge_dicts(self.mapping, mapping)
        return eval(self.template, {}, mapping)


class RepFunc(CustomTemplate):
    """Template function to be called with argument names from `mapping`."""

    def substitute(self, mapping: tp.Optional[tp.Mapping] = None) -> tp.Any:
        """Call `RepFunc.template` as a function."""
        mapping = merge_dicts(self.mapping, mapping)
        func_arg_names = get_func_arg_names(self.template)
        func_kwargs = dict()
        for k, v in mapping.items():
            if k in func_arg_names:
                func_kwargs[k] = v
        return self.template(**func_kwargs)


def has_templates(obj: tp.Any) -> tp.Any:
    """Check if the object has any templates."""
    if isinstance(obj, (Template, CustomTemplate)):
        return True
    if isinstance(obj, dict):
        for k, v in obj.items():
            if has_templates(v):
                return True
    if isinstance(obj, (tuple, list, set, frozenset)):
        for v in obj:
            if has_templates(v):
                return True
    return False


def deep_substitute(obj: tp.Any,
                    mapping: tp.Optional[tp.Mapping] = None,
                    strict: bool = False,
                    make_copy: bool = True) -> tp.Any:
    """Traverses the object recursively and, if any template found, substitutes it using a mapping.

    Traverses tuples, lists, dicts and (frozen-)sets. Does not look for templates in keys.

    If `strict` is True, raises an error if processing template fails. Otherwise, returns the original template.

    !!! note
        If the object is deep (such as a dict or a list), creates a copy of it if any template found inside,
        thus loosing the reference to the original. Make sure to do a deep or hybrid copy of the object
        before proceeding for consistent behavior, or disable `make_copy` to override the original in place.

    ## Example

    ```python-repl
    >>> import vectorbt as vbt

    >>> vbt.deep_substitute(vbt.Sub('$key', {'key': 100}))
    100
    >>> vbt.deep_substitute(vbt.Sub('$key', {'key': 100}), {'key': 200})
    200
    >>> vbt.deep_substitute(vbt.Sub('$key$key'), {'key': 100})
    100100
    >>> vbt.deep_substitute(vbt.Rep('key'), {'key': 100})
    100
    >>> vbt.deep_substitute([vbt.Rep('key'), vbt.Sub('$key$key')], {'key': 100})
    [100, '100100']
    >>> vbt.deep_substitute(vbt.RepFunc(lambda key: key == 100), {'key': 100})
    True
    >>> vbt.deep_substitute(vbt.RepEval('key == 100'), {'key': 100})
    True
    >>> vbt.deep_substitute(vbt.RepEval('key == 100', strict=True))
    NameError: name 'key' is not defined
    >>> vbt.deep_substitute(vbt.RepEval('key == 100', strict=False))
    <vectorbt.utils.template.RepEval at 0x7fe3ad2ab668>
    ```"""
    if mapping is None:
        mapping = {}
    if not has_templates(obj):
        return obj
    try:
        if isinstance(obj, (Template, CustomTemplate)):
            return obj.substitute(mapping)
        if isinstance(obj, dict):
            if make_copy:
                obj = copy(obj)
            for k, v in obj.items():
                set_dict_item(obj, k, deep_substitute(v, mapping=mapping, strict=strict), force=True)
            return obj
        if isinstance(obj, list):
            if make_copy:
                obj = copy(obj)
            for i in range(len(obj)):
                obj[i] = deep_substitute(obj[i], mapping=mapping, strict=strict)
            return obj
        if isinstance(obj, (tuple, set, frozenset)):
            result = []
            for o in obj:
                result.append(deep_substitute(o, mapping=mapping, strict=strict))
            if checks.is_namedtuple(obj):
                return type(obj)(*result)
            return type(obj)(result)
    except Exception as e:
        if strict:
            raise e
    return obj
