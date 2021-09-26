# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""Utilities for parsing."""

import inspect
import ast
import sys

from vectorbt import _typing as tp


def get_func_kwargs(func: tp.Callable) -> dict:
    """Get keyword arguments with defaults of a function."""
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


def get_func_arg_names(func: tp.Callable, arg_kind: tp.Optional[tp.MaybeTuple[int]] = None) -> tp.List[str]:
    """Get argument names of a function."""
    signature = inspect.signature(func)
    if arg_kind is not None and isinstance(arg_kind, int):
        arg_kind = (arg_kind,)
    if arg_kind is None:
        return [
            p.name for p in signature.parameters.values()
            if p.kind != p.VAR_POSITIONAL and p.kind != p.VAR_KEYWORD
        ]
    return [
        p.name for p in signature.parameters.values()
        if p.kind in arg_kind
    ]


def annotate_args(func, *args, **kwargs):
    """Annotate arguments and keyword arguments using the function's signature."""
    signature = inspect.signature(func)
    signature.bind(*args, **kwargs)
    new_args = ()
    arg_i = 0

    for p in signature.parameters.values():
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY):
            if p.kind != p.KEYWORD_ONLY and arg_i < len(args):
                # Either positional-only arguments or keyword arguments passed as such
                arg = args[arg_i]
                arg_spec = dict(name=p.name, kind=p.kind, default=p.default, arg=arg)
                new_args += (arg_spec,)
            else:
                # Either keyword-only arguments or positional arguments passed as such
                if p.name in kwargs:
                    arg = kwargs.pop(p.name)
                    arg_spec = dict(name=p.name, kind=p.kind, default=p.default, arg=arg)
                else:
                    arg_spec = dict(name=p.name, kind=p.kind, default=p.default)
                new_args += (arg_spec,)
            arg_i += 1
        elif p.kind == p.VAR_POSITIONAL:
            # *args
            arg_spec = dict(name=p.name, kind=p.kind, arg=args[arg_i:])
            new_args += (arg_spec,)
        else:
            # **kwargs
            arg_spec = dict(name=p.name, kind=p.kind, arg=kwargs)
            new_args += (arg_spec,)
    return new_args


def get_ex_var_names(expression: str) -> tp.List[str]:
    """Get variable names listed in the expression."""
    return [node.id for node in ast.walk(ast.parse(expression)) if type(node) is ast.Name]


def get_context_vars(var_names: tp.Iterable[str],
                     frames_back: int = 0,
                     local_dict: tp.Optional[tp.Mapping] = None,
                     global_dict: tp.Optional[tp.Mapping] = None) -> tp.List[tp.Any]:
    """Get variables from the local/global context."""
    call_frame = sys._getframe(frames_back + 1)
    clear_local_dict = False
    if local_dict is None:
        local_dict = call_frame.f_locals
        clear_local_dict = True
    try:
        frame_globals = call_frame.f_globals
        if global_dict is None:
            global_dict = frame_globals
        clear_local_dict = clear_local_dict and frame_globals is not local_dict
        args = []
        for var_name in var_names:
            try:
                a = local_dict[var_name]
            except KeyError:
                a = global_dict[var_name]
            args.append(a)
    finally:
        # See https://github.com/pydata/numexpr/issues/310
        if clear_local_dict:
            local_dict.clear()
    return args
