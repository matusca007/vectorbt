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


ann_argsT = tp.Dict[str, tp.Kwargs]


def annotate_args(func: tp.Callable, *args, **kwargs) -> ann_argsT:
    """Annotate arguments and keyword arguments using the function's signature."""
    signature = inspect.signature(func)
    signature.bind(*args, **kwargs)
    ann_args = dict()
    arg_i = 0

    for p in signature.parameters.values():
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY):
            if p.kind != p.KEYWORD_ONLY and arg_i < len(args):
                # Either positional-only arguments or keyword arguments passed as such
                ann_args[p.name] = dict(kind=p.kind, value=args[arg_i])
            else:
                # Either keyword-only arguments or positional arguments passed as such
                if p.name in kwargs:
                    ann_args[p.name] = dict(kind=p.kind, value=kwargs.pop(p.name))
                else:
                    ann_args[p.name] = dict(kind=p.kind, value=p.default)
            arg_i += 1
        elif p.kind == p.VAR_POSITIONAL:
            # *args
            ann_args[p.name] = dict(kind=p.kind, value=args[arg_i:])
        else:
            # **kwargs
            ann_args[p.name] = dict(kind=p.kind, value=kwargs)
    return ann_args


def get_from_ann_args(ann_args: ann_argsT, i: tp.Optional[int] = None, name: tp.Optional[str] = None) -> tp.Any:
    """Get argument from annotated arguments using its position or name.

    The position can stretch over any variable argument."""
    if (i is None and name is None) or (i is not None and name is not None):
        raise ValueError("Either i or name must be provided")
    flat_args = []
    args_by_name = {}
    for arg_name, ann_arg in ann_args.items():
        if ann_arg['kind'] == inspect.Parameter.VAR_POSITIONAL:
            flat_args.extend(ann_arg['value'])
        elif ann_arg['kind'] == inspect.Parameter.VAR_KEYWORD:
            for var_arg_name, var_value in ann_arg['value'].items():
                flat_args.append(var_value)
                args_by_name[var_arg_name] = var_value
        else:
            flat_args.append(ann_arg['value'])
            args_by_name[arg_name] = ann_arg['value']
    if name is not None:
        return args_by_name[name]
    return flat_args[i]


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
