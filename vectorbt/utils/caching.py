# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""Utilities for caching."""

from vectorbt import _typing as tp
from vectorbt.utils.decorators import class_or_instancemethod


class Cacheable:
    """Class that contains cacheable properties and methods.

    Takes care of registering child setups using `vectorbt.ca_registry.CAInstanceSetup`."""

    def __init__(self) -> None:
        self.__class__.get_ca_setup()
        for setup in self.get_ca_setup().unbound_setups:
            setup.cacheable.get_ca_setup(self)

    @class_or_instancemethod
    def get_ca_setup(cls_or_self) -> tp.Union['CAClassSetup', 'CAInstanceSetup']:
        """Get instance setup of type `vectorbt.ca_registry.CAInstanceSetup` if the instance method
        was called and class setup of type `vectorbt.ca_registry.CAClassSetup` otherwise."""
        from vectorbt.ca_registry import CAClassSetup, CAInstanceSetup

        if isinstance(cls_or_self, type):
            return CAClassSetup.get(cls_or_self)
        return CAInstanceSetup.get(cls_or_self)

    def __del__(self) -> None:
        from vectorbt.ca_registry import ca_registry

        for setup in self.get_ca_setup().child_setups:
            ca_registry.deregister_setup(setup)
