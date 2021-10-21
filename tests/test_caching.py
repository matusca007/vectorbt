import numpy as np
import pytest
import weakref
import gc

import vectorbt as vbt
from vectorbt.ca_registry import ca_registry, CAQuery, CARunSetup
from vectorbt.utils.caching import Cacheable


# ############# ca_registry.py ############# #

class TestCacheableRegistry:
    def test_ca_query(self):
        class A(Cacheable):
            @vbt.cacheable_property(x=10, y=10)
            def f(self):
                return None

            @vbt.cacheable_property(x=10)
            def f2(self):
                return None

        class B(A):
            @vbt.cacheable_property(x=20, y=10)
            def f2(self):
                return None

            @vbt.cacheable_method(x=20)
            def f3(self):
                return None

        @vbt.cacheable(my_option1=True)
        def f4():
            return None

        a = A()
        b = B()

        def match_query(query):
            matched = set()
            if query.matches_setup(A.f.get_ca_setup()):
                matched.add('A.f')
            if query.matches_setup(A.f.get_ca_setup(a)):
                matched.add('a.f')
            if query.matches_setup(A.f2.get_ca_setup()):
                matched.add('A.f2')
            if query.matches_setup(A.f2.get_ca_setup(a)):
                matched.add('a.f2')
            if query.matches_setup(B.f.get_ca_setup()):
                matched.add('B.f')
            if query.matches_setup(B.f.get_ca_setup(b)):
                matched.add('b.f')
            if query.matches_setup(B.f2.get_ca_setup()):
                matched.add('B.f2')
            if query.matches_setup(B.f2.get_ca_setup(b)):
                matched.add('b.f2')
            if query.matches_setup(B.f3.get_ca_setup()):
                matched.add('B.f3')
            if query.matches_setup(B.f3.get_ca_setup(b)):
                matched.add('b.f3')
            if query.matches_setup(f4.get_ca_setup()):
                matched.add('f4')
            if query.matches_setup(A.get_ca_setup()):
                matched.add('A')
            if query.matches_setup(B.get_ca_setup()):
                matched.add('B')
            if query.matches_setup(a.get_ca_setup()):
                matched.add('a')
            if query.matches_setup(b.get_ca_setup()):
                matched.add('b')
            return matched

        assert match_query(CAQuery(cacheable=A.f)) == {
            'A.f', 'B.f', 'a.f', 'b.f'
        }
        assert match_query(CAQuery(cacheable=A.f2)) == {
            'A.f2', 'a.f2'
        }
        assert match_query(CAQuery(cacheable=B.f2)) == {
            'B.f2', 'b.f2'
        }
        assert match_query(CAQuery(cacheable=B.f3)) == {
            'B.f3', 'b.f3'
        }
        assert match_query(CAQuery(cacheable=f4)) == {
            'f4'
        }
        assert match_query(CAQuery(cacheable=A.f.func)) == {
            'A.f', 'B.f', 'a.f', 'b.f'
        }
        assert match_query(CAQuery(cacheable=A.f2.func)) == {
            'A.f2', 'a.f2'
        }
        assert match_query(CAQuery(cacheable=B.f2.func)) == {
            'B.f2', 'b.f2'
        }
        assert match_query(CAQuery(cacheable=B.f3.func)) == {
            'B.f3', 'b.f3'
        }
        assert match_query(CAQuery(cacheable=f4.func)) == {
            'f4'
        }
        assert match_query(CAQuery(cacheable='f')) == {
            'A.f', 'B.f', 'a.f', 'b.f'
        }
        assert match_query(CAQuery(cacheable='f2')) == {
            'A.f2', 'B.f2', 'a.f2', 'b.f2'
        }
        assert match_query(CAQuery(cacheable='f3')) == {
            'B.f3', 'b.f3'
        }
        assert match_query(CAQuery(cacheable='f4')) == {
            'f4'
        }
        assert match_query(CAQuery(cacheable=vbt.Regex('(f2|f3)'))) == {
            'A.f2', 'B.f2', 'B.f3', 'a.f2', 'b.f2', 'b.f3'
        }
        assert match_query(CAQuery(instance=a)) == {
            'a', 'a.f', 'a.f2'
        }
        assert match_query(CAQuery(instance=b)) == {
            'b', 'b.f', 'b.f2', 'b.f3'
        }
        assert match_query(CAQuery(cls=A)) == {
            'A', 'a', 'a.f', 'a.f2'
        }
        assert match_query(CAQuery(cls=B)) == {
            'B', 'b', 'b.f', 'b.f2', 'b.f3'
        }
        assert match_query(CAQuery(cls='A')) == {
            'A', 'a', 'a.f', 'a.f2'
        }
        assert match_query(CAQuery(cls='B')) == {
            'B', 'b', 'b.f', 'b.f2', 'b.f3'
        }
        assert match_query(CAQuery(cls=('A', 'B'))) == {
            'A', 'B', 'a', 'a.f', 'a.f2', 'b', 'b.f', 'b.f2', 'b.f3'
        }
        assert match_query(CAQuery(cls=vbt.Regex('(A|B)'))) == {
            'A', 'B', 'a', 'a.f', 'a.f2', 'b', 'b.f', 'b.f2', 'b.f3'
        }
        assert match_query(CAQuery(base_cls=A)) == {
            'A', 'B', 'a', 'a.f', 'a.f2', 'b', 'b.f', 'b.f2', 'b.f3'
        }
        assert match_query(CAQuery(base_cls=B)) == {
            'B', 'b', 'b.f', 'b.f2', 'b.f3'
        }
        assert match_query(CAQuery(base_cls='A')) == {
            'A', 'B', 'a', 'a.f', 'a.f2', 'b', 'b.f', 'b.f2', 'b.f3'
        }
        assert match_query(CAQuery(base_cls='B')) == {
            'B', 'b', 'b.f', 'b.f2', 'b.f3'
        }
        assert match_query(CAQuery(base_cls=('A', 'B'))) == {
            'A', 'B', 'a', 'a.f', 'a.f2', 'b', 'b.f', 'b.f2', 'b.f3'
        }
        assert match_query(CAQuery(base_cls=vbt.Regex('(A|B)'))) == {
            'A', 'B', 'a', 'a.f', 'a.f2', 'b', 'b.f', 'b.f2', 'b.f3'
        }
        assert match_query(CAQuery(options=dict(x=10))) == {
            'A.f', 'A.f2', 'B.f', 'a.f', 'a.f2', 'b.f'
        }
        assert match_query(CAQuery(options=dict(y=10))) == {
            'A.f', 'B.f', 'B.f2', 'a.f', 'b.f', 'b.f2'
        }
        assert match_query(CAQuery(options=dict(x=20, y=10))) == {
            'B.f2', 'b.f2'
        }

        assert CAQuery(options=dict(
            cacheable=A.f,
            instance=a,
            cls=A,
            base_cls='A',
            options=dict(my_option1=True)
        )) == CAQuery(options=dict(
            cacheable=A.f,
            instance=a,
            cls=A,
            base_cls='A',
            options=dict(my_option1=True)
        ))

        assert CAQuery.parse(None) == CAQuery()
        assert CAQuery.parse(A.get_ca_setup()) == CAQuery(base_cls=A)
        assert CAQuery.parse(a.get_ca_setup()) == CAQuery(instance=a)
        assert CAQuery.parse(A.f.get_ca_setup()) == CAQuery(cacheable=A.f)
        assert CAQuery.parse(A.f.get_ca_setup(a)) == CAQuery(cacheable=A.f, instance=a)
        assert CAQuery.parse(A.f) == CAQuery(cacheable=A.f)
        assert CAQuery.parse(B.f) == CAQuery(cacheable=B.f)
        assert CAQuery.parse(B.f.func) == CAQuery(cacheable=B.f.func)
        assert CAQuery.parse('f') == CAQuery(cacheable='f')
        assert CAQuery.parse('A.f') == CAQuery(cacheable='f', base_cls='A')
        assert CAQuery.parse('A.f', use_base_cls=False) == CAQuery(cacheable='f', cls='A')
        assert CAQuery.parse('A') == CAQuery(base_cls='A')
        assert CAQuery.parse('A', use_base_cls=False) == CAQuery(cls='A')
        assert CAQuery.parse(vbt.Regex('A')) == CAQuery(base_cls=vbt.Regex('A'))
        assert CAQuery.parse(vbt.Regex('A'), use_base_cls=False) == CAQuery(cls=vbt.Regex('A'))
        assert CAQuery.parse(A) == CAQuery(base_cls=A)
        assert CAQuery.parse(A, use_base_cls=False) == CAQuery(cls=A)
        assert CAQuery.parse((A, B)) == CAQuery(base_cls=(A, B))
        assert CAQuery.parse((A, B), use_base_cls=False) == CAQuery(cls=(A, B))
        assert CAQuery.parse(dict(my_option1=True)) == CAQuery(options=dict(my_option1=True))
        assert CAQuery.parse(a) == CAQuery(instance=a)

    def test_ca_run_setup(self):
        @vbt.cacheable(max_size=2)
        def f(a, b, c=3):
            return a + b + c

        setup = f.get_ca_setup()
        with pytest.raises(Exception):
            _ = CARunSetup(f)
        assert setup.unbound_setup is None
        assert setup.instance_setup is None
        assert setup is CARunSetup.get(f)

        assert setup.run_func(10, 20, c=30) == 60
        assert setup.run_func_and_cache(10, 20, c=30) == 60
        assert setup.misses == 1
        assert setup.hits == 0
        assert len(setup.cache) == 1
        assert setup.cache[hash((('a', 10), ('b', 20), ('c', 30)))] == 60

        assert setup.run_func_and_cache(10, 20, c=30) == 60
        assert setup.misses == 1
        assert setup.hits == 1
        assert len(setup.cache) == 1
        assert setup.cache[hash((('a', 10), ('b', 20), ('c', 30)))] == 60

        assert setup.run_func_and_cache(10, 20, c=40) == 70
        assert setup.misses == 2
        assert setup.hits == 1
        assert len(setup.cache) == 2
        assert setup.cache[hash((('a', 10), ('b', 20), ('c', 40)))] == 70

        assert setup.run_func_and_cache(10, 20, c=50) == 80
        assert setup.misses == 3
        assert setup.hits == 1
        assert len(setup.cache) == 2
        assert setup.cache[hash((('a', 10), ('b', 20), ('c', 40)))] == 70
        assert setup.cache[hash((('a', 10), ('b', 20), ('c', 50)))] == 80

        setup.clear_cache()
        assert setup.misses == 0
        assert setup.hits == 0
        assert len(setup.cache) == 0

        assert setup.run(10, 20, c=30) == 60
        assert len(setup.cache) == 0

        setup.enable_caching()
        assert setup.run(10, 20, c=30) == 60
        assert len(setup.cache) == 1

        setup.disable_caching(clear_cache=False)
        assert setup.run(10, 20, c=30) == 60
        assert len(setup.cache) == 1

        setup.disable_caching(clear_cache=True)
        assert len(setup.cache) == 0

        setup.enable_caching()
        vbt.settings['caching']['enabled'] = False
        assert setup.run(10, 20, c=30) == 60
        assert len(setup.cache) == 0

        setup.enable_whitelist()
        assert setup.run(10, 20, c=30) == 60
        assert len(setup.cache) == 1

        vbt.settings['caching']['override_whitelist'] = True
        assert setup.run(10, 20, c=30) == 60
        assert len(setup.cache) == 1

        setup.clear_cache()
        vbt.settings['caching'].reset()

        np.testing.assert_array_equal(
            setup.run(10, 20, c=np.array([1, 2, 3])),
            np.array([31, 32, 33])
        )
        assert len(setup.cache) == 0

        @vbt.cached(ignore_args=['c'])
        def f(a, b, c=3):
            return a + b + c

        setup = f.get_ca_setup()
        np.testing.assert_array_equal(
            setup.run(10, 20, c=np.array([1, 2, 3])),
            np.array([31, 32, 33])
        )
        assert len(setup.cache) == 1

        class A(Cacheable):
            @vbt.cacheable_property
            def f(self):
                return 10

        with pytest.raises(Exception):
            _ = CARunSetup.get(A.f)

        a = A()

        setup = A.f.get_ca_setup(a)
        assert setup.unbound_setup is A.f.get_ca_setup()
        assert setup.instance_setup is a.get_ca_setup()
        assert setup.run() == 10
        assert len(setup.cache) == 0

        setup.enable_caching()
        assert setup.run() == 10
        assert len(setup.cache) == 1
        assert setup.run() == 10
        assert len(setup.cache) == 1

        class B(Cacheable):
            @vbt.cacheable_method
            def f(self, a, b, c=30):
                return a + b + c

        b = B()

        setup = B.f.get_ca_setup(b)
        assert setup.unbound_setup is B.f.get_ca_setup()
        assert setup.instance_setup is b.get_ca_setup()
        assert setup.run(10, 20, c=30) == 60
        assert len(setup.cache) == 0

        setup.enable_caching()
        assert setup.run(10, 20, c=30) == 60
        assert len(setup.cache) == 1
        assert setup.run(10, 20, c=30) == 60
        assert len(setup.cache) == 1

    def test_ca_unbound_setup(self):
        class A(Cacheable):
            @vbt.cached_method(whitelist=True)
            def f(self, a, b, c=30):
                return a + b + c

        class B(A):
            @vbt.cacheable_method(whitelist=False)
            def f2(self, a, b, c=30):
                return a + b + c

        a = A()
        b = B()

        assert A.f.get_ca_setup() is B.f.get_ca_setup()
        assert A.f.get_ca_setup() is not B.f2.get_ca_setup()
        assert A.f.get_ca_setup(a) is not B.f.get_ca_setup(b)
        assert A.f.get_ca_setup(a) is not B.f2.get_ca_setup(b)
        assert A.f.get_ca_setup(a).unbound_setup is B.f.get_ca_setup(b).unbound_setup
        assert A.f.get_ca_setup(a).unbound_setup is not B.f2.get_ca_setup(b).unbound_setup

        assert A.f.get_ca_setup().run_setups == {
            A.f.get_ca_setup(a),
            B.f.get_ca_setup(b)
        }
        assert B.f.get_ca_setup().run_setups == {
            A.f.get_ca_setup(a),
            B.f.get_ca_setup(b)
        }
        assert B.f2.get_ca_setup().run_setups == {
            B.f2.get_ca_setup(b)
        }

        unbound_setup1 = A.f.get_ca_setup()
        unbound_setup2 = B.f2.get_ca_setup()
        run_setup1 = A.f.get_ca_setup(a)
        run_setup2 = B.f.get_ca_setup(b)
        run_setup3 = B.f2.get_ca_setup(b)

        assert unbound_setup1.use_cache
        assert unbound_setup1.whitelist
        assert not unbound_setup2.use_cache
        assert not unbound_setup2.whitelist
        assert run_setup1.use_cache
        assert run_setup1.whitelist
        assert run_setup2.use_cache
        assert run_setup2.whitelist
        assert not run_setup3.use_cache
        assert not run_setup3.whitelist

        unbound_setup1.disable_whitelist()
        unbound_setup2.disable_whitelist()
        assert not unbound_setup1.whitelist
        assert not unbound_setup2.whitelist
        assert not run_setup1.whitelist
        assert not run_setup2.whitelist
        assert not run_setup3.whitelist

        unbound_setup1.enable_whitelist()
        unbound_setup2.enable_whitelist()
        assert unbound_setup1.whitelist
        assert unbound_setup2.whitelist
        assert run_setup1.whitelist
        assert run_setup2.whitelist
        assert run_setup3.whitelist

        unbound_setup1.disable_caching()
        unbound_setup2.disable_caching()
        assert not unbound_setup1.use_cache
        assert not unbound_setup2.use_cache
        assert not run_setup1.use_cache
        assert not run_setup2.use_cache
        assert not run_setup3.use_cache

        unbound_setup1.enable_caching()
        unbound_setup2.enable_caching()
        assert unbound_setup1.use_cache
        assert unbound_setup2.use_cache
        assert run_setup1.use_cache
        assert run_setup2.use_cache
        assert run_setup3.use_cache

        assert run_setup1.run(10, 20, c=30) == 60
        assert len(run_setup1.cache) == 1
        assert run_setup2.run(10, 20, c=30) == 60
        assert len(run_setup2.cache) == 1
        assert run_setup3.run(10, 20, c=30) == 60
        assert len(run_setup3.cache) == 1
        unbound_setup1.clear_cache()
        unbound_setup2.clear_cache()
        assert len(run_setup1.cache) == 0
        assert len(run_setup2.cache) == 0
        assert len(run_setup3.cache) == 0

        b2 = B()
        run_setup4 = B.f.get_ca_setup(b2)
        run_setup5 = B.f2.get_ca_setup(b2)
        assert run_setup4.use_cache
        assert run_setup4.whitelist
        assert run_setup5.use_cache
        assert run_setup5.whitelist

    def test_ca_instance_setup(self):
        class A(Cacheable):
            @vbt.cached_method(whitelist=True)
            def f(self, a, b, c=30):
                return a + b + c

        class B(A):
            @vbt.cacheable_method(whitelist=False)
            def f2(self, a, b, c=30):
                return a + b + c

        a = A()
        b = B()

        assert a.get_ca_setup() is not b.get_ca_setup()

        assert a.get_ca_setup().class_setup is A.get_ca_setup()
        assert b.get_ca_setup().class_setup is B.get_ca_setup()
        assert a.get_ca_setup().unbound_setups == {
            A.f.get_ca_setup()
        }
        assert b.get_ca_setup().unbound_setups == {
            B.f.get_ca_setup(),
            B.f2.get_ca_setup()
        }
        assert a.get_ca_setup().run_setups == {
            A.f.get_ca_setup(a)
        }
        assert b.get_ca_setup().run_setups == {
            B.f.get_ca_setup(b),
            B.f2.get_ca_setup(b)
        }

        instance_setup1 = a.get_ca_setup()
        instance_setup2 = b.get_ca_setup()
        run_setup1 = A.f.get_ca_setup(a)
        run_setup2 = B.f.get_ca_setup(b)
        run_setup3 = B.f2.get_ca_setup(b)

        assert instance_setup1.use_cache is None
        assert instance_setup1.whitelist is None
        assert instance_setup2.use_cache is None
        assert instance_setup2.whitelist is None
        assert run_setup1.use_cache
        assert run_setup1.whitelist
        assert run_setup2.use_cache
        assert run_setup2.whitelist
        assert not run_setup3.use_cache
        assert not run_setup3.whitelist

        instance_setup1.disable_whitelist()
        instance_setup2.disable_whitelist()
        assert not instance_setup1.whitelist
        assert not instance_setup2.whitelist
        assert not run_setup1.whitelist
        assert not run_setup2.whitelist
        assert not run_setup3.whitelist

        instance_setup1.enable_whitelist()
        instance_setup2.enable_whitelist()
        assert instance_setup1.whitelist
        assert instance_setup2.whitelist
        assert run_setup1.whitelist
        assert run_setup2.whitelist
        assert run_setup3.whitelist

        instance_setup1.disable_caching()
        instance_setup2.disable_caching()
        assert not instance_setup1.use_cache
        assert not instance_setup2.use_cache
        assert not run_setup1.use_cache
        assert not run_setup2.use_cache
        assert not run_setup3.use_cache

        instance_setup1.enable_caching()
        instance_setup2.enable_caching()
        assert instance_setup1.use_cache
        assert instance_setup2.use_cache
        assert run_setup1.use_cache
        assert run_setup2.use_cache
        assert run_setup3.use_cache

        assert run_setup1.run(10, 20, c=30) == 60
        assert len(run_setup1.cache) == 1
        assert run_setup2.run(10, 20, c=30) == 60
        assert len(run_setup2.cache) == 1
        assert run_setup3.run(10, 20, c=30) == 60
        assert len(run_setup3.cache) == 1
        instance_setup1.clear_cache()
        instance_setup2.clear_cache()
        assert len(run_setup1.cache) == 0
        assert len(run_setup2.cache) == 0
        assert len(run_setup3.cache) == 0

        B.get_ca_setup().disable_caching()
        B.get_ca_setup().disable_whitelist()
        b2 = B()
        instance_setup3 = b2.get_ca_setup()
        run_setup4 = B.f.get_ca_setup(b2)
        run_setup5 = B.f2.get_ca_setup(b2)
        assert not instance_setup3.use_cache
        assert not instance_setup3.whitelist
        assert not run_setup4.use_cache
        assert not run_setup4.whitelist
        assert not run_setup5.use_cache
        assert not run_setup5.whitelist

    def test_ca_class_setup(self):
        class A(Cacheable):
            @vbt.cacheable_method
            def f(self, a, b, c=30):
                return a + b + c

        class B(A):
            @vbt.cacheable_method
            def f2(self, a, b, c=30):
                return a + b + c

        class C(Cacheable):
            @vbt.cacheable_method
            def f3(self, a, b, c=30):
                return a + b + c

        a = A()
        b = B()
        c = C()

        assert A.get_ca_setup() is not B.get_ca_setup()
        assert A.get_ca_setup() is not C.get_ca_setup()

        class_setup1 = A.get_ca_setup()
        class_setup2 = B.get_ca_setup()
        class_setup3 = C.get_ca_setup()
        assert class_setup1.lazy_superclass_setups == set()
        assert class_setup2.lazy_superclass_setups == {
            class_setup1
        }
        assert class_setup3.lazy_superclass_setups == set()
        assert class_setup1.lazy_subclass_setups == {
            class_setup2
        }
        assert class_setup2.lazy_subclass_setups == set()
        assert class_setup3.lazy_subclass_setups == set()
        assert class_setup1.unbound_setups == {
            A.f.get_ca_setup()
        }
        assert class_setup2.unbound_setups == {
            A.f.get_ca_setup(),
            B.f2.get_ca_setup()
        }
        assert class_setup3.unbound_setups == {
            C.f3.get_ca_setup()
        }
        assert class_setup1.instance_setups == {
            a.get_ca_setup()
        }
        assert class_setup2.instance_setups == {
            b.get_ca_setup()
        }
        assert class_setup3.instance_setups == {
            c.get_ca_setup()
        }
        assert class_setup1.child_setups == {
            a.get_ca_setup(),
            B.get_ca_setup()
        }
        assert class_setup2.child_setups == {
            b.get_ca_setup()
        }
        assert class_setup3.child_setups == {
            c.get_ca_setup()
        }

        class_setup1 = A.get_ca_setup()
        class_setup2 = B.get_ca_setup()
        class_setup3 = C.get_ca_setup()
        instance_setup1 = a.get_ca_setup()
        instance_setup2 = b.get_ca_setup()
        instance_setup3 = c.get_ca_setup()

        assert class_setup1.use_cache is None
        assert class_setup1.whitelist is None
        assert class_setup2.use_cache is None
        assert class_setup2.whitelist is None
        assert class_setup3.use_cache is None
        assert class_setup3.whitelist is None
        assert instance_setup1.use_cache is None
        assert instance_setup1.whitelist is None
        assert instance_setup2.use_cache is None
        assert instance_setup2.whitelist is None
        assert instance_setup3.use_cache is None
        assert instance_setup3.whitelist is None

        class_setup1.enable_whitelist()
        assert class_setup1.whitelist
        assert class_setup2.whitelist
        assert class_setup3.whitelist is None
        assert instance_setup1.whitelist
        assert instance_setup2.whitelist
        assert instance_setup3.whitelist is None

        class_setup1.enable_caching()
        assert class_setup1.use_cache
        assert class_setup2.use_cache
        assert class_setup3.use_cache is None
        assert instance_setup1.use_cache
        assert instance_setup2.use_cache
        assert instance_setup3.use_cache is None

        class_setup2.disable_whitelist()
        assert class_setup1.whitelist
        assert not class_setup2.whitelist
        assert class_setup3.whitelist is None
        assert instance_setup1.whitelist
        assert not instance_setup2.whitelist
        assert instance_setup3.whitelist is None

        class_setup2.disable_caching()
        assert class_setup1.use_cache
        assert not class_setup2.use_cache
        assert class_setup3.use_cache is None
        assert instance_setup1.use_cache
        assert not instance_setup2.use_cache
        assert instance_setup3.use_cache is None

        class D(A):
            @vbt.cacheable_method
            def f4(self, a, b, c=30):
                return a + b + c

        d = D()
        class_setup4 = D.get_ca_setup()
        instance_setup4 = d.get_ca_setup()

        assert class_setup4.use_cache
        assert class_setup4.whitelist
        assert instance_setup4.use_cache
        assert instance_setup4.whitelist

        class E(B):
            @vbt.cacheable_method
            def f5(self, a, b, c=30):
                return a + b + c

        e = E()
        class_setup5 = E.get_ca_setup()
        instance_setup5 = e.get_ca_setup()

        assert not class_setup5.use_cache
        assert not class_setup5.whitelist
        assert not instance_setup5.use_cache
        assert not instance_setup5.whitelist

    def test_match_setups(self):
        class A(Cacheable):
            @vbt.cacheable_property
            def f_test(self):
                return 10

        class B(A):
            @vbt.cacheable_method
            def f2_test(self, a, b, c=30):
                return a + b + c

        @vbt.cacheable
        def f3_test(a, b, c=30):
            return a + b + c

        a = A()
        b = B()

        queries = [
            A.get_ca_setup().query,
            B.get_ca_setup().query,
            A.f_test.get_ca_setup().query,
            B.f2_test.get_ca_setup().query,
            f3_test.get_ca_setup().query
        ]
        assert ca_registry.match_setups(queries) == {
            A.get_ca_setup(),
            B.get_ca_setup(),
            a.get_ca_setup(),
            b.get_ca_setup(),
            A.f_test.get_ca_setup(),
            B.f2_test.get_ca_setup(),
            A.f_test.get_ca_setup(a),
            B.f_test.get_ca_setup(b),
            B.f2_test.get_ca_setup(b),
            f3_test.get_ca_setup()
        }
        assert ca_registry.match_setups(queries, kind='runnable') == {
            A.f_test.get_ca_setup(a),
            B.f_test.get_ca_setup(b),
            B.f2_test.get_ca_setup(b),
            f3_test.get_ca_setup()
        }
        assert ca_registry.match_setups(queries, kind='unbound') == {
            A.f_test.get_ca_setup(),
            B.f2_test.get_ca_setup()
        }
        assert ca_registry.match_setups(queries, kind='instance') == {
            a.get_ca_setup(),
            b.get_ca_setup()
        }
        assert ca_registry.match_setups(queries, kind='class') == {
            A.get_ca_setup(),
            B.get_ca_setup()
        }
        assert ca_registry.match_setups(queries, kind=('class', 'instance')) == {
            A.get_ca_setup(),
            B.get_ca_setup(),
            a.get_ca_setup(),
            b.get_ca_setup()
        }
        assert ca_registry.match_setups(queries, kind=('class', 'instance'), exclude=b.get_ca_setup()) == {
            A.get_ca_setup(),
            B.get_ca_setup(),
            a.get_ca_setup()
        }
        assert ca_registry.match_setups(queries, collapse=True) == {
            A.get_ca_setup(),
            A.f_test.get_ca_setup(),
            B.f2_test.get_ca_setup(),
            f3_test.get_ca_setup()
        }
        assert ca_registry.match_setups(queries, collapse=True, kind='instance') == {
            a.get_ca_setup(),
            b.get_ca_setup()
        }
        assert ca_registry.match_setups(queries, collapse=True, kind=('instance', 'runnable')) == {
            a.get_ca_setup(),
            b.get_ca_setup(),
            f3_test.get_ca_setup()
        }
        assert ca_registry.match_setups(
            queries, collapse=True, kind=('instance', 'runnable'),
            exclude=a.get_ca_setup()) == {
                   b.get_ca_setup(),
                   f3_test.get_ca_setup()
               }
        assert ca_registry.match_setups(
            queries, collapse=True, kind=('instance', 'runnable'),
            exclude=a.get_ca_setup(), exclude_children=False) == {
                   b.get_ca_setup(),
                   A.f_test.get_ca_setup(a),
                   f3_test.get_ca_setup()
               }

    def test_gc(self):
        class A(Cacheable):
            @vbt.cacheable_property
            def f(self):
                return 10

        a = A()
        assert ca_registry.match_setups(CAQuery(cls=A)) == {
            A.get_ca_setup(),
            a.get_ca_setup(),
            A.f.get_ca_setup(a)
        }
        a_ref = weakref.ref(a)
        del a
        gc.collect()
        assert a_ref() is None
        assert ca_registry.match_setups(CAQuery(cls=A)) == {
            A.get_ca_setup()
        }

        class B(Cacheable):
            @vbt.cacheable_method
            def f(self):
                return 10

        b = B()
        assert ca_registry.match_setups(CAQuery(cls=B)) == {
            B.get_ca_setup(),
            B.f.get_ca_setup(b),
            b.get_ca_setup()
        }
        b_ref = weakref.ref(b)
        del b
        gc.collect()
        assert b_ref() is None
        assert ca_registry.match_setups(CAQuery(cls=B)) == {
            B.get_ca_setup()
        }
