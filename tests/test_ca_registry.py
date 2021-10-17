import pytest

import vectorbt as vbt
from vectorbt.ca_registry import ca_registry


# ############# ca_registry.py ############# #

class TestCacheableRegistry:
    def test_ca_query(self):
        class A:
            @vbt.cacheable_property(my_option1=True, my_option2=False)
            def f(self):
                return None

        class B(A):
            @vbt.cacheable_method(my_option1=False, my_option2=True)
            def f(self):
                return None

        @vbt.cacheable(my_option1=True)
        def f():
            return None

        a = A()
        b = B()

        def match_query(query):
            matched = []
            if query.rank_setup(A.f.get_setup(instance=a)) > -1:
                matched.append('A.f')
            if query.rank_setup(B.f.get_setup(instance=b)) > -1:
                matched.append('B.f')
            if query.rank_setup(f.get_setup()) > -1:
                matched.append('f')
            return matched

        assert match_query(vbt.CAQuery(cacheable=A.f)) == ['A.f']
        assert match_query(vbt.CAQuery(cacheable=B.f)) == ['B.f']
        assert match_query(vbt.CAQuery(cacheable=A.f.func)) == ['A.f']
        assert match_query(vbt.CAQuery(cacheable=B.f.func)) == ['B.f']
        assert match_query(vbt.CAQuery(cacheable='f')) == ['A.f', 'B.f', 'f']
        assert match_query(vbt.CAQuery(instance=a)) == ['A.f']
        assert match_query(vbt.CAQuery(instance=b)) == ['B.f']
        assert match_query(vbt.CAQuery(cls=A)) == ['A.f']
        assert match_query(vbt.CAQuery(cls=B)) == ['B.f']
        assert match_query(vbt.CAQuery(cls='A')) == ['A.f']
        assert match_query(vbt.CAQuery(cls='B')) == ['B.f']
        assert match_query(vbt.CAQuery(cls=vbt.Regex('[A-B]'))) == ['A.f', 'B.f']
        assert match_query(vbt.CAQuery(base_cls='A')) == ['A.f', 'B.f']
        assert match_query(vbt.CAQuery(base_cls='B')) == ['B.f']
        assert match_query(vbt.CAQuery(base_cls=('A', 'B'))) == ['A.f', 'B.f']
        assert match_query(vbt.CAQuery(base_cls=vbt.Regex('[A-B]'))) == ['A.f', 'B.f']
        assert match_query(vbt.CAQuery(options=dict(my_option1=True, my_option2=False))) == ['A.f']
        assert match_query(vbt.CAQuery(options=dict(my_option1=True))) == ['A.f', 'f']
        assert match_query(vbt.CAQuery(options=dict(my_option2=True))) == ['B.f']

        setup = A.f.get_setup(instance=a)
        assert vbt.CAQuery(instance=a, cacheable=B.f).rank_setup(setup) == -1
        assert vbt.CAQuery(instance=a, cacheable=A.f).rank_setup(setup) == 35
        assert vbt.CAQuery(instance=a, options=dict(my_option1=True)).rank_setup(setup) == 31
        assert vbt.CAQuery(instance=a).rank_setup(setup) == 30
        assert vbt.CAQuery(cls=A, cacheable=A.f).rank_setup(setup) == 25
        assert vbt.CAQuery(cls=A, options=dict(my_option1=True)).rank_setup(setup) == 21
        assert vbt.CAQuery(cls=A).rank_setup(setup) == 20
        assert vbt.CAQuery(base_cls=A, cacheable=A.f).rank_setup(setup) == 15
        assert vbt.CAQuery(base_cls=A, options=dict(my_option1=True)).rank_setup(setup) == 11
        assert vbt.CAQuery(base_cls=A).rank_setup(setup) == 10
        assert vbt.CAQuery(cacheable=A.f, options=dict(my_option1=True)).rank_setup(setup) == 9
        assert vbt.CAQuery(cacheable=A.f).rank_setup(setup) == 5
        assert vbt.CAQuery(options=dict(my_option1=True)).rank_setup(setup) == 1
        assert vbt.CAQuery().rank_setup(setup) == 0

        assert vbt.CAQuery(options=dict(
            cacheable=A.f,
            instance=a,
            cls=A,
            base_cls='A',
            options=dict(my_option1=True)
        )) == vbt.CAQuery(options=dict(
            cacheable=A.f,
            instance=a,
            cls=A,
            base_cls='A',
            options=dict(my_option1=True)
        ))

        assert vbt.CAQuery.parse(None) == vbt.CAQuery()
        assert vbt.CAQuery.parse(A.f) == vbt.CAQuery(cacheable=A.f)
        assert vbt.CAQuery.parse(B.f) == vbt.CAQuery(cacheable=B.f)
        assert vbt.CAQuery.parse(B.f.func) == vbt.CAQuery(cacheable=B.f.func)
        assert vbt.CAQuery.parse('f') == vbt.CAQuery(cacheable='f')
        assert vbt.CAQuery.parse('A.f') == vbt.CAQuery(cacheable='f', base_cls='A')
        assert vbt.CAQuery.parse('A.f', use_base_cls=False) == vbt.CAQuery(cacheable='f', cls='A')
        assert vbt.CAQuery.parse('A') == vbt.CAQuery(base_cls='A')
        assert vbt.CAQuery.parse('A', use_base_cls=False) == vbt.CAQuery(cls='A')
        assert vbt.CAQuery.parse(vbt.Regex('A')) == vbt.CAQuery(base_cls=vbt.Regex('A'))
        assert vbt.CAQuery.parse(vbt.Regex('A'), use_base_cls=False) == vbt.CAQuery(cls=vbt.Regex('A'))
        assert vbt.CAQuery.parse(A) == vbt.CAQuery(base_cls=A)
        assert vbt.CAQuery.parse(A, use_base_cls=False) == vbt.CAQuery(cls=A)
        assert vbt.CAQuery.parse((A, B)) == vbt.CAQuery(base_cls=(A, B))
        assert vbt.CAQuery.parse((A, B), use_base_cls=False) == vbt.CAQuery(cls=(A, B))
        assert vbt.CAQuery.parse(dict(my_option1=True)) == vbt.CAQuery(options=dict(my_option1=True))
        assert vbt.CAQuery.parse(a) == vbt.CAQuery(instance=a)

    def test_ca_setup(self):
        @vbt.cacheable(max_size=2)
        def f(a, b, c=3):
            return a + b + c

        setup = vbt.CASetup.get(f)
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

        class A:
            @vbt.cacheable_method(max_size=2)
            def f(self, a, b, c=3):
                return a + b + c

        a = A()
        setup2 = vbt.CASetup.get(A.f, instance=a)
        assert setup2.run_func(10, 20, c=30) == 60
        assert setup2.run_func_and_cache(10, 20, c=30) == 60
        assert setup2.misses == 1
        assert setup2.hits == 0
        assert len(setup2.cache) == 1
        assert setup2.cache[hash((('self', id(a)), ('a', 10), ('b', 20), ('c', 30)))] == 60

        assert vbt.CASetup.get(A.f) is vbt.CASetup.get(A.f)
        assert vbt.CASetup.get(A.f, instance=a) is vbt.CASetup.get(A.f, instance=a)
        assert vbt.CASetup.get(A.f) is not vbt.CASetup.get(A.f, instance=a)

        with pytest.raises(Exception):
            _ = vbt.CASetup(A.f, instance=a)

    def test_cacheable_registry(self):
        pass
