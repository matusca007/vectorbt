import pytest

import vectorbt as vbt
from vectorbt.ca_registry import ca_registry


# ############# ca_registry.py ############# #

class TestCacheableRegistry:
    def test_casetup(self):
        @vbt.cacheable
        def f(a, b, c=3):
            return a + b + c

        setup = vbt.CASetup(f, args=(10, 20), kwargs=dict(c=30))
        setup2 = vbt.CASetup(f, args=(10, 20), kwargs=dict(c=30))
        assert setup.run_func() == 60
        assert setup.hash_key == (f, None, (10, 20), (('c', 30),))
        assert setup == setup2
        assert hash(setup) == hash(setup2)
        with pytest.raises(Exception):
            _ = vbt.CASetup(f, args=(10, 20), kwargs=dict(d=30)).hash
        with pytest.raises(Exception):
            _ = vbt.CASetup(f, args=(10, 20), kwargs=dict(c=[1, 2, 3])).hash

        @vbt.cacheable(ignore_args=['c'])
        def f2(a, b, c=3):
            return a + b + c

        _ = vbt.CASetup(f2, args=(10, 20), kwargs=dict(c=[1, 2, 3])).hash

        class A:
            @vbt.cacheable
            def f(self, a, b, c=3):
                return a + b + c

        a = A()
        setup3 = vbt.CASetup(A.f, instance=a, args=(10, 20), kwargs=dict(c=30))
        assert setup3.run_func() == 60
        assert setup3.hash_key == (A.f, id(a), (10, 20), (('c', 30),))
        _ = setup3.hash

        assert {setup: 0, setup3: 1}[setup] == 0
        assert {setup: 0, setup3: 1}[setup3] == 1

    def test_caquery(self):
        class A:
            @vbt.cacheable_property(my_option=True)
            def f(self):
                return None

            @property
            def f2(self):
                return None

        class B(A):
            @vbt.cacheable_method(my_option=False)
            def f(self):
                return None

            @vbt.utils.decorators.custom_function
            def f2(self):
                return None

        @vbt.cacheable
        def f():
            return None

        a = A()
        b = B()

        def match_query(query):
            matched = []
            if query.matches_setup(vbt.CASetup(A.f, a)):
                matched.append('A.f')
            if query.matches_setup(vbt.CASetup(B.f, b)):
                matched.append('B.f')
            if query.matches_setup(vbt.CASetup(f)):
                matched.append('f')
            return matched

        assert match_query(vbt.CAQuery(instance=a)) == ['A.f']
        assert match_query(vbt.CAQuery(instance=b)) == ['B.f']
        assert match_query(vbt.CAQuery(cacheable=A.f)) == ['A.f']
        assert match_query(vbt.CAQuery(cacheable=B.f)) == ['B.f']
        assert match_query(vbt.CAQuery(cacheable=A.f.func)) == ['A.f']
        with pytest.raises(Exception):
            _ = match_query(vbt.CAQuery(cacheable=A.f2))
        assert match_query(vbt.CAQuery(cacheable=B.f.func)) == ['B.f']
        with pytest.raises(Exception):
            _ = match_query(vbt.CAQuery(cacheable=B.f2))
        assert match_query(vbt.CAQuery(cacheable='f')) == ['A.f', 'B.f', 'f']
        assert match_query(vbt.CAQuery(instance=a)) == ['A.f']
        assert match_query(vbt.CAQuery(instance=b)) == ['B.f']
        assert match_query(vbt.CAQuery(cls=A)) == ['A.f']
        assert match_query(vbt.CAQuery(cls=B)) == ['B.f']
        assert match_query(vbt.CAQuery(cls='A')) == ['A.f']
        assert match_query(vbt.CAQuery(cls='B')) == ['B.f']
        assert match_query(vbt.CAQuery(cls='A')) == ['A.f']
        assert match_query(vbt.CAQuery(cls='B')) == ['B.f']
