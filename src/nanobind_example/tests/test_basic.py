import nanobind_example as m

def test_add():
    assert m.add(1, 2) == 3

def test_subtract():
    assert m.subtract(1, 2) == -1
