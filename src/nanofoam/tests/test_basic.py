import nanofoam as m

def test_add():
    assert m.add(1, 2) == 3

def test_subtract():
    assert m.subtract(1, 2) == -1

def test_multiply():
    assert m.multiply(2, 3) == 6

def test_divide():
    assert m.divide(6, 3) == 2
