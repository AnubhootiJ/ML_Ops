import math
from sklearn import datasets

def test_sqrt():
    num = 25
    assert math.sqrt(num) == 5

def test_square():
    num = 7
    assert 7*7 == 49

def test_equality():
    # assert 10 == 9 # gives AssertionError
    assert 10 == 10