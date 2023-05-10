from pycsou_pywt import NullFunc


def test_nullfunc():
    assert NullFunc(1)._name == "ModifiedNullFunc"
