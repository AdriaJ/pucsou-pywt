import numpy as np

from pycsou_pywt import eigh


# tmp_path is a pytest fixture
def test_math():
    rng = np.random.default_rng(seed=0)
    arg_shape = (5, 5)
    batch_shape = (3, 2)
    input_shape = batch_shape + arg_shape
    x = rng.normal(size=input_shape)
    x += x.swapaxes(2,3)
    w, v = eigh(x.reshape(*batch_shape, -1), arg_shape=arg_shape)
    w_np, v_np = np.linalg.eigh(x)
    assert np.allclose(v.ravel(), v_np.ravel())
    assert np.allclose(w.ravel(), w_np.ravel())