import numpy as np

from pycsou_pywt import Flip


def test_linop():
    """An example of how you might test your plugin."""
    rng = np.random.default_rng(seed=0)
    arg_shape = (5, 5)
    batch_shape = (3, 2)
    input_shape = batch_shape + arg_shape
    x = rng.normal(size=input_shape)
    flip = Flip(arg_shape=arg_shape, axis=0)
    out = flip(x.reshape(*batch_shape, -1)).reshape(*input_shape)
    assert np.allclose(out, np.flip(x, axis=2))