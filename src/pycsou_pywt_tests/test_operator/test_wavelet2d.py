import itertools

import numpy as np
import pycsou.runtime as pycrt
import pycsou.util.deps as pycd
import pycsou_tests.operator.conftest as conftest
import pytest
import pywt

import pycsou_pywt as pycwt


class TestWaveletDec2(conftest.LinOpT):
    @pytest.fixture(params=[(16, 16), (17, 24)])
    def input_shape(self, request):
        return request.param

    @pytest.fixture(params=["db2", "sym2"])
    def wavelet_name(self, request):
        return request.param

    @pytest.fixture(params=[1, None])
    def level(self, request):
        return request.param

    @pytest.fixture(
        params=itertools.product(
            [pycd.NDArrayInfo.NUMPY],
            pycrt.Width,
        )
    )
    def spec(self, input_shape, wavelet_name, level, request):
        ndi, width = request.param
        op = pycwt.WaveletDec2(input_shape, wavelet_name, level=level)
        return op, ndi, width

    @pytest.fixture
    def data_shape(self, input_shape, wavelet_name, level):
        dim = np.prod(input_shape)
        arr, _ = pywt.coeffs_to_array(pywt.wavedec2(np.zeros(input_shape), wavelet_name, mode="zero", level=level))
        codim = np.prod(arr.shape)
        return (codim, dim)

    @pytest.fixture
    def data_apply(self, input_shape, wavelet_name, level):
        input_im = self._random_array(input_shape, seed=1)
        arr, _ = pywt.coeffs_to_array(pywt.wavedec2(input_im, wavelet_name, mode="zero", level=level))
        return dict(
            in_=dict(arr=input_im.flatten()),
            out=arr.flatten(),
        )
