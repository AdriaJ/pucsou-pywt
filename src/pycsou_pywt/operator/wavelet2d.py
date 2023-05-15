import numpy as np
import pycsou.abc.operator as pyco
import pycsou.operator as pycop
import pycsou.runtime as pycrt
import pycsou.util.ptype as pyct
import pywt

__all__ = ["WaveletDec2", "stackedWaveletDec"]


class WaveletDec2(pyco.LinOp):
    def __init__(self, input_shape, wavelet_name: str, level=None, mode="zero"):
        self.image_shape = input_shape
        self._wavelet = wavelet_name
        if level is None:
            level = pywt.dwtn_max_level(input_shape, wavelet_name)
        self._level = level
        self._mode = mode
        init_coeffs = pywt.wavedec2(np.zeros(self.image_shape), self._wavelet, mode=self._mode, level=self._level)
        arr, slices = pywt.coeffs_to_array(init_coeffs, axes=(-2, -1))
        self.coeff_slices = slices
        self.coeff_shape = arr.shape
        super().__init__(shape=(arr.shape[0] * arr.shape[1], input_shape[0] * input_shape[1]))

        if self._mode == "zero":
            self._lipschitz = 1.0

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        raster_image = arr.reshape(*arr.shape[:-1], *self.image_shape)
        coeffs = pywt.wavedec2(
            raster_image,
            self._wavelet,
            mode=self._mode,
            level=self._level,
        )
        return pywt.coeffs_to_array(coeffs, axes=(-2, -1))[0].reshape(*arr.shape[:-1], -1)

    @pycrt.enforce_precision(i="arr")
    def adjoint(self, arr: pyct.NDArray) -> pyct.NDArray:
        # coeffs_from_arr = pywt.unravel_coeffs(arr, self.coeff_slices, self.coeff_shapes, output_format='wavedec2')
        raster_arr = arr.reshape(*arr.shape[:-1], *self.coeff_shape)
        if arr.ndim == 1:
            slices = self.coeff_slices
        else:
            _, slices = pywt.coeffs_to_array(
                pywt.wavedec2(np.zeros((*arr.shape[:-1], *self.image_shape)), self._wavelet, level=self._level),
                axes=(-2, -1),
            )
        coeffs = pywt.array_to_coeffs(raster_arr, slices, output_format="wavedec2")
        res = pywt.waverec2(
            coeffs,
            self._wavelet,
            mode=self._mode,
        )
        return self._odd_cropping(res).reshape(*arr.shape[:-1], -1)

    @property
    def level(self):
        return self._level

    def _odd_cropping(self, arr: np.ndarray) -> np.ndarray:
        r"""

        Parameters
        ----------
        arr: np.ndarray
            (..., n, m) 2D image (or stack of 2D images) returned by pywt.waverec2().

        Returns
        -------
        out: np.ndarray
            (..., *self.image_shape) 2D image with correct size.

        """
        odd_test = [dim % 2 == 1 for dim in self.image_shape]
        if np.any(odd_test):
            # output_shape = tuple(dim + b for dim, b in zip(self.image_shape, odd_test))
            # large_arr = arr.reshape((*arr.shape[-1:], *output_shape))
            s = (slice(None, None, None),) * (np.ndim(arr) - 2) + tuple(
                slice(None, -1 if elem else None, None) for elem in odd_test
            )
            return arr[s]
        else:
            return arr

    # def reshape_coeffs_arr2im(self, arr: np.ndarray):
    #     coeffs_from_arr = pywt.unravel_coeffs(arr, self.coeff_slices, self.coeff_shapes, output_format='wavedec2')
    #     im, _ = pywt.coeffs_to_array(coeffs_from_arr, axes=(-2, -1))
    #     return im


def stackedWaveletDec(input_shape, wl_list, level_list, mode_list="zero", include_dirac: bool = True) -> pyco.LinOp:
    length = len(wl_list) + include_dirac
    if not isinstance(level_list, list):
        level_list = [
            level_list,
        ] * length
    if not isinstance(mode_list, list):
        mode_list = [
            mode_list,
        ] * length
    op_list = [
        WaveletDec2(input_shape, wl, level=level, mode=mode) for wl, level, mode in zip(wl_list, level_list, mode_list)
    ]
    if include_dirac:
        op_list.append(pycop.IdentityOp(dim=input_shape[0] * input_shape[1]))

    return (1.0 / np.sqrt(length)) * pycop.stack(op_list, axis=0)


# TODO test wavelet 2d and potentially extend to more dimensions, write tests: apply, adjoint
# TODO test dask

if __name__ == "__main__":
    image_shape = (64, 64)
    modes = ["zero", "symmetric"]  # Only zero padding works for auto-adjoint operator
    wls = ["db1", "db3", "db5"]
    levels = [1, 2, None]
    for mode in modes:
        print("Mode " + mode + " :")
        for wl in wls:
            print("\tWavelet " + wl + " :")
            for level in levels:
                print("\t\tLevel {} :".format(level))
                wdec = WaveletDec2(image_shape, wl, level=level, mode=mode)
                print("\t\t\tLipschitz value: {}".format(wdec.lipschitz(tol=1e-3)))
                a = abs(np.random.normal(size=image_shape)).flatten()
                b = abs(np.random.normal(size=wdec.coeffs_shape)).flatten()
                print("\t\t\tShape of the decomposition:", wdec(a).shape)
                print("\t\t\t\tCorrect adjoint: ", np.allclose(np.dot(wdec(a), b), np.dot(a, wdec.adjoint(b))))

                a = abs(np.random.normal(size=(5,) + image_shape)).reshape((5, -1))
                b = abs(np.random.normal(size=(5,) + wdec.coeffs_shape)).reshape((5, -1))
                print("\t\t\tShape of the decomposition:", wdec(a).shape)
                print(
                    "\t\t\t\tCorrect adjoint: ",
                    np.allclose(np.sum(wdec(a) * b, axis=1), np.sum(a * wdec.adjoint(b), axis=1)),
                )

    # ## Test stack
    # wl_list = pywt.wavelist('db')[:8]
    # input_shape = (512, 512)
    # stack = stackedWaveletDec(input_shape, wl_list, 2)
    # print(stack._lipschitz, stack.lipschitz(recompute=True))
    # print(stack.shape)
    # # test adjoint
    # a = abs(np.random.normal(size=input_shape).reshape((-1)))
    # b = np.random.normal(size=stack.shape[0])
    # import time
    #
    # start = time.time()
    # print(np.allclose((stack(a) * b).sum(), (a * stack.adjoint(b)).sum()))
    # print(f"Computation time: {time.time() - start:.4f} s")
