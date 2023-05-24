import collections.abc as cabc
import typing as typ

import numpy as np
import pycsou.operator as pycop
import pycsou.util.ptype as pyct

import pycsou_pywt as pycwt


def stackedWaveletDec(
    input_shape: typ.Union[pyct.NDArrayShape, cabc.Sequence[pyct.NDArrayShape]],
    wl_list: typ.Union[str, cabc.Sequence[str]],
    level_list: typ.Union[int, cabc.Sequence[int]],
    mode_list: typ.Union[str, cabc.Sequence[str]] = "zero",
    include_id: bool = True,
) -> pyct.OpT:
    """
    Multi dictionaries wavelet decomposition.
    Computes the decomposition of an image according to different wavelet bases, stacked together in a single array.
    Optionally, include the identity operator among the list of operators.

    ``wl_list`` and ``level_list`` parameters can be specified as list or single elements. When only an element is
    provided, it is used for all the wavelet operators considered.

    Refer to :py:class:`~pycsou_pywt.operator.wavelet2d.WaveletDec2` for a description of the parameters.


    Parameters
    ----------
    input_shape: pyct.NDArrayShape
        (..., m, n) Shape of the images to apply the decomposition of, must be a 2-dimensional tuple.
    include_id: bool
        If ``True``, the identity operation is included among the wavelet transforms.

    Returns
    -------
    op: :py:class:`~pycsou.abc.operator.LinOp`
        Linear operator created as the vertical stacking of the considered wavelet operators.
    """
    length = len(wl_list) + include_id
    if not isinstance(level_list, list):
        level_list = [
            level_list,
        ] * length
    if not isinstance(mode_list, list):
        mode_list = [
            mode_list,
        ] * length
    op_list = [
        pycwt.WaveletDec2(input_shape, wl, level=level, mode=mode)
        for wl, level, mode in zip(wl_list, level_list, mode_list)
    ]
    if include_id:
        op_list.append(pycop.IdentityOp(dim=input_shape[0] * input_shape[1]))
    res_op = (1.0 / np.sqrt(length)) * pycop.stack(op_list, axis=0)

    if all([m == "zero" for m in mode_list]):
        res_op._lipschitz = 1.0

    return res_op


if __name__ == "__main__":
    import time

    import pywt

    ## Test stack
    wl_list = pywt.wavelist("sym")[:8]
    input_shape = (256, 256)
    stack = stackedWaveletDec(input_shape, wl_list, None)
    start = time.time()
    print(stack._lipschitz, stack.lipschitz(tight=True))
    print(f"Computation time of the Lipschitz constant: {time.time() - start:.4f} s")

    print(stack.shape)
    # test adjoint
    a = abs(np.random.normal(size=input_shape).reshape((-1)))
    b = np.random.normal(size=stack.shape[0])

    start = time.time()
    print(np.allclose((stack(a) * b).sum(), (a * stack.adjoint(b)).sum()))
    print(f"Evaluation time if the stacked operator time: {time.time() - start:.4f} s")
