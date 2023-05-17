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

    return (1.0 / np.sqrt(length)) * pycop.stack(op_list, axis=0)
