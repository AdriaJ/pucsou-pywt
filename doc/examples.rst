Examples
========

WaveletDec2
-----------

We provide so far two Python scripts to illustrate how to use the wavelet decomposition operator
:py:class:`~pycsou_pywt.operator.wavelet2d.WaveletDec2`. To do so we load an image, provided in ``examples/data/``, that is decomposed according to a
given wavelet basis. Then, some high or low frequency coefficients are filtered out before reconstructing the image.

With these examples, we illustrate the capability of the Pycsou's :py:class:`~pycsou.abc.operator.LinOp` to handle
stacked inputs.

.. list-table:: Example files
    :header-rows: 1

    * - Path
      - Description
    * - ``examples/wavedec2_bw.py``
      - Deals with signle channel grayscale image.
    * - ``examples/wavedec2_color.py``
      - Uses stacked input to deal with 3-channels images.

