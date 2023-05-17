pycsou-pywt
===========

|License MIT| |PyPI| |Python Version| |tests| |codecov| |pycsou fair|

A simple plugin to implement wavelet decomposition as a Pycsou_ linear
operator. The plugin interfaces Pycsou with the PyWavelets_  package [PyWt]_ and thus enables
a wide range of wavelet bases to be used.


Installation
------------

You can install ``pycsou-pywt`` via
`pip <https://pypi.org/project/pip/>`__:

.. code:: bash

   pip install pycsou-pywt

Contributing
------------

Contributions are very welcome. Tests can be run with
`tox <https://tox.readthedocs.io/en/latest/>`__, please ensure the
coverage at least stays the same before you submit a pull request.

License
-------

Distributed under the terms of the
`MIT <http://opensource.org/licenses/MIT>`__ license, “pycsou-pywt” is
free and open source software

Issues
------

If you encounter any problems, please `file an issue <https://github.com/AdriaJ/pycsou-pywt/issues>`_ along with a
detailed description.

--------------

This Pycsou_ plugin was generated with `Cookiecutter <https://github.com/audreyr/cookiecutter>`__ using
Pycsou’s `cookiecutter-pycsou-plugin <https://github.com/matthieumeo/cookiecutter-pycsou-plugin>`__
template.

.. raw:: html

   <!--
   Don't miss the full getting started guide to set up your new package:
   https://github.com/matthieumeo/cookiecutter-pycsou-plugin#getting-started

   and review the pycsou docs for plugin developers:
   https://www.pycsou-fair.org/plugins
   -->

Todo list
---------
- Turn wavelet operators from LinOp into UnitOp (need to define the adjoint for any type of mode though)
    - Add a method ``invert``
    - Define adjoint as invert if mode='zero'
- Implement 1d and Nd wavelet decomposition.
    - Define wavelet2d as a special case of Nd
- Validate the tests
    - diff_lipschitz, lipschitz and svdvals are still causing troubles
- Finish README.rst file
- Link the doc with Pycsou's doc so that we can access the references target.
- Fix the display format of the references
- Fix the badges


.. _Python: https://www.python.org/
.. _Pycsou: https://github.com/matthieumeo/pycsou
.. _PyWavelets: https://pywavelets.readthedocs.io/en/latest/


.. |License MIT| image:: https://img.shields.io/pypi/l/pycsou-pywt.svg?color=green
   :target: https://github.com/AdriaJ/pycsou-pywt/raw/main/LICENSE
.. |PyPI| image:: https://img.shields.io/pypi/v/pycsou-pywt.svg?color=green
   :target: https://pypi.org/project/pycsou-pywt
.. |Python Version| image:: https://img.shields.io/pypi/pyversions/pycsou-pywt.svg?color=green
   :target: https://python.org
.. |tests| image:: https://github.com/AdriaJ/pycsou-pywt/workflows/tests/badge.svg
   :target: https://github.com/AdriaJ/pycsou-pywt/actions
.. |codecov| image:: https://codecov.io/gh/AdriaJ/pycsou-pywt/branch/main/graph/badge.svg
   :target: https://codecov.io/gh/AdriaJ/pycsou-pywt
.. |pycsou fair| image:: https://img.shields.io/endpoint?url=https://api.pycsou-fair.org/shields/pycsou-pywt
   :target: https://pycsou-fair.org/plugins/pycsou-pywt