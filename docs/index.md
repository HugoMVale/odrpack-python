# odrpack

`odrpack` provides python bindings for the well-known weighted orthogonal distance regression
(ODR) solver [odrpack95].   

ODR, also known as [errors-in-variables regression], is designed primarily for instances when both
the explanatory and response variables have significant errors. 

<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/8/81/Total_least_squares.svg/220px-Total_least_squares.svg.png" width="200" alt="Deming regression; special case of ODR.">
</p>

[errors-in-variables regression]: https://en.wikipedia.org/wiki/Errors-in-variables_models
[odrpack95]: https://github.com/HugoMVale/odrpack95

## Installation

`odrpack` requires Python >= 3.10, because it makes use of recent type hint syntax. Besides that,
it only requires `numpy`.

In order to install the latest stable version from PyPI do:

```bash
pip install odrpack
```

Alternatively, the very latest code (no guarantee it will work!) may be installed directly from
the source code repository:
```bash
pip install git+https://github.com/HugoMVale/odrpack-python.git
```

