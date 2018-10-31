r"""
a truncated log-normal distribution
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from scipy.stats import rv_continuous, lognorm
from scipy.special import erf, erfinv, logit, expit


__all__ = ('TruncLogNorm',)
__author__ = ('Duncan Campbell')


class TruncLogNorm(rv_continuous):
    r"""
    A truncated log-normal continuous random variable.

    Notes
    -----
    The probability density function for `TruncLogNorm` is:
    
    .. math::
        f(x, s) \propto \frac{1}{s x \sqrt{2\pi}}
                  \exp(-\frac{1}{2} (\frac{\log(x)}{s})^2)
    
    for ``a < x <= b``, and ``s > 0``.
    
    `TruncLogNorm` takes ``a``, ``b``, and ``s`` as a shape parameters.
    
    A common parametrization for a lognormal random variable ``Y`` is in
    terms of the mean, ``mu``, and standard deviation, ``sigma``, of the
    unique normally distributed random variable ``X`` such that exp(X) = Y.
    This parametrization corresponds to setting ``s = sigma`` and ``scale =
    exp(mu)``.

    The standard form of this distribution is a standard log-normal
    truncated to the range [a,b] - notice that a and b
    are defined over the domain of the standard log-normal. 
    
    To convert clip values for a specific mean and standard deviation, use::

        a, b = (myclip_a - loc) / scale, (myclip_b - loc) / scale

    To get a "frozen" RV object, holding the given parameters fixed, use::

        >>> trunc_lognorm = TruncLogNorm()
        >>> frozen_rv = trunc_lognorm(a=a, b=b, s=s, loc=loc, scale=scale)

    where a, b, s, loc, and scale are the desired paramaters of the 
    frozen distribution.
    """

    def _argcheck(self, a, b, s):
        r"""
        check arguments
        """
        self.a = a  # lower bound
        self.b = b  # upper bound
        return (self.a >= 0.0) & (self.b > self.a) & (s>0.0)

    def _n1(self, a, b, s):
        r"""
        """
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(a>0.0, 0.5 + 0.5*erf((np.log(a))/(np.sqrt(2)*s)), 0.0)

    def _n2(self, a, b, s):
        r"""
        """
        return 1.0 - 0.5 + 0.5*erf((np.log(b))/(np.sqrt(2)*s))

    def _norm(self, a, b, s):
        r"""
        normalization factor
        """
        return 1.0/(s * np.sqrt(2.0*np.pi)) * 1.0/(self._n1(a, b, s) + self._n2(a, b, s))

    def _f_trunc(self, a, b, s):
        r"""
        """
        return self._norm(a, b, s)/self._norm(0.0, np.inf, s)

    def _pdf(self, x, a, b, s):
        r"""
        probability distribution function
        """
        norm = self._norm(a, b, s)
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where((x>a) & (x<=b), (norm/x)*np.exp(-1.0*(np.log(x))**2/(2.0*s**2)), 0.0)

    def _cdf(self, x, a, b, s):
        """
        cumulative distribution function
        """
        cdf = 0.5 + 0.5 * erf((np.log(x))/(np.sqrt(2.0) * s)) - self._n1(a, b, s)
        cdf = cdf/self._n2(a, b, s)
        cdf = np.where(x>a, cdf, 0.0)
        cdf = np.where(x<=b, cdf, 1.0)
        return cdf

    def _ppf(self, q, a, b, s):
        r"""
        percent point function (inverse of cdf - percentiles)
        """
        q = q*self._n2(a, b, s)
        ppf = np.exp(erfinv(2.0*q-1.0+2.0*self._n1(a, b, s))*np.sqrt(2.0*s**2))
        return ppf

