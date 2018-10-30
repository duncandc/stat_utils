r"""
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from scipy.stats import rv_continuous, lognorm
from scipy.special import erf, erfinv, logit, expit


__all__ = ('TruncLogNorm',)
__author__ = ('Duncan Campbell')


class TruncLogNorm(rv_continuous):
    r"""
    truncted log-normal distribution
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
        """
        return 1.0/(s * np.sqrt(2.0*np.pi)) * 1.0/(self._n1(a, b, s) + self._n2(a, b, s))

    def _f_trunc(self, a, b, s):
        r"""
        """
        return self._norm(a, b, s)/self._norm(0.0, np.inf, s)

    def _pdf(self, x, a, b, s):
        r"""
        """
        norm = self._norm(a, b, s)
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where((x>a) & (x<=b), (norm/x)*np.exp(-1.0*(np.log(x))**2/(2.0*s**2)), 0.0)

    def _cdf(self, x, a, b, s):
        """
        """
        cdf = 0.5 + 0.5 * erf((np.log(x))/(np.sqrt(2.0) * s)) - self._n1(a, b, s)
        cdf = cdf/self._n2(a, b, s)
        cdf = np.where(x>a, cdf, 0.0)
        cdf = np.where(x<=b, cdf, 1.0)
        return cdf

    def _ppf(self, q, a, b, s):
        r"""
        """
        q = q*self._n2(a, b, s)
        ppf = np.exp(erfinv(2.0*q-1.0+2.0*self._n1(a, b, s))*np.sqrt(2.0*s))
        return ppf

