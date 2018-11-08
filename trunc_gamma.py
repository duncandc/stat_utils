r"""
a truncated gamma distribution
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from scipy.stats import rv_continuous
from scipy.special import gamma, gammainc, gammaincinv


__all__ = ('TruncGamma',)
__author__ = ('Duncan Campbell')

class TruncGamma(rv_continuous):
    r"""
    """

    def _argcheck(self, a, b, s):
        r"""
        check arguments
        """
        self.a = a  # lower bound
        self.b = b  # upper bound
        return (self.a >= 0.0) & (self.b > self.a)

    def norm(self, a, b, s):
        """
        """
        return 1.0/(gammainc(s, a) + gammainc(s, b))

    def _pdf(self, x, a, b, s):
        """
        """
        mask = (x<self.a) | (x>self.b)
        result = x**(a-1) * np.exp(-x) / gamma(s)
        result[mask] = 0.0
        return self.norm(a, b, s)*result

    def _cdf(self, x, a, b, s):
        """
        """
        return (gammainc(s, x) - gammainc(s, a))/gammainc(s, b)

    def _ppf(self, q, a, b, s):
        """
        """
        return gammaincinv(s, q*gammainc(s, b) + gammainc(s, a))

