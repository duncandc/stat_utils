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

    def _argcheck(self, a, b, mu, sigma):
        r"""
        check arguments
        """
        self.a = a  # lower bound
        self.b = b  # upper bound
        return (self.a >= 0.0) & (self.b > self.a)

    def n1(a, b, mu, sigma):
        return 0.5 + 0.5*erf((np.log(a)-mu)/(np.sqrt(2)*sigma))

    def n2(a, b, mu, sigma):
        return 1.0 - 0.5 + 0.5*erf((np.log(b)-mu)/(np.sqrt(2)*sigma))

    def norm(self, a, b):
        """
        """
        return 1.0/(sigma*np.sqrt(2.0*np.pi))* 1.0/(n1(a, b, mu, sigma) + n2(a, b, mu, sigma))

    def _pdf(self, x, a, b, mu, sigma):
        """
        """
        norm = self.norm(a, b)
        return np.where((x>a) & (x<=b), norm/x*np.exp(-1.0*(np.log(x)-mu)**2/(2.0*sigma**2)), 0.0)

    def _cdf(self, x, a, b, mu, sigma):
        cdf = 0.5 + 0.5*erf((np.log(a)-mu)/(np.sqrt(2)*sigma)) - n1(a, b, mu, sigma)
        cdf = np.where(x>a, cdf, 0.0)
        cdf = np.where(x<=b, cdf, 1.0)
        return cdf
    
    def _ppf(self, q, a, b, mu, sigma):
        
