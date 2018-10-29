"""
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from ..trunc_lognorm import TruncLogNorm
from scipy.stats import lognorm
import matplotlib.pyplot as plt


def test_1():
    """
    test pdf
    """

    a = 0.0
    b = 1.0
    sigma = 1.0
    mu = 0.0

    s = sigma
    scale = np.exp(mu)
    loc = 0.0

    trunc_lognorm = TruncLogNorm()
    d1 = trunc_lognorm(a, b, s, loc=loc, scale=scale)
    d2 = lognorm(s, loc=loc, scale=scale)

    x = np.linspace(a, b, 100)
    y1 = d1.pdf(x)
    y2 = d2.pdf(x)

    # fractional difference in the normalization constant
    # between the truncated and un-truncated pdf
    f = trunc_lognorm._f_trunc(a, b, s)

    assert np.allclose(y1, (y2*f))


def test_2():
    """
    test cdf
    """

    a = 0.0
    b = 1.0
    sigma = 1.0
    mu = 0.0

    s = sigma
    scale = np.exp(mu)
    loc = 0.0

    trunc_lognorm = TruncLogNorm()
    d1 = trunc_lognorm(a, b, s, loc=loc, scale=scale)
    d2 = lognorm(s, loc=loc, scale=scale)

    x = np.linspace(0.0, 1.0, 100)
    y1 = d1.cdf(x)
    y2 = d2.cdf(x)

    # fractional difference in the normalization constant
    # between the truncated and un-truncated pdf
    f = trunc_lognorm._f_trunc(a, b, s)

    assert np.allclose(y1, y2*f)


def test_3():
    """
    test random variates
    """

    N = 100000

    a = 0.0
    b = 1.0
    sigma = 1.0
    mu = 0.0

    s = sigma
    scale = np.exp(mu)
    loc = 0.0

    trunc_lognorm = TruncLogNorm()
    d = trunc_lognorm(a, b, s, loc=loc, scale=scale)

    y = d.rvs(size=N)

    bins = np.linspace(0,2.0,100)
    bin_centers = (bins[:-1]+bins[1:])/2.0
    counts = np.histogram(y, bins=bins)[0]
    counts = counts/np.sum(counts)/np.diff(bins)

    # examine random vriate distribution
    #plt.figure()
    #plt.step(bins[:-1], counts)
    #plt.plot(bin_centers, d.pdf(bin_centers))
    #plt.show()

    assert (np.min(y)>a) & (np.max(y)<=b)

