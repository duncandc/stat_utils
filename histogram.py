"""
1D histogram with poisson error estimates
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np

__all__ = ('histogram',)
__author__ = ('Duncan Campbell', )


def histogram(x, bins, weights=None):
    """
    Compute the histogram of a set of data 
    and estimate the standard devation on the counts

    Parameters
    ----------
    x : array_like
        Input data. The histogram is computed over the flattened array.
    
    bins : int or sequence of scalars or str
        If bins is an int, it defines the number of equal-width bins in the given range (10, by default). 
        If bins is a sequence, it defines the bin edges, including the rightmost edge, allowing for non-uniform bin widths.

    weights : array_like, optional
        array of weights.  if None, all weights are set to unity.

    Returns
    -------
    weighted_counts, err
       arrays of weighted counts and errors on the counts in bins.
    """

    # process weights argument
    if weights is None:
        weights = np.ones(len(x))
    else:
        weights = np.atleast_1d(weights)

    # weighted counts in bins
    counts = np.histogram(x, bins=bins, weights=weights)[0]

    # sum the square of weights in bins to estimate poisson errors
    inds = np.digitize(x, bins=bins)
    nbins = len(bins) - 1
    err = np.zeros(nbins)
    for i in range(0, nbins):
        mask = (inds == (i+1))
        err[i] = np.sqrt(np.sum(weights[mask]**2))

    return counts, err


