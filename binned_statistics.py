"""
statistics caulcated in bins
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np

__all__ = ('binned_std', 'binned_mean',)
__author__ = ('Duncan Campbell', )


def binned_std(x, bins, bin_key, value_key, use_log=False):
    """
    """

    inds = np.digitize(x[bin_key],bins=bins)
    result = np.empty(len(bins)-1)
    for i in range(0,len(bins)-1):
        in_bin = (inds== i+1)
        if use_log==True:
            result[i] = np.std(np.log10(x[value_key][in_bin]))
        else:
            result[i] = np.std(x[value_key][in_bin])

    return bins, result


def binned_mean(x, bins, bin_key, value_key, use_log=False):
    """
    """

    inds = np.digitize(x[bin_key],bins=bins)
    result = np.empty(len(bins)-1)
    for i in range(0,len(bins)-1):
        in_bin = (inds== i+1)
        if use_log==True:
            result[i] = np.mean(np.log10(x[value_key][in_bin]))
        else:
            result[i] = np.mean(x[value_key][in_bin])

    return bins, result
