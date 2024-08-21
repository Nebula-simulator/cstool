import numpy as np
from cstool.common import units
from scipy.integrate import trapezoid, cumulative_trapezoid

def icdf(x, cdf, P):
	"""Compute the Inverse Cumulative Distribution Function (ICDF), given the
	variable (x), and the cumulative distribution function (cdf) for each x.

	P represents the cdf probabilities (between 0 and 1) that the icdf is to be
	evaluated at.
	"""
    # We need this to work around a bug in pint.
    # a = np.zeros(10) * units.dimensionless; a != a gives just a single "False".
    # If any of the values is non-zero, pint does what you'd expect.
	if hasattr(cdf, 'units'):
		assert cdf.units == units.dimensionless
		cdf = cdf.magnitude

	# Make sure to only pass points where the CDF has changed.
	OK = np.r_[True, cdf[1:] != cdf[:-1]]
	return np.interp(P, cdf[OK], x[OK].magnitude) * x.units

def compute_tcs_icdf(f, P, eval_x):
	"""Compute the integrated cross section and ICDF for the one-dimensional
	differential cross section function f(x). P represents the CDF probabilities
	for which the ICDF is to be evaluated.

	eval_x are the coordinates at which f is to be evaluated.

	Returns the integrated cross section and ICDF.
	"""
	eval_x *= units.dimensionless
	y = f(eval_x) * units.dimensionless

	cf = np.r_[0, cumulative_trapezoid(y.magnitude, eval_x.magnitude)] * y.units*eval_x.units

	if cf[-1] <= 0*cf.units:
		return 0*cf.units, np.zeros_like(P) * eval_x.units
	return cf[-1], icdf(eval_x, cf/cf[-1], P)

def compute_tcs(f, eval_x):
	"""Compute the integrated cross section for the one-dimensional differential
	cross section function f(x).

	eval_x are the coordinates at which f is to be evaluated.

	Returns the integrated cross section.
	"""
	eval_x *= units.dimensionless
	y = f(eval_x) * units.dimensionless
	return trapezoid(y.magnitude, eval_x.magnitude) * y.units * eval_x.units
