import numpy as np
from cstool.common import units

def interp_ll(x, y, left=None, right=None):
	"""Log-log interpolation of 1D data.

	x and y are arrays of values. This function returns a function that can be
	used to find interpolated or extrapolated values.

	If out-of-bounds, left and right indicate the default behaviour. If they
	are none, extrapolate; otherwise, fill with the indicated value."""

	x *= units.dimensionless
	y *= units.dimensionless

	x_log_steps = np.log(x[1:]/x[:-1])
	log_y = np.log(y.magnitude, where=y.magnitude>0)
	log_y[y.magnitude <= 0] = -np.inf

	def f(xp):
		xp = (xp * units.dimensionless).to(x.units)

		x_idx = np.searchsorted(x.magnitude, xp.magnitude)
		mx_idx = np.clip(x_idx - 1, 0, len(x) - 2)

		# Compute the weight factor. Have to take magnitude because of a bug in pint.
		w = np.log((xp / x[mx_idx]).magnitude,
				where=xp>0*xp.units, out=np.zeros(xp.shape)) \
			/ x_log_steps[mx_idx]

		# Interpolated / extrapolated values on log-log scale
		yp = np.exp((1 - w) * log_y[mx_idx] + w * log_y[mx_idx + 1]) * y.units

		# Out-of-bounds behaviour
		if left is not None:
			yp[x_idx == 0] = left
		if right is not None:
			yp[x_idx == len(x)] = right

		return yp

	return f

def interpolate_f(f1, f2, a, b, h = lambda x : x):
    """Interpolate between two functions in a certain range.

	This function returns a new function, g. For x<a, g(x) == f1(x). For x>b,
	g(x) == f2(x). For a<x<b, g(x) interpolates between f1 and f2 using the
	function h. Function h takes a parameter in the range [0,1] and returns a
	number in the same range.
	"""
    assert callable(f1)
    assert callable(f2)
    assert callable(h)

    def g(x):
        y1 = f1(x)
        y2 = f2(x)

        u = np.clip((x - a)/(b - a), 0.0, 1.0)
        w = h(u)
        ym = (1 - w) * y1 + w * y2

        return np.where(
            x < a, y1, np.where(
                x > b, y2, ym))
    return g
