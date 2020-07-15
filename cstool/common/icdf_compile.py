import numpy as np
from cstool.common import units
from scipy.integrate import cumtrapz

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

	cf = np.r_[0, cumtrapz(y.magnitude, eval_x.magnitude)] * y.units*eval_x.units

	if cf[-1] <= 0*cf.units:
		return 0*cf.units, np.zeros_like(P) * eval_x.units
	return cf[-1], icdf(eval_x, cf/cf[-1], P)


def compute_2d_tcs_icdf(function_data, eval_x, eval_y, # Function, plus x and y coordinates at which it's been sampled
	y_low_f, y_high_f,                                 # Range for which we are interested in the function. y_low_f and y_high_f must be within sample_y.
	P1d_axis, x2d_axis, P2d_axis):                     # The axes for the returned 1D and 2D icdfs. 1D icdf represents the x axis, 2d icdf represents y given x.
	"""Compute integrated cross section and ICDFs for the two-dimensional
	differential cross section function f(x, y).

	This function takes the function's value, function_data, as a 2D numpy
	array on all coordinates (eval_x, eval_y). For each eval_x, the data is
	only used for each eval_y between y_low_f(x) and y_high_f(x). The case
	y_low_f(x) > y_high_f(x) is allowed: the total cross section will be zero.

	The reason for this design, instead of accepting f() as a function and
	evaluating it only at relevant coordinates, is that this function is often
	used in a loop where each iteration has the same f, eval_x and eval_y; but
	different y_low_f and y_high_f. Empirically, evaluating f() every time is
	way too slow (hours in our situation).

	This function returns:
	  1. The total cross section
	  2. The ICDF for x, for each P in P1d_axis.
	  3. The ICDF for y given x. First index is x, from axis x2d_axis; second
		 index is P, from P2d_axis.
	"""
	if len(eval_x) == 0 or len(eval_y) == 0:
		return 0 * eval_x.units * eval_y.units * function_data.units, \
			np.zeros(len(P1d_axis)) * eval_x.units, \
			np.zeros([len(x2d_axis), len(P2d_axis)]) * eval_y.units

	###########################################################################
	# Integrals over the differential cross section function
	###########################################################################

	# CIy_x[i,j] == ∫_{y_low(eval_x[i])}^{eval_y[j]} f(eval_x[i], y')
	# for eval_y[j] between y_low(eval_x[i]) and y_high(eval_x[i])
	CIy_x = np.zeros([len(eval_x), len(eval_y)])
	y_low = y_low_f(eval_x).to(eval_y.units).magnitude;   # Out of loop, stripping units for performance.
	y_high = y_high_f(eval_x).to(eval_y.units).magnitude; # pint seems to give HUGE penalties.
	for i in range(len(eval_x)):
		data = np.zeros(len(eval_y))
		OK = np.logical_and(eval_y.magnitude > y_low[i], eval_y.magnitude < y_high[i])
		data[OK] = function_data[i,OK].magnitude
		CIy_x[i,1:] = cumtrapz(data, eval_y.magnitude)

	# CIx[i] = ∫_{eval_x[0]}^{eval_x[i]} dx' ∫_{y_low(x')}^{y_high(x')} dy' f(x', y')
	#        = ∫_{eval_x[0]}^{eval_x[i]} dx' CIy_x[x', -1]
	CIx = np.zeros(len(eval_x))
	CIx[1:] = cumtrapz(CIy_x[:,-1], eval_x.magnitude)

	# total_cs = ∫_{eval_x[0]}^{eval_x[-1]} dx' ∫_{y_low(x')}^{y_high(x')} dy' f(x', y')
	#          = CIx[-1]
	total_cs = CIx[-1]
	if total_cs <= 0:
		return 0 * eval_x.units * eval_y.units * function_data.units, \
			np.zeros(len(P1d_axis)) * eval_x.units, \
			np.zeros([len(x2d_axis), len(P2d_axis)]) * eval_y.units

	###########################################################################
	# Normalize the integrals to cumulative probability distributions
	###########################################################################
	# CDF for x
	# P[i] = ∫_{eval_x[0]}^{eval_x[i]} dx' p(x')
	#      = ∫_{eval_x[0]}^{eval_x[i]} dx' ∫_{y_low(x')}^{y_high(x')} f(x', y') / total_cs
	cPx = CIx / total_cs

	# CDF for y given x
	# Notation: xx == eval_x[i]
	# cPy_x[i,j] = ∫_{y_low(xx),eval_y[j]} dy' p(y' | xx)
	#            = ∫_{y_low(xx),eval_y[j]} dy' p(xx, y') / p(xx)
	#            = [∫_{y_low(xx),eval_y[j]} dy' p(xx, y')] / [∫_{y_low(xx)}^{y_high(xx)} dy'' p(xx, y'')]
	#            =                                f(xx, y')                                       f(xx, y'')
	cPy_x = np.apply_along_axis(np.divide, 0, CIy_x, CIy_x[:,-1],
		out=np.zeros(CIy_x.shape[0]), where=(CIy_x[:,-1]>0))

	###########################################################################
	# Invert the CDFs to ICDFs
	###########################################################################
	icdf_x = icdf(eval_x, cPx, P1d_axis)

	#icdf_yx = griddata((np.repeat(eval_x, len(eval_y)), cPy_x.flatten()),
	#	np.tile(eval_y, len(eval_x)),
	#	tuple(np.meshgrid(x2d_axis, P2d_axis, indexing='ij')))
	icdf_yx = np.zeros([len(x2d_axis), len(P2d_axis)]) * eval_y.units
	for i, xx in enumerate(x2d_axis):
		i_eval_x = min(np.searchsorted(eval_x.to(xx.units).magnitude, xx.magnitude), len(eval_x) - 2)
		frac = (xx - eval_x[i_eval_x]) / (eval_x[i_eval_x+1] - eval_x[i_eval_x])
		cdf = (1-frac)*cPy_x[i_eval_x,:] + frac*cPy_x[i_eval_x+1,:]
		icdf_yx[i,:] = icdf(eval_y, cdf, P2d_axis)

	return total_cs * eval_x.units * eval_y.units * function_data.units, \
		icdf_x, icdf_yx
