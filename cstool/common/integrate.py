from scipy.integrate import quad as sciquad

def quad(func, a, b, args=()):
	"""Units-aware wrapper around scipy.quad().

	Does not support the full functionality of scipy.quad yet.
	"""
	x_units = a.units
	f_units = func(.5*(a+b)).units

	I, abserr = sciquad(
		lambda x : func(x*x_units).to(f_units).magnitude,
		a.magnitude, b.to(x_units).magnitude,
		args)

	return I*x_units*f_units, abserr*x_units*f_units
