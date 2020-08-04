import numpy as np
from scipy.integrate import trapz, cumtrapz
from cstool.common import units
from cstool.common.icdf_compile import icdf, compute_tcs_icdf

@units.check(None, units.eV, units.dimensionless)
def compile_ashley_imfp_icdf(dimfp, K, P_omega):
	"""Compute inverse mean free path, and ω' ICDF for a dielectric function
	given in the format of Ashley (doi:10.1016/0368-2048(88)80019-7).

	Ashley transforms the dielectric function ε(ω, q) to ε(ω, ω'); with an
	analytical relationship between ω and ω'. Therefore, we only need to store
	the probability distribution for ω'.

	This function computes the total mean free path and ICDF for ω', given a
	function dimfp(K, ω').

	This function calls dimfp() for ω' between 0 and K, despite conservation of
	momentum dictating that ω' < K/2. This is because some models (e.g. Kieft,
	doi:10.1088/0022-3727/41/21/215310) ignore this restriction.

	Parameters:
	 - dimfp:   function, taking 2 parameters (K, ω') and returning differential
				inverse mean free paths.
	 - K:       Array of interesting electron kinetic energies
	 - P_omega: The probabilities for which the ICDF needs to be evaluated

	Returns:
	 - Inverse mean free path array, same length as K
	 - Inverse cumulative distribution function, a 2D array of shape (len(K) × len(P_omega))
	"""

	imfp = np.zeros(K.shape) * units('nm^-1')
	icdf = np.zeros((K.shape[0], P_omega.shape[0])) * units.eV

	for i, E in enumerate(K):
		tcs, ticdf = compute_tcs_icdf(lambda w : dimfp(E, w), P_omega,
			np.linspace(0, E.magnitude, 100000)*E.units)
		imfp[i] = tcs.to('nm^-1')
		icdf[i] = ticdf.to('eV')
		print('.', end='', flush=True)
	print()

	return imfp, icdf


@units.check(
	units.eV, 1/units.nm, units.dimensionless,
	units.eV, units.dimensionless, units.dimensionless, units.dimensionless,
	units.eV)
def compile_full_imfp_icdf(elf_omega, elf_q, elf_data,
	K, P_omega, n_omega_q, P_q,
	F):
	"""Compute inverse mean free path, energy transfer ICDF and momentum transfer
	ICDF for an arbitrary dielectric function ε(ω, q).

	ε(ω, q) is given by elf_omega, elf_q and elf_data.

	The data is evaluated for energies given by K. P_omega is the probability
	axis for the energy loss ICDF.
	The momentum transfer ICDF is computed for each (K, omega), with the second
	parameter in n_omega_q evenly-spaced steps between 0 and K. The probability
	axis for the momentum ICDF is given by P_q.

	F is the Fermi energy of the material. This is used for a "Fermi correction"
	to prevent omega > K-F.

	This function is relativistically correct.

	Parameters:
	 - elf_omega: ω for which the ELF 1/ε(ω, q) is known
	 - elf_q:     q for which the ELF is known
	 - elf_data:  The actual data. Shape is (len(ω) × len(q))
	 - K:         Electron kinetic energies to evaluate at.
	 - P_omega:   The probabilities to evaluate the ICDF for energy loss at.
	 - n_omega_q: The "energy loss axis" for the momentum ICDF
	 - P_q:       The probabilities to evaluate the ICDF for momentum transfer at.
	 - F:         The Fermi energy of the material

	Returns:
	 - Inverse mean free path, same length as K
	 - Stopping power, same length as K
	 - ICDF for energy loss, shape (len(K) × len(P_omega))
	 - 2D ICDF for momentum transfer, shape (len(K) × n_omega_q × len(P_q))
	"""

	K_units = units.eV
	q_units = units('nm^-1')

	mc2 = units.m_e * units.c**2

	# Helper function, sqrt(2m/hbar^2 * K(1 + K/mc^2)), appears when getting
	# momentum boundaries from kinetic energy
	q_k = lambda _k : np.sqrt(2*units.m_e * _k*(1 + _k/(2 * mc2))) / units.hbar;

	def elf(omega, q):
		# Linear interpolation, with extrapolation if out of bounds
		def find_index(a, v):
			low_i = np.clip(np.searchsorted(a, v, side='right')-1, 0, len(a)-2)
			vl = a[low_i]
			vh = a[low_i + 1]
			return low_i, (v - vl) / (vh - vl)
		low_x, frac_x = find_index(elf_omega.magnitude, omega.to(elf_omega.units).magnitude)
		low_y, frac_y = find_index(elf_q.magnitude, q.to(elf_q.units).magnitude)
		elf = (1-frac_x)*(1-frac_y) * elf_data[low_x, low_y] + \
			frac_x*(1-frac_y) * elf_data[low_x+1, low_y] + \
			(1-frac_x)*frac_y * elf_data[low_x, low_y+1] + \
			frac_x*frac_y * elf_data[low_x+1, low_y+1]

		elf[elf <= 0] = 0
		return elf

	def q_part(eval_omega, eval_q):
		# Returns DCS[i,j] = ∫_0^{eval_q[j]} dq/q ELF(eval_omega[i], q)
		dcs_data = np.divide(
			elf(eval_omega[:,np.newaxis], eval_q),
			eval_q.magnitude)
		for i in range(len(eval_omega)):
			dcs_data[i,1:] = cumtrapz(dcs_data[i,:], eval_q.magnitude)
			dcs_data[i,0] = 0
		return dcs_data

	eval_omega = np.geomspace(
		elf_omega[0].to(K_units).magnitude,
		(K[-1]-F).to(K_units).magnitude,
		10000) * K_units
	eval_q = np.geomspace(
		q_k(elf_omega[0]).to(q_units).magnitude,
		2*q_k(K[-1]).to(q_units).magnitude,
		10000) * q_units
	dcs_data = q_part(eval_omega, eval_q)

	inel_imfp = np.zeros(K.shape) * units('nm^-1')
	inel_sp = np.zeros(K.shape) * units('eV/nm')
	inel_omega_icdf = np.zeros((K.shape[0], P_omega.shape[0])) * K_units
	inel_q_2dicdf = np.zeros((K.shape[0], n_omega_q, P_q.shape[0])) * q_units

	for i, E in enumerate(K):
		tcs, sp, omega_icdf, q_2dicdf = tcs_2dicdf(
			dcs_data[eval_omega<E-F,:],
			eval_omega[eval_omega < E-F], eval_q,
			lambda omega : q_k(E) - q_k(np.maximum(0*units.eV, E-omega)),
			lambda omega : q_k(E) + q_k(np.maximum(0*units.eV, E-omega)),
			P_omega,
			np.linspace(0, (E-F).to(K_units).magnitude, n_omega_q)*K_units,
			P_q);
		tcs /= np.pi * units.a_0 * .5*(1 - 1 / (E/mc2 + 1)**2) * mc2
		sp /= np.pi * units.a_0 * .5*(1 - 1 / (E/mc2 + 1)**2) * mc2

		inel_imfp[i] = tcs.to('nm^-1')
		inel_sp[i] = sp.to('eV/nm')
		inel_omega_icdf[i] = omega_icdf.to('eV')
		inel_q_2dicdf[i] = q_2dicdf.to('nm^-1')
		print('.', end='', flush=True)
	print()

	return inel_imfp, inel_sp, inel_omega_icdf, inel_q_2dicdf


def tcs_2dicdf(function_data, # Cumulative integral of ELF over dq/q
	eval_x, eval_y,           # ω and q values for which function_data was tabulated
	y_low_f, y_high_f,        # Range of q values which are interesting
	P_x,                      # Probability for ω ICDF
	x2d_axis, P_y):           # Axes for q 2D ICDF
	"""Compute integrated cross section and inverse cumulative distribution
	functions (ICDFs) for an optical data model.

	This function requires the energy-loss function, Im[-1/ε], to be
	pre-integrated over dq/q (you should provide the cumulative integral). This
	pre-integrated ELF is given to function_data, which should be a numpy array
	of shape (len(eval_x), len(eval_y)). eval_x and eval_y represent the ω and q
	values on which function_data has been evaluated.

	y_low_f and y_high_f must be functions of eval_x. They should return the
	minumum and maximum allowed q as function of ω.

	This function returns:
	  1. The total cross section, ∫dω ∫dq/q Im[1/ε]
	  2. The stopping power, ∫dω ω ∫dq/q Im[1/ε]
	  3. The ICDF for ω, for each P in P1d_axis.
	  4. The ICDF for q given ω. First index is ω, from axis x2d_axis; second
	     index is P, from P2d_axis.
	"""

	# Indices for integration boundaries
	yi_low = np.searchsorted(eval_y.magnitude, y_low_f(eval_x).to(eval_y.units).magnitude)
	yi_high = np.searchsorted(eval_y.magnitude, y_high_f(eval_x).to(eval_y.units).magnitude)

	# CIx[i] = ∫_{eval_x[0]}^{eval_x[i]} dx' ∫_{y_low(x')}^{y_high(x')} dy' f(x', y')
	#        = ∫_{eval_x[0]}^{eval_x[i]} dx' CIy_x[x', -1]
	CIx = np.zeros(len(eval_x))
	CIx[1:] = cumtrapz(
		function_data[range(len(eval_x)),yi_high] - function_data[range(len(eval_x)),yi_low],
		eval_x.magnitude)

	# total_cs = ∫_{eval_x[0]}^{eval_x[-1]} dx' ∫_{y_low(x')}^{y_high(x')} dy' f(x', y')
	#          = CIx[-1]
	total_cs = CIx[-1]
	if total_cs <= 0:
		return (
			0 * eval_x.units,
			0 * eval_x.units * eval_x.units,
			np.zeros(len(P_x)) * eval_x.units,
			np.zeros((len(x2d_axis), len(P_y))) * eval_y.units
		)

	# CDF for x
	# P[i] = ∫_{eval_x[0]}^{eval_x[i]} dx' p(x')
	#      = ∫_{eval_x[0]}^{eval_x[i]} dx' ∫_{y_low(x')}^{y_high(x')} f(x', y') / total_cs
	cPx = CIx / total_cs

	# ICDF for omega
	icdf_x = icdf(eval_x, cPx, P_x)

	# ICDF for q
	icdf_yx = np.zeros([len(x2d_axis), len(P_y)]) * eval_y.units
	for j, xx in enumerate(x2d_axis):
		i_eval_x = min(
			np.searchsorted(eval_x.to(xx.units).magnitude, xx.magnitude),
			len(eval_x) - 2)

		if yi_low[i_eval_x] == yi_high[i_eval_x]:
			cdf1 = np.zeros(len(eval_y))
		else:
			cdf1 = np.copy(function_data[i_eval_x])
			cdf1 -= cdf1[yi_low[i_eval_x]]
			cdf1 /= cdf1[yi_high[i_eval_x]]

		if yi_low[i_eval_x+1] == yi_high[i_eval_x+1]:
			cdf2 = np.zeros(len(eval_y))
		else:
			cdf2 = np.copy(function_data[i_eval_x+1])
			cdf2 -= cdf2[yi_low[i_eval_x+1]]
			cdf2 /= cdf2[yi_high[i_eval_x+1]]

		frac = (xx - eval_x[i_eval_x]) / (eval_x[i_eval_x+1] - eval_x[i_eval_x])
		cdf = (1 - frac)*cdf1.clip(0, 1) + frac*cdf2.clip(0, 1)
		icdf_yx[j,:] = icdf(eval_y, cdf, P_y)

	# Stopping power
	SP = trapz(eval_x.magnitude *
		(function_data[range(len(eval_x)),yi_high] - function_data[range(len(eval_x)),yi_low]),
		eval_x.magnitude)

	return (
		total_cs * eval_x.units,
		SP * eval_x.units * eval_x.units,
		icdf_x,
		icdf_yx)
