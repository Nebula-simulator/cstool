import numpy as np
from cstool.common import units

@units.check(units.eV, units.eV, None)
def dimfp_kieft(K, w, material_params):
	"""Compute differential inverse mean free paths from the model of Kieft
	(doi:10.1088/0022-3727/41/21/215310). This model has been derived from
	Ashley's (doi:10.1016/0368-2048(88)80019-7).

	This function is to be called with a scalar parameter K for kinetic energy
	and a numpy arraw w for ω'. Returns a numpy array, with one value for each
	ω'."""
	mc2 = units.m_e * units.c**2
	elf_fn = material_params.optical.get_elf()
	return elf_fn(w) * L_Kieft(K, w, material_params.band_structure.get_fermi()) \
		/ (np.pi * units.a_0) \
		/ (1 - 1 / (K/mc2 + 1)**2) / mc2

def L_Kieft(K, w0, F):
	# For sqrt & log calls, we have to strip the units. pint does not like "where".

	a = (w0 / K).magnitude
	s = np.sqrt(1 - 2*a, where = (a <= .5), out = np.zeros(a.shape))

	L1_range = (a > 0) * (a < .5) * (K-F > w0) * (K > F)
	L2_range = (a > 0) * (K-F > w0) * (K > F)

	# Calculate L1
	x1 = np.divide(2, a, where=L1_range, out=np.zeros(a.shape)) * (1 + s) - 1
	x2 = K - F - w0
	x3 = K - F + w0
	L1 = 1.5 * np.log((x1 * x2 / x3).magnitude, where = L1_range, out = np.zeros(a.shape))

	# Calculate L2
	L2 = -np.log(a, where = L2_range, out = np.zeros(a.shape))

	return np.maximum(0, (w0 < 50 * units.eV) * L1 + (w0 > 50 * units.eV) * L2)

