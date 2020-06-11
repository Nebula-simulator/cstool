import numpy as np
from cstool.common import units
from cstool.common.integrate import quad

def ac_phonon_loss(material_params, T=297*units.K):
	if material_params.phonon.use_dualbranch():
		l_lon = ac_phonon_loss_raw(
			material_params.phonon.lattice,
			material_params.phonon.get_lon_c_s(),
			material_params.phonon.get_lon_alpha(),
			T)
		l_tra = ac_phonon_loss_raw(
			material_params.phonon.lattice,
			material_params.phonon.get_tra_c_s(),
			material_params.phonon.get_tra_alpha(),
			T)
		return (l_lon + 2*l_tra) / 3
	else:
		return ac_phonon_loss_raw(
			material_params.phonon.lattice,
			material_params.phonon.get_iso_c_s(),
			material_params.phonon.get_iso_alpha(),
			T)

@units.check(
	units.angstrom,
	units.m/units.s,
	units.m**2/units.s,
	units.K)
def ac_phonon_loss_raw(lattice, c_s, alpha, T=297*units.K):
	"""Compute the net average energy loss to acoustic phonons

	Parameters:
	 - lattice: Lattice constant (Å)
	 - c_s:     Speed of sound (m/s)
	 - alpha:   Band bending parameters (m²/s)
	 - T:       Temperature (K)
	"""

	# Wave factor at 1st Brillouin Zone Boundary
	k_BZ = 2 * np.pi / lattice
	kB_T = units.boltzmann_constant * T

	# Average net loss per acoustic scattering event
	def hbar_w_AC(k):
		return units.hbar * (c_s * k - alpha * k**2)

	# Bose-Einstein distribution
	def N_BE(k):
		return 1 / np.expm1(hbar_w_AC(k) / kB_T)

	def nominator(k):
		return hbar_w_AC(k) * k**2

	def denominator(k):
		return (2 * N_BE(k) + 1) * k**2

	y1, err1 = quad(nominator, 0/units.nm, k_BZ)
	y2, err2 = quad(denominator, 0/units.nm, k_BZ)

	return (y1 / y2).to('eV')

