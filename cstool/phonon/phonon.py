# Based on Schreiber & Fitting (doi:10.1016/S0368-2048(01)00368-1)
import numpy as np
from cstool.common import units

def ac_phonon_dimfp(material_params, T=297*units.K):
	if material_params.phonon.use_dualbranch():
		f_lon = ac_phonon_dimfp_raw(
			material_params.density,
			material_params.phonon.lattice,
			material_params.phonon.get_lon_ac_def(),
			material_params.phonon.get_lon_c_s(),
			material_params.phonon.get_m_eff(),
			material_params.phonon.get_m_dos(),
			material_params.phonon.get_lon_alpha(),
			T)
		f_tra = ac_phonon_dimfp_raw(
			material_params.density,
			material_params.phonon.lattice,
			material_params.phonon.get_tra_ac_def(),
			material_params.phonon.get_tra_c_s(),
			material_params.phonon.get_m_eff(),
			material_params.phonon.get_m_dos(),
			material_params.phonon.get_tra_alpha(),
			T)

		def f_all(*args):
			return f_lon(*args) + 2*f_tra(*args)
		return f_all
	else:
		f_single = ac_phonon_dimfp_raw(
			material_params.density,
			material_params.phonon.lattice,
			material_params.phonon.get_iso_ac_def(),
			material_params.phonon.get_iso_c_s(),
			material_params.phonon.get_m_eff(),
			material_params.phonon.get_m_dos(),
			material_params.phonon.get_iso_alpha(),
			T)

		# Factor 3: 2 transverse, 1 longitudinal mode
		def f_all(*args):
			return 3*f_single(*args)
		return f_all


@units.check(
	units.g/units.cm**3,
	units.angstrom,
	units.eV,
	units.m/units.s,
	units.g,
	units.g,
	units.m**2/units.s,
	units.K)
def ac_phonon_dimfp_raw(rho_m, lattice, eps_ac, c_s, m_eff, m_dos, alpha, T=297*units.K):
	"""Compute differential electron-acoustic phonon scatering cross sections.

	Parameters should be provided as pint quantities. Canonical dimensionalities
	are listed below for clarity, they are not required.

	Parameters:
	 - rho_m:   Mass density (g/cm³)
	 - lattice: Lattice constant (Å)
	 - eps_ac:  Acoustic deformation potential (eV)
	 - c_s:     Isotropic speed of sound (m/s)
	 - m_eff:   Effective mass (g)
	 - m_dos:   Density of states mass (g)
	 - alpha:   Band bending parameter (m²/s)
	 - T:       Temperature (K)

	Returns:
	 - A function, which takes an energy array and an angle array. It returns
	   the differential inverse mean free path per unit solid angle, as a 2D array.
	"""

	# Brillouin zone energy
	k_BZ = 2 * np.pi / lattice
	E_BZ = (units.hbar * k_BZ)**2 / (2*units.m_e)

	# A: screening factor (eV); 5 is constant for every material.
	A = 5 * E_BZ

	# Helper quantities
	hbar_w_BZ = units.hbar * (c_s * k_BZ - alpha * k_BZ**2)
	kB_T = units.boltzmann_constant * T

	# Phonon population density
	n_BZ = 1 / np.expm1(hbar_w_BZ / kB_T)

	# Inverse MFP multiplier
	L_inv = ((np.sqrt(m_eff * m_dos**3) * eps_ac**2 * kB_T) /
		(np.pi * units.hbar**4 * c_s**2 * rho_m))

	# Differential inverse mean free path for energies < E_BZ / 4
	def dimfp_low(mu, E):
		# mu: [1 - cos(theta)] / 2
		return L_inv / (4*np.pi) / (1 + mu*E/A)**2

	# Differential inverse mean free path for energies > E_BZ
	def dimfp_high(mu, E):
		return L_inv / (4*np.pi) *                                \
			(8 * m_dos * c_s**2 * A) / (hbar_w_BZ * kB_T) * \
			(n_BZ + .5) *                                         \
			(mu*E/A) / (1 + mu*E/A)**2

	# Linear interpolation
	def dimfp(E, cos_theta):
		mu = .5 * (1 - cos_theta)

		# Linear interpolation between dcs_low and dcs_high,
		# between E_BZ/4 and E_BZ
		weight = np.clip(
			-4*E/(3*E_BZ) + 4/3,
			0, 1)
		dimfp_interp = weight*dimfp_low(mu, E) + (1-weight)*dimfp_high(mu, E)

		return dimfp_interp.to('nm⁻¹/sr')

	return dimfp
