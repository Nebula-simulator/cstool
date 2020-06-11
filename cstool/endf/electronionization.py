import numpy as np
from scipy.interpolate import griddata
from cstool.common import units
from cstool.common.interpolate import interp_ll
from .obtain_endf_files import obtain_endf_files
from .parse_endf import parse_electrons

def get_dcs_loss(K, omega, Z):
	"""Get electron-ionization cross sections.

	Parameters:
	 - K:     Electron kinetic energies of interest
	 - omega: Electron energy losses of interest
	 - Z:     Atomic number of target atom

	Returns two lists:
	 - Photoionization cross sections dσ/dω for each shell
	 - Binding energy for each shell
	"""
	endf_files = obtain_endf_files()
	e_data = parse_electrons(endf_files['electrons'], Z)

	dcs = []
	binding = []

	for rx in e_data.reactions.values():
		if rx.MT <= 533:
			continue

		CS_shell = interp_ll(
			rx.cross_section.x*units.eV, rx.cross_section.y*units.barn,
			left=0*units.barn)(K)

		# Read cross section data
		primary_K = [p['E1'] for p in rx.products if p['ZAP'] == 11][0] # eV
		recoil_E = [[pep.flatten() for pep in p['Ep']]
				for p in rx.products if (p['ZAP'] == 11 and p['LAW'] == 1)][0] # eV
		recoil_P = [[pb.flatten() for pb in p['b']]
				for p in rx.products if (p['ZAP'] == 11 and p['LAW'] == 1)][0] # eV^-1

		# Make symmetry in recoil_E and recoil_P explicit
		for i, KK in enumerate(primary_K):
			re = recoil_E[i]
			rp = recoil_P[i]

			recoil_E[i] = np.r_[re,
				KK-rx.binding_energy-re[-2::-1]]
			recoil_P[i] = np.r_[.5*rp,
				.5*rp[-2::-1]]

		# Obtain differential cross sections for desired K, omega
		# by log-log-log interpolation
		dcs.append(np.exp(griddata((
			np.log(np.repeat(primary_K, [len(a) for a in recoil_E])),
			np.log(np.concatenate(recoil_E) + rx.binding_energy)),
			np.log(np.concatenate([recoil_P[i] for i in range(len(primary_K))])),
			(np.log(K.to(units.eV).magnitude), np.log(omega.to(units.eV).magnitude)),
			fill_value = -np.inf
		))*CS_shell/units.eV)

		binding.append(rx.binding_energy*units.eV)

	return dcs, binding


@units.check(None, units.eV, units.dimensionless, units.dimensionless)
def compile_electronionization_icdf(material_params, K, omega_frac, P):
	"""Compile inverse cumumlative distribution function for electron-ionization.

	Parameters:
	 - material_params: Material parameters class
	 - K:               Photon energies of interest
	 - omega_fac:       Fractional energy losses of interest
	                    (as fraction of K, between 0 and 1)
	 - P:               Ionization probabilities (between 0 and 1) of interest

	Returns the ICDF for binding energies, shape len(K)×len(omega_frac)×len(P).
	"""
	# len(K) x len(omega_frac) array of interesting energy losses
	omega = K[:, np.newaxis] * omega_frac[np.newaxis, :]

	# Generate sorted list of shells, sorted by binding energy.
	# Each shell is represented by a dict:
	#   - B: binding energy
	#   - DIMFP (array, shape of omega): Differential inverse mean free path
	shells = []
	for element in material_params.elements:
		dcs_element, binding_element = get_dcs_loss(
			K.repeat(omega.shape[1]).reshape(omega.shape),
			omega, element.Z)

		for shelli in range(len(dcs_element)):
			shells.append({
				'B': binding_element[shelli],
				'DIMFP': dcs_element[shelli] * element.count * material_params.get_rho_n()
			})
	shells.sort(key = lambda s : s['B'])

	# Compute the running cumulative differential inverse mean free paths
	# for each shell, and then normalize
	total_DIMFP = np.zeros(omega.shape) / units.nm / units.eV
	for shell in shells:
		total_DIMFP += shell['DIMFP']
		shell['DIMFP_cum'] = np.copy(total_DIMFP)
	for shell in shells:
		shell['P_cum'] = np.divide(shell['DIMFP_cum'], total_DIMFP,
			out = np.zeros(shell['DIMFP_cum'].shape),
			where = total_DIMFP > 0*total_DIMFP.units)

	# Build Inverse Cumulative Distribution Function
	icdf = np.zeros((len(K), len(omega_frac), len(P))) * units.eV
	icdf[:] = np.nan
	for shell in reversed(shells):
		icdf[np.logical_and(
				shell['DIMFP'][:,:,np.newaxis] > 0*shell['DIMFP'].units,
				P[np.newaxis, np.newaxis, :] <= shell['P_cum'][:,:,np.newaxis]
			)] = shell['B']

	return icdf
