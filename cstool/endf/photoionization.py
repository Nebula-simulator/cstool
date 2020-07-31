import numpy as np
import warnings
from cstool.common import units
from cstool.common.interpolate import interp_ll
from .obtain_endf_files import obtain_endf_files
from .parse_endf import parse_electrons, parse_photoat

def get_cs(K, Z):
	"""Get photoatomic ionization cross sections.

	Parameters:
	 - K: Photon energy or energies of interest
	 - Z: Atomic number of target atom

	Returns two lists:
	 - Photoionization cross sections σ(K) for each shell
	 - Binding energy for each shell
	"""
	endf_files = obtain_endf_files()

	# parse_photoat() will warn that it does not know how to read file 27.
	# That's OK, we don't need it anyway. So suppress the warning.
	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		e_data = parse_photoat(endf_files['photoat'], Z)

	cs = []
	binding = []
	for rx in e_data.reactions.values():
		if rx.MT <= 533:
			continue

		cs.append(interp_ll(
			rx.cross_section.x*units.eV, rx.cross_section.y*units.barn,
			left=0*units.barn)(K))
		binding.append(rx.binding_energy * units.eV)
	return cs, binding

@units.check(None, units.eV, units.dimensionless)
def compile_photoionization_icdf(material_params, K, p_ion):
	"""Compile inverse cumumlative distribution function for photoionization.

	Parameters:
	 - material_params: Material parameters class
	 - K:               Photon energies of interest
	 - p_ion:           Ionization probabilities (between 0 and 1) of interest

	Returns the ICDF for binding energies, shape len(K)×len(p_ion).
	"""

	# Generate sorted list of shells, sorted by binding energy.
	# Each shell is represented by a dict:
	#   - B: binding energy
	#   - IMFP: (array, shape of K): Differential inverse mean free path
	# Also get total IMFP of all shells
	shells = []
	for element in material_params.elements:
		cs, binding = get_cs(K, element.Z)

		for shelli in range(len(binding)):
			imfp = cs[shelli] * element.count * material_params.get_rho_n()
			shells.append({
				'B': binding[shelli],
				'IMFP': imfp
			})
	shells.sort(key = lambda s : s['B'])

	# Compute the running cumulative inverse mean free paths for each shell
	total_IMFP = np.zeros(K.shape)
	for shell in shells:
		total_IMFP += shell['IMFP']
		shell['IMFP_cum'] = np.copy(total_IMFP.magnitude) * total_IMFP.units
	# shell['P_cum'] is the cumulative probability for this shell and others before it.
	for shell in shells:
		shell['P_cum'] = np.divide(shell['IMFP_cum'], total_IMFP,
			out = np.zeros(shell['IMFP_cum'].shape),
			where = total_IMFP > 0*total_IMFP.units)

	# Compute ICDF from P_cum.
	ionization_icdf = np.zeros((K.shape[0], p_ion.shape[0]))*units.eV
	ionization_icdf[:] = np.nan
	for shell in reversed(shells):
		ionization_icdf[np.logical_and(
			shell['IMFP'][:,np.newaxis] > 0/units.nm,
			p_ion[np.newaxis,:] <= shell['P_cum'][:,np.newaxis]
		)] = shell['B']

	return ionization_icdf
