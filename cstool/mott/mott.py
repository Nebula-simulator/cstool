import numpy as np
from multiprocessing.pool import ThreadPool
from cstool.common import units
from cstool.common import dimfp_table
from .elsepa_input import elscata_settings
from .elsepa_output import get_dcs_filename
from .run_elsepa import run_elscata


# Pint does not support numpy.array_split...
def pint_array_split(array, indices_or_sections):
	return np.array_split(array.magnitude, indices_or_sections) * array.units

def run_elscata_helper(Z, energies):
	# There are no muffin-potential radii for atomic numbers 1, 7, 8
	no_muffin_Z = [1, 7, 8]

	settings = elscata_settings(energies, Z,
		IHEF=0,
		MCPOL=2,
		MUFFIN=False if Z in no_muffin_Z else True)

	return run_elscata(settings)

def run_elscata_parallel(Zs, energies, threads):
	"""Run elscata in parallel, with the default settings.

	Parameters:
	 - Zs:       list of atomic numbers
	 - energies: list of electron energies
	 - threads:  number of parallel execution threads for ELSEPA.

	Returns:
	 - A nested list.
	   The first index corresponds to the index of the Zs array,
	   the second index corresponds to the index in the energies array.
	"""
	with ThreadPool(threads) as pool:
		# Split the energies array into chunks
		energy_chunks = pint_array_split(energies, min(4*threads, len(energies)))

		# Run ELSEPA in parallel
		results = pool.starmap(run_elscata_helper,
			[(Z, ec) for Z in Zs for ec in energy_chunks])

		# Combine the energy chunks
		# Data will be accessible as elsepa_data[Z_idx][energy_idx]
		elsepa_data = []
		for iz in range(len(Zs)):
			iz_data = []
			for ic, Ec in enumerate(energy_chunks):
				chunk_dict = results[iz*len(energy_chunks) + ic]
				for E in Ec:
					iz_data.append(chunk_dict[get_dcs_filename(E)])
			elsepa_data.append(iz_data)

	return elsepa_data


@units.check(None, units.eV, None)
def mott_dimfp(material_params, energies, threads=4):
	"""Get differential inverse mean free path (DIMFP) for Mott scattering.

	Returns a DIMFP_table class, where the 'q' axis represents cosθ. The units
	of the DIMFP are nm⁻¹ sr⁻¹, so if you get the inverse mean free path from
	the DIMFP_table class, you still need to multiply by 2π to integrate out the
	azimuthal angle.
	"""
	elsepa_data = run_elscata_parallel(
		[el.Z for el in material_params.elements],
		energies, threads)

	theta = elsepa_data[0][0].THETA
	dimfp = np.zeros((len(energies), len(theta))) / units.nm / units.sr
	for iE, E in enumerate(energies):
		for iZ, element in enumerate(material_params.elements):
			dimfp[iE] += elsepa_data[iZ][iE].DCS * element.count * material_params.get_rho_n()

			# Sanity check
			if not np.allclose(theta, elsepa_data[iZ][iE].THETA):
				raise RuntimeError("ELSEPA is unexpectedly returning different"
					" scattering angles for different atomic numbers/energies."
					" Please contact the developers.")

	# Reverse along the second axis to make cosθ strictly ascending
	return dimfp_table(energies, np.cos(theta)[::-1], dimfp[:,::-1])
