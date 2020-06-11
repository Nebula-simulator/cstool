import numpy as np
from cstool.common import units
from .helpers import load_quantity

class phonon:
	"""Holds phonon parameters.

	Properties:
	 - lattice:       Lattice constant
	 - m_eff:         Effective mass
	 - m_dos:         Density-of-states mass

	 - i_ac_def:      Acoustic deformation potential (isotropic)
	 - i_c_s:         Speed of sound (isotropic)
	 - i_alpha:       Bending of the dispersion relation near the Brillouin zone edge
	 - i_resistivity: Optional, can be used instead of the acoustic deformation
	                  potential if that's unknown.

	 - l_ac_def:      Acoustic deformation potential (longitudinal)
	 - l_c_s:         Speed of sound (longitudinal)
	 - l_alpha:       Bending of the dispersion relation near the Brillouin zone edge

	 - t_ac_def:      Acoustic deformation potential (transversal)
	 - t_c_s:         Speed of sound (transversal)
	 - t_alpha:       Bending of the dispersion relation near the Brillouin zone edge
	"""

	def __init__(self, yaml_data, parameters):
		self.lattice = load_quantity(yaml_data, 'lattice', units.m)
		self.m_eff = load_quantity(yaml_data, 'm_eff', units.g, True)
		self.m_dos = load_quantity(yaml_data, 'm_dos', units.g, True)


		# Isotropic
		self.i_ac_def = load_quantity(yaml_data.get('isotropic', {}),
			'ac_def', units.eV, True)
		self.i_c_s = load_quantity(yaml_data.get('isotropic', {}),
			'c_s', units.m/units.s, True)
		self.i_alpha = load_quantity(yaml_data.get('isotropic', {}),
			'alpha', units.m**2/units.s, True)
		self.i_resistivity = load_quantity(yaml_data.get('isotropic', {}),
			'resistivity', units.ohm*units.m, True)

		# Longitudinal
		self.l_ac_def = load_quantity(yaml_data.get('longitudinal', {}),
			'ac_def', units.eV, True)
		self.l_c_s = load_quantity(yaml_data.get('longitudinal', {}),
			'c_s', units.m/units.s, True)
		self.l_alpha = load_quantity(yaml_data.get('longitudinal', {}),
			'alpha', units.m**2/units.s, True)

		# Transversal
		self.t_ac_def = load_quantity(yaml_data.get('transversal', {}),
			'ac_def', units.eV, True)
		self.t_c_s = load_quantity(yaml_data.get('transversal', {}),
			'c_s', units.m/units.s, True)
		self.t_alpha = load_quantity(yaml_data.get('transversal', {}),
			'alpha', units.m**2/units.s, True)


		self._parameters = parameters



	def get_m_eff(self):
		"""Get effective mass.

		If this is not provided, returns the electron mass."""
		if self.m_eff is not None:
			return self.m_eff
		else:
			return 1*units.m_e

	def get_m_dos(self):
		"""Get density-of-states mass.

		If this is not provided, returns the electron mass."""
		if self.m_dos is not None:
			return self.m_dos
		else:
			return 1*units.m_e

	def use_dualbranch(self):
		"""Returns whether or not the dual-branch model should be used.

		If not, the isotropic model should be used.
		"""
		has_transversal = self.l_ac_def is not None and self.l_c_s is not None
		has_longitudinal = self.t_ac_def is not None and self.t_c_s is not None
		return has_transversal and has_longitudinal



	def get_iso_c_s(self):
		"""Get speed of sound for models that require an isotropic version.

		If longitudinal and transversal speed of sounds are provided separately,
		they are averaged.
		"""
		if self.i_c_s is not None:
			return self.i_c_s
		elif self.l_c_s is not None and self.t_c_s is not None:
			return (2*self.t_c_s + self.l_c_s)/3
		else:
			raise RuntimeError("Phonon model: expected either an isotropic "
				"speed of sound, or a longitudinal and transversal one.")

	def get_iso_ac_def(self):
		"""Get the acoustic deformation potential.

		If this is not provided, it is estimated from the conductivity.
		"""
		if self.i_ac_def is not None:
			return self.i_ac_def

		if self.i_resistivity is None:
			raise RuntimeError("Phonon model: expected either the acoustic "
				"deformation potential, or the resisitvity to be given.")

		eps_sq = 2*units.hbar * (units.e)**2 / (3*np.pi*units.m_e) * \
			self.get_iso_c_s()**2 * self.i_resistivity * self._parameters.density * \
			self._parameters.band_structure.get_fermi() / (units.boltzmann_constant * 297*units.K)

		return np.sqrt(eps_sq)

	def get_iso_alpha(self):
		"""Get the α parameter, which describes bending of the phonon dispersion
		relation near the Brillouin zone edge. Default value is zero."""
		if self.i_alpha is None:
			return 0 * units.m**2/units.s
		return self.i_alpha

	def get_lon_c_s(self):
		"""Get the longitudinal speed of sound"""
		return self.l_c_s

	def get_lon_ac_def(self):
		"""Get the longitudinal acoustic deformation potential"""
		return self.l_ac_def

	def get_lon_alpha(self):
		"""Get the longitudinal band bending parameter"""
		if self.l_alpha is None:
			return 0 * units.m**2/units.s
		return self.l_alpha


	def get_tra_c_s(self):
		"""Get the transversal speed of sound"""
		return self.t_c_s

	def get_tra_ac_def(self):
		"""Get the transversal acoustic deformation potential"""
		return self.t_ac_def

	def get_tra_alpha(self):
		"""Get the transversal band bending parameter"""
		if self.t_alpha is None:
			return 0 * units.m**2/units.s
		return self.t_alpha
