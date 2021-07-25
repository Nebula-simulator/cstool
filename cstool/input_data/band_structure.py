from cstool.common import units
from .helpers import load_quantity

class band_structure:
	"""Simple band structure model.

	Properties:
	 - model:         Can be insulator, semiconductor or metal.
	                  The insulator and semiconductor models are equivalent.
	 - fermi:         For metals, the position of the Fermi energy above the
	                  bottom of the band.
	 - work_function: For metals, the work function.
	 - band_gap:      For insulators and semiconductors, the band gap.
	 - affinity:      For insulators and semiconductors, the electron affinity.
	"""

	def __init__(self, yaml_data):
		self.model = yaml_data['model']

		self.fermi         = load_quantity(yaml_data, 'fermi', units.eV) if self.model == 'metal' else None
		self.work_function = load_quantity(yaml_data, 'work_function', units.eV) if self.model == 'metal' else None

		self.valence  = load_quantity(yaml_data, 'valence', units.eV) if self.model != 'metal' else None
		self.band_gap = load_quantity(yaml_data, 'band_gap', units.eV) if self.model != 'metal' else None
		self.affinity = load_quantity(yaml_data, 'affinity', units.eV) if self.model != 'metal' else None

		if self.model not in ['insulator', 'semiconductor', 'metal']:
			raise RuntimeError("Unknown band structure model {}, expected"
				" insulator, semiconductor or metal.")

	def get_fermi(self):
		"""Get Fermi energy above the bottom of the band.

		If this is a semiconductor or an insulator, it is estimated as halfway
		inside the band gap.
		"""
		if self.fermi is not None:
			return self.fermi

		return self.valence + .5 * self.band_gap

	def get_min_excitation(self):
		"""Get the minimum energy where an electron can excite a secondary
		electron.

		For metals, this is the Fermi energy.
		For semiconductors and insulators, this is the bottom of the conduction
		band plus the band gap, because both the primary and secondary electron
		must be in the conduction band after the event.
		"""
		if self.model == 'metal':
				return self.fermi
		return self.valence + self.band_gap

	def get_barrier(self):
		"""Get the "inner potential" for this material."""
		if self.model == 'metal':
			return self.fermi + self.work_function
		else:
			return self.valence + self.band_gap + self.affinity
