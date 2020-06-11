import os
import numpy as np
import yaml
from cstool.common import units
from .element import element
from .band_structure import band_structure
from .optical import optical
from .phonon import phonon

class param_file:
	"""Top-level file parameters class.

	Loads the parameters from file, stores all values.

	Properties:
	 - name:           Name of the material
	 - density:        Mass density
	 - elements:       Array of elements classes
	 - band_structure: Instance of band_structure class
	 - optical:        Instance of optical class
	 - phonon:         Instance of phonon class
	"""

	def __init__(self, filename):
		basedir = os.path.dirname(os.path.abspath(filename))
		with open(filename, encoding='utf-8') as file:
			data = yaml.safe_load(file)
			self.load(data, basedir)

	def load(self, yaml_data, basedir):
		try:
			self.name = yaml_data.get('name')
			self.density = units(yaml_data['density'])
			self.elements = [element(*el) for el in yaml_data['elements'].items()]
			self.band_structure = band_structure(yaml_data['band_structure'])
			self.optical = optical(yaml_data['optical'], basedir)
			self.phonon = phonon(yaml_data['phonon'], self)
		except KeyError as ke:
			raise RuntimeError("Expected key {} was not found in parameters file".format(ke))

		if not self.density.is_compatible_with(units.gram/units.cubic_centimeter):
			raise RuntimeError("Provided density has inconsistent units")

	def get_rho_n(self):
		"""Get the "number density" of unit cells in the substance, i.e. number
		of unit cells per unit volume."""
		m_cell = sum([el.M*el.count for el in self.elements]) / units.N_A
		return self.density / m_cell
