from cstool.common import units
from .helpers import load_quantity

class element:
	"""One of the elements (e.g. C, O, Si) in the substance.

	Properties:
	 - name:  Name
	 - count: Number per unit cell
	 - Z:     Atomic number
	 - M:     Standard atomic weight, in g/mol
	"""

	def __init__(self, name, yaml_data):
		self.name = name
		self.count = int(yaml_data['count'])
		self.Z = int(yaml_data['Z'])
		self.M = load_quantity(yaml_data, 'M', units.gram/units.mole)

		if self.count <= 0:
			raise RuntimeError("'Count' for element {} is invalid".format(self.name))

		if self.Z <= 0:
			raise RuntimeError("Atomic number for element {} is invalid".format(self.name))

