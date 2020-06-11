import numpy as np
from cstool.common import units

# Reading output files from ELSEPA
# doi:10.1016/j.cpc.2004.09.006
# http://cpc.cs.qub.ac.uk/cpc/summaries/ADUS

def get_dcs_filename(energy):
	# Get filename for a DCS output file.
	# Slightly mangled because of the way ELSEPA rounds energies
	s = 'dcs_{:.4e}'.format(energy.to('eV').magnitude) \
		.replace('.', 'p').replace('+', '')
	return s[:9] + s[10:] + '.dat'

class dcs_parser:
	"""Parser for ELSEPA's dcs_xxxx files.

	Properties:
	 - THETA:   Scattering angle
	 - MU:      (1 - cos(THETA)) / 2
	 - DCS:     Differential cross section
	 - Sherman: Sherman function
	 - Error:   Error estimate
	"""
	def __init__(self, filename):
		data = np.loadtxt(filename, unpack=True)
		if data.shape[0] != 6:
			raise RuntimeError("dcs file from ELSEPA does not follow expected format")

		self.THETA   = data[0] * units.degrees
		self.MU      = data[1]
		self.DCS     = data[3] * units.a0**2 / units.sr
		self.Sherman = data[4]
		self.error   = data[5]

