import os
import numpy as np
from cstool.common import units
from cstool.common.interpolate import interp_ll

class optical:
	"""Referes to optical data.

	Properties:
	 - nk_file:
	 - df_file:
	"""
	def __init__(self, yaml_data, basedir):
		try:
			self.df_file = os.path.join(basedir, yaml_data.get('df_file'))
		except:
			raise RuntimeError("No optical data provided")

		self._load_df_file()

	def get_elf(self):
		"""Get the optical energy-loss function (ELF), Im[1/ε(ω)].

		This function returns another function which can be called to get Im[1/ε(ω)].
		"""
		return interp_ll(self.energy_data, self.elf_data, left=0)

	def get_outer_shells(self):
		if hasattr(self, 'outer_shells'):
			return np.copy(self.outer_shells)
		else:
			return np.array([])*units.eV

	def get_min_energy(self):
		"""Get lowest energy tabulated."""
		return self.energy_data[0]
	def get_max_energy(self):
		"""Get highest energy tabulated."""
		return self.energy_data[-1]

	def get_serial_elf(self):
		"""Get the optical energy-loss function (ELF), 1/ε(ω), on a regularly
		spaced grid.

		The resulting data set will be much bigger than usual, but the regular
		grid can be used for a much faster lookup of data.

		The grid will have equal steps in logarithmic space. That is, the i'th
		point will be for log(ω) = a + bi. The values of a and b are the first
		two return values from this functions. The ELF values are also returned
		as logarithms, for easy log-log interpolation.

		Returns:
		 - The lowest log(ω), where ω is in hartree (atomic units)
		 - Step size in log(ω)
		 - Number of data points
		 - Array of log(Im[1/ε(ω)])
		"""
		log_x = np.log(self.energy_data.to('hartree').magnitude)
		log_y = np.log(self.elf_data)
		min_interval = np.min(log_x[1:] - log_x[:-1])

		lx_min = log_x[0]
		lx_max = log_x[-1]
		N_points = int(np.ceil((lx_max - lx_min) / min_interval));
		lx_step = (lx_max - lx_min) / (N_points - 1);

		ly_data = np.interp(
			np.linspace(lx_min, lx_max, N_points),
			log_x, log_y);

		return lx_min, lx_step, N_points, ly_data;

	def _load_df_file(self):
		# File format is quite strict:
		# First line is outer-shell energies in eV, terminated by -1. The
		# following lines are energy in eV and corresponding 1/ε(ω), separated
		# by whitespace. The last line is two times -1.
		with open(self.df_file) as file:
			self.outer_shells = np.array([float(v) for v in file.readline().split()[:-1]])*units.eV

			data = np.loadtxt(file, unpack=True)
			self.energy_data = data[0,:-1] * units.eV
			self.elf_data = data[1,:-1]
