import numpy as  np
from scipy.interpolate import RegularGridInterpolator
from cstool.common import units
from cstool.common.icdf_compile import compute_tcs_icdf

class dimfp_table:
	"""Differential inverse mean free path (DIMFP), tabulated from pre-computed
	values.

	Essentially a 2D table, with energy and a 'q' axis, where 'q' represents
	whatever the IMFP is differential to.
	"""

	def __init__(self, energy, q, dimfp):
		self.energy = energy
		self.q = q
		self.dimfp = dimfp

		assert energy.dimensionality == units.eV.dimensionality, \
			"Energy units check."
		assert dimfp.dimensionality == (units.m**-1 / q.units).dimensionality, \
			"DIMFP units check."
		assert dimfp.shape == (energy.size, q.size), \
			"Array dimensions do not match."

		self.interpolate_fn = RegularGridInterpolator((
			np.log(self.energy.magnitude),
			self.q.magnitude),
			self.dimfp.magnitude,
			bounds_error = False, fill_value = 0)

	def __call__(self, E, q):
		"""Get the interpolated differential inverse mean free path."""
		return self.interpolate_fn((
			np.log(E.to(self.energy.units).magnitude),
			q.to(self.q.units).magnitude)) * self.dimfp.units

	def IMFP_ICDF(self, E, P):
		"""Get the integrated inverse mean free path (IMFP) and inverse
		cumulative distribution function (ICDF).

		Parameters:
		 - E: Energies to evaluate the imfp and icdf at
		 - P: The cumulative probabilities for which the ICDF should be found.

		Returns:
		 - IMFP: numpy array of len(E)
		 - ICDF: numpy array of (len(E) × len(P))
		"""
		imfp = np.zeros(len(E)) * units('nm^-1')
		icdf = np.zeros((len(E), len(P_omega))) * self.q.units

		for i, K in enumerate(E):
			timfp, ticdf = compute_tcs_icdf(self, P, self.q)
			imfp[i] = timfp
			icdf[i] = ticdf

		return imfp, icdf
