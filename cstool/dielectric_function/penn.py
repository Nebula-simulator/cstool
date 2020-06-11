import numba as nb
from scipy.optimize import brentq
from scipy.integrate import quad, IntegrationWarning
from scipy import LowLevelCallable
import warnings
import math
import struct
import ctypes

"""
Implementation of the Full Penn Algorithm (doi:10.1103/PhysRevB.35.482), with
the method taken from Shinotsuka (doi:10.1002/sia.5789).

We use scipy for integration and root finding; numba to speed up the function
evaluations.

The user is expected to call penn(elf, omega, q), where elf is an ELF object;
and omega and q are numpy arrays with the appropriate units.

Internally, atomic units are used: energy is in hartree, length in Bohr radii.
"""

@nb.jit(nb.types.UniTuple(nb.f8,3)(nb.f8,nb.f8,nb.f8),nopython=True,nogil=True)
def kzx(q, omega, omega_p):
	kf = math.pow(3*math.pi / 4 * omega_p*omega_p, 1/3)
	Ef = .5 * kf*kf
	z = q / (2*kf)
	x = omega / Ef
	return kf, z, x

# Used in eq (14)
@nb.jit(nb.f8(nb.f8),nopython=True,nogil=True)
def F(t):
	if t+1 == 0 or t-1 == 0:
		return 0;
	return (1 - t*t) * math.log(abs( (t+1)/(t-1) ));

# Real part of Lindhard dielectric function, eq. (14)
@nb.jit(nb.f8(nb.f8,nb.f8,nb.f8),nopython=True,nogil=True)
def eps_r(q, omega, omega_p):
	kf, z, x = kzx(q, omega, omega_p)

	u = x / (4*z)
	v = 1 / (8*z)

	# Approximations for limiting cases
	if u < .01:
		return 1 + (.5 + ((1 - z*z - u*u)*math.log(abs((z+1)/(z-1)))
			+ (z*z - u*u - 1)*(2*u*u*z)/(z*z - 1)**2 ) / (4*z)) \
			/ (math.pi * kf * z*z);

	if u/(z+1) > 100:
		return 1 - (omega_p/omega)**2 * (1 + (z*z + 3/5)/(u*u))

	return 1 + (.5 + v*F(z - u) + v*F(z + u))/(math.pi * kf * z*z)

# Imaginary part of Lindhard dielectric function, eq. (15)
@nb.jit(nb.f8(nb.f8,nb.f8,nb.f8),nopython=True,nogil=True)
def eps_i(q, omega, omega_p):
	kf, z, x = kzx(q, omega, omega_p)
	u = x / (4*z)

	# Approximations for limiting cases
	if u < .01:
		return u / (2 * kf * z*z);
	if u/(z+1) > 100:
		return 0;

	# Exact version
	if (0<x) and (x < 4*z*(1 - z)):
		return x / (8 * kf * z**3);
	if (abs(4*z*(1 - z)) < x) and (x < 4*z*(1 + z)):
		return (1 - (z - u)**2) / (8 * kf * z**3);
	return 0;

# Im[-1 / Lindhard dielectric function]
@nb.jit(nb.f8(nb.f8,nb.f8,nb.f8),nopython=True,nogil=True)
def lindhard_elf(q, omega, omega_p):
	epsi = eps_i(q, omega, omega_p);
	epsr = eps_r(q, omega, omega_p);
	return epsi / (epsr*epsr + epsi*epsi);

# Get measured optical ELF by log-log interpolation (extrapolation)
@nb.jit(nb.f8(nb.f8, nb.f8,nb.f8,nb.i8,nb.f8[:]),nopython=True,nogil=True)
def elf_optical(omega, lx_min, lx_step, N_points, ly_data):
	true_index = (math.log(omega) - lx_min) / lx_step;
	low_index = int(max(0, min(true_index, N_points-2)));
	frac_index = true_index - low_index;

	log_y = (1-frac_index)*ly_data[low_index] \
		 + frac_index*ly_data[low_index+1];

	return math.exp(log_y);

# Get g() from measured ELF data, eq. (5).
@nb.jit(nb.f8(nb.f8, nb.f8,nb.f8,nb.i8,nb.f8[:]),nopython=True,nogil=True)
def g(omega, lx_min,lx_step,N_points,ly_data):
	return 2 * elf_optical(omega, lx_min, lx_step, N_points, ly_data) \
		/ (math.pi * omega)

# eq. (10)
@nb.jit(nb.f8(nb.f8,nb.f8,nb.f8),nopython=True,nogil=True)
def plasmon_slope(q, omega, omega_p):
	kf, z, x = kzx(q, omega, omega_p)

	a = z/x;
	if a < .01:
		L = -64*z*a*a/3 * (3 + 48*(1 + z*z)*a*a +
			256*(3 + z*z)*(1 + 3*z*z)*a**4)
	elif a > 100:
		b = x/(z*(z*z - 1))
		L = math.log(((z+1)/(z-1))**2) + 4*z*b*b * (1 + (1+z*z)*b*b +
			(3+z*z)*(1 + 3*z*z)*(b**4) / 3)
	else:
		Yp = z + x/(4*z)
		Ym = z - x/(4*z)
		L = math.log(abs((Ym+1)/(Ym-1))) + math.log(abs((Yp+1)/(Yp-1)))
	return L / (3 * math.pi * omega_p * q * z*z)

# Integrand for single excitation
# Most importantly a wrapper extracting the ELF table data from memory.
@nb.cfunc(nb.f8(nb.f8, nb.types.voidptr),nopython=True,nogil=True)
def single_integrand(omega0, data):
	q, omega, lx_min, lx_step = nb.carray(data, 4, dtype=nb.f8);
	N_points = nb.carray(data, 5, dtype=nb.i8)[4];
	ly_data = nb.carray(data, 5+N_points, dtype=nb.f8)[5:];

	return g(omega0, lx_min,lx_step,N_points,ly_data) * \
		lindhard_elf(q, omega, omega0);

# Get the single-excitation part of the Penn model.
# Parameters are in atomic units.
def single_part(q, omega, elfdata):
	if q == 0:
		return 0

	# Find lower integration boundary
	kf_min = max(0,
		(2 * omega - q*q) / (2 * q),
		(q*q - 2 * omega) / (2 * q)
	)
	omega0_min = math.sqrt(kf_min**3 * 4 / (3*math.pi))

	# Pack the ELF data into a block of memory, to pass to the single_integrand
	# function through scipy's LowLevelCallable interface.
	d = struct.pack('ddddq%dd' % elfdata[2],
		q, omega, *(elfdata[0:3]), *elfdata[3])
	user_data = ctypes.cast(d, ctypes.c_void_p)
	ll_integrand = LowLevelCallable(single_integrand.ctypes, user_data);

	# Ignore all integration warnings. Yes, I know this is horrible.
	# Fact is that even with convergence warnings, you get decent results.
	with warnings.catch_warnings():
		warnings.filterwarnings('ignore', category=IntegrationWarning)
		return quad(ll_integrand, omega0_min, math.inf)[0];

# Get the plasmon part of the Penn model.
# Parameters are in atomic units.
def plasmon_part(q, omega, elfdata):
	if q == 0:
		return elf_optical(omega, *elfdata)

	# The function whose root we need to find.
	# scipy.brentq does not support LowLevelCallables yet.
	def _e(omega_p):
		return eps_r(q, omega, omega_p);

	# Upper bound for finding the root
	kf_max = (2*omega - q*q) / (2*q)
	if kf_max <= 0:
		return 0

	omega0_max = math.sqrt(kf_max**3 * 4 / (3*math.pi))
	omega0_min = omega0_max / 1e6

	# Technically, we should use omega0_min == 0, but _e(0) diverges. For
	# _e(omega0_min) < 0, q is very small, so we just return the optical ELF.
	if _e(omega0_min) < 0:
		return elf_optical(omega, *elfdata)
	if _e(omega0_max) > 0:
		return 0;

	omega0 = brentq(_e, omega0_min, omega0_max)
	return g(omega0, *elfdata) * math.pi / abs(plasmon_slope(q, omega, omega0))

# Get the total Penn ELF, with parameters still in atomic units.
def penn_elf(q, omega, elfdata):
	return plasmon_part(q, omega, elfdata) + single_part(q, omega, elfdata)


import numpy as np
from functools import partial
from multiprocessing import Pool
from cstool.common import units

@units.check(None, units.eV, units.dimensionless, units.dimensionless)
def elf_full_penn(material_params, omega_max, N_omega, N_q):
	"""Get full Penn energy-loss function, Im[1/ε(ω,q)].

	The lowest ω value tabulated will equal whatever lowest value is available.
	The min and max q values are determined from the desired range of ω values.

	Parameters:
	 - material_params: Material parameters class
	 - omega_max:       Maximal value of ω of interest
	 - N_omega:         Number of ω data points
	 - N_q:             Number of q data points

	Returns:
	 - ω values
	 - q values
	 - 2D array of Im[1/ε(ω,q)], shape (N_omega × N_q)
	"""
	elf_data = material_params.optical.get_serial_elf()
	result = np.zeros([N_omega, N_q])

	omega_min = material_params.optical.get_min_energy()
	q_min = np.sqrt(2*units.m_e*omega_min)/units.hbar
	q_max = np.sqrt(2*units.m_e*omega_max)/units.hbar

	omega = np.geomspace(
		omega_min.to(omega_max.units).magnitude,
		omega_max.magnitude,
		N_omega) * omega_max.units;
	q = np.r_[0,
		np.geomspace(q_min.to(q_max.units).magnitude,
		q_max.magnitude, N_q-1)] * q_max.units

	with Pool() as pool:
		for iw, w in enumerate(omega):
			print("# Computing Penn ELF: {}%".format(int(100*iw / len(omega))), end='\r');
			result[iw,:] = pool.map(partial(penn_elf,
				omega=w.to('hartree').magnitude,
				elfdata=elf_data),
				q.to('a_0^-1').magnitude)
		print()

	return omega, q, result;
