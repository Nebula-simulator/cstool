import io
from cstool.common import units

# Generating input files for ELSEPA
# doi:10.1016/j.cpc.2004.09.006
# http://cpc.cs.qub.ac.uk/cpc/summaries/ADUS 
class elscata_settings:
	"""
	Properties:                                                                 default
	 - EV:     output kinetic energies (numpy array)
	 - IZ:     Atomic number
	 - IELEC:  Type of projectile (-1=electron, +1=positron)                      -1
	 - MNUCL:  Nuclear charge distribution (1=P, 2=U, 3=F, 4=Uu)                  3
	 - NELEC:  Number of bound electrons, use IZ if None                          None
	 - MELEC:  Electron distribution model (1=TFM, 2=TFD, 3=DHFS, 4=DF, 5=file)   4
	 - MUFFIN: False=free atom, True=muffin-tin model                             False
	 - RMUF:   Muffin-tin radius. None for measured, only used when MUFFIN=True   None
	 - MEXCH:  Electron exchange potential (0=none, 1=FM, 2=TF, 3=RT)             1
	 - MCPOL:  Correlation-polarization potential (0=none, 1=B, 2=LDA)            0
	 - VPOLA:  Atomic polarizability. None for experimental, only when MCPOL>0    None
	 - VPOLB:  Cut-off parameter b_pol. None for equation, only when MCPOL>0      None
	 - MABS:   Absorption correction (0=none, 1=LDA)                              0
	 - VABSA:  Absorption strength. Only when MABS=1                              2.0
	 - VABSD:  energy gap DELTA. None for equation, only when MABS=1              None
	 - IHEF:   Calculation method (0=full, 1=high-energy factorization, 2=Born)   1
	"""

	def __init__(self, EV, IZ,
			IELEC=-1,     MNUCL=3,    NELEC=None, MELEC=4,
			MUFFIN=False, RMUF=None,  MEXCH=1,
			MCPOL=0,      VPOLA=None, VPOLB=None,
			MABS=0,       VABSA=2.0,  VABSD=None, IHEF=1):
		self.IELEC  = IELEC
		self.EV     = EV
		self.IZ     = IZ
		self.MNUCL  = MNUCL
		self.NELEC  = NELEC
		self.MELEC  = MELEC
		self.MUFFIN = MUFFIN
		self.RMUF   = RMUF
		self.MEXCH  = MEXCH
		self.MCPOL  = MCPOL
		self.VPOLA  = VPOLA
		self.VPOLB  = VPOLB
		self.MABS   = MABS
		self.VABSA  = VABSA
		self.VABSD  = VABSD
		self.IHEF   = IHEF

	def generate_string(self):
		f = io.StringIO()

		self._write_line(f, 'IELEC',  self.IELEC)
		self._write_line(f, 'IZ',     self.IZ)
		self._write_line(f, 'MNUCL',  self.MNUCL)
		self._write_line(f, 'NELEC',  self.NELEC)
		self._write_line(f, 'MELEC',  self.MELEC)
		self._write_line(f, 'MUFFIN', int(self.MUFFIN))
		self._write_line(f, 'RMUF',   self.RMUF, units.cm)
		self._write_line(f, 'MEXCH',  self.MEXCH)
		self._write_line(f, 'MCPOL',  self.MCPOL)
		self._write_line(f, 'VPOLA',  self.VPOLA, units.cm**3)
		self._write_line(f, 'VPOLB',  self.VPOLB)
		self._write_line(f, 'MABS',   self.MABS)
		self._write_line(f, 'VABSA',  self.VABSA)
		self._write_line(f, 'VABSD',  self.VABSD, units.eV)
		self._write_line(f, 'IHEF',   self.IHEF)
		for E in self.EV:
			self._write_line(f, 'EV', E, units.eV)

		return f.getvalue()


	def _write_line(self, file, name, value, unit=None):
		if value is None:
			return

		if unit is None:
			file.write('{:6} {:<12}\n'.format(name, value))
		else:
			file.write('{:6} {:.4e}\n'.format(name, value.to(unit).magnitude))

