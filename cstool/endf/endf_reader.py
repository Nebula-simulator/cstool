import numpy as np
import re
import warnings
from .endf_data import reaction, atomic_relaxation

class endf_reader:
	"""ENDF-6 format reader.
	See https://www.nndc.bnl.gov/endfdocs/ENDF-102-2018.pdf for the specification.

	Only (MF=1, MT=451), MF=23, MF=26 are implemented.

	Properties:
	 - MAT: Material identifier.
	 - reactions: Dictionary of reaction instances.
	    Key is MT number.
	 - atomic_relaxation: Dictionary of atomic relaxation data.
	    Key is subshell ID.
	"""

	def __init__(self, file_handle):
		"""Starts reading the file in file_handle,
		populating the reactions and atomic_relaxation members."""
		self._file = file_handle
		self.reactions = {}
		self.atomic_relaxation = {}

		# Determine MAT number for this file,
		# while skipping ahead to start of first header
		MF = 0
		while MF == 0:
			position = self._file.tell()
			line = self._file.readline()
			MF = int(line[70:72])
		self.MAT = int(line[66:70])
		self._file.seek(position)

		self._read()


	def _read(self):
		self._read_1_451()

		while True:
			# Next line should be one of
			#   - Start of new section
			#   - End of file (FEND)
			#   - End of material (MEND)
			line = self._peekline()
			if self._is_FEND(line):
				self._FEND(self.MAT)
				continue
			if self._is_MEND(line):
				self._MEND()
				break

			# Next line is start of section.
			# Forward reading to relevant member function
			MF = endf_reader._eint(line[70:72])
			MT = endf_reader._eint(line[72:75])

			if MF == 23:
				self._read_23(MT)
			elif MF == 26:
				self._read_26(MT)
			elif MF == 28 and MT == 533:
				self._read_28_533()
			else:
				# Don't know how to read this file.
				warnings.warn("Not reading file {}".format(MF))
				self._seek_FEND()


	#
	# Seeking through file
	#
	def _peekline(self):
		pos = self._file.tell()
		line = self._file.readline()
		self._file.seek(pos)
		return line

	def _seek_FEND(self):
		while not self._is_FEND(self._file.readline()):
			continue



	#
	# Reading individual files
	#

	# Header
	def _read_1_451(self):
		# Doesn't actually read the data :)
		MMM = [self.MAT, 1, 451]
		self._print_debug(MMM)

		self._HEAD(MMM)
		self._CONT(MMM)
		self._CONT(MMM)
		TEMP, _, LDRV, _, NWD, NXC = self._CONT(MMM)

		for _ in range(NWD):
			self._TEXT(MMM)

		for _ in range(NXC):
			self._CONT(MMM, blankC=True)

		self._SEND(MMM[0], 1)
		self._FEND(MMM[0])

	# Photo-atomic or electro-atomic interaction data
	def _read_23(self, MT):
		MMM = [self.MAT, 23, MT]
		self._print_debug(MMM)

		if MT not in self.reactions:
			self.reactions[MT] = reaction(MT)
		rx = self.reactions[MT]

		self._HEAD(MMM) # ZA, AWR, which are in File 1 anyway

		params, rx.cross_section = self._TAB1(MMM)
		if MT >= 534 and MT <= 599:
			# Subshell ionization
			rx.binding_energy = params[0]
			rx.fluorescence_yield = params[1]

		self._SEND(MMM[0], MMM[1])

	# Secondary distributions
	def _read_26(self, MT):
		MMM = [self.MAT, 26, MT]
		self._print_debug(MMM)

		if MT not in self.reactions:
			self.reactions[MT] = reaction(MT)
		rx = self.reactions[MT]

		ZA, AWR, _, _, NK, _ = self._HEAD(MMM)
		for i in range(NK):
			product = {}
			rx.products.append(product)

			params, _yield = self._TAB1(MMM)
			product['ZAP'] = params[0] # Product identifier
			product['LAW'] = params[3]

			if product['LAW'] == 1:
				params, [NBT, INT] = self._TAB2(MMM)

				NE = params[5]
				product['LANG'] = params[2]
				product['LEP'] = params[3]
				product['E1'] = np.zeros(NE)
				product['ND'] = np.zeros(NE, dtype=int) # Number of discrete energies
				product['Ep'] = [] # Outgoing energy, list of arrays, each with length ND[i]
				product['b'] = []  # Amplitude, list of (ND x a) 2D arrays

				for i in range(NE):
					params, data = self._LIST(MMM)
					data = data.reshape((params[5], params[3]+2))

					product['E1'][i] = params[1]
					product['ND'][i] = params[2]
					product['Ep'].append(data[:,0])
					product['b'].append(data[:,1:])

			elif product['LAW'] == 2:
				params, [NBT, INT] = self._TAB2(MMM)

				NE = params[5]
				product['E1'] = np.zeros(NE)
				product['LANG'] = np.zeros(NE, dtype=int)
				product['Al'] = []

				for i in range(NE):
					params, data = self._LIST(MMM)
					product['E1'][i] = params[1]
					product['LANG'][i] = params[2]
					product['Al'].append(data)

			elif product['LAW'] == 8:
				params, product['ET'] = self._TAB1(MMM)

		self._SEND(MMM[0], MMM[1])

	# Atomic relaxation
	def _read_28_533(self):
		MMM = [self.MAT, 28, 533]
		self._print_debug(MMM)

		ZA, AWR, _, _, NSS, _ = self._HEAD(MMM)
		# NSS: Number of subshells
		for i in range(NSS):
			params, data = self._LIST(MMM)
			SUBI = int(params[0])
			NTR = int(params[5])
			EBI = data[0]
			ELN = data[1]

			ar = atomic_relaxation(SUBI)
			self.atomic_relaxation[SUBI] = ar
			ar.binding_energy = EBI
			ar.number_electrons = ELN

			# NTR: Number of transitions
			for j in range(NTR):
				SUBJ = int(data[6*(j+1) + 0])
				SUBK = int(data[6*(j+1) + 1])
				ETR  = data[6*(j+1) + 2]
				FTR  = data[6*(j+1) + 3]
				ar.transitions.append((SUBJ, SUBK, ETR, FTR))

		self._SEND(MMM[0], MMM[1])

	#
	# Reading individual records
	#
	def _TEXT(self, MMM):
		line = self._file.readline()
		self._verify_MMM(line, MMM)

		return line[:66]

	def _CONT(self, MMM, blankC=False):
		line = self._file.readline()
		self._verify_MMM(line, MMM)

		if blankC:
			C1 = None
			C2 = None
		else:
			C1 = self._efloat(line[:11])
			C2 = self._efloat(line[11:22])
		L1 = self._eint(line[22:33])
		L2 = self._eint(line[33:44])
		N1 = self._eint(line[44:55])
		N2 = self._eint(line[55:66])
		return [C1, C2, L1, L2, N1, N2]

	def _HEAD(self, MMM):
		line = self._file.readline()
		self._verify_MMM(line, MMM)

		ZA  = int(self._efloat(line[:11]))
		AWR = self._efloat(line[11:22])
		L1  = self._eint(line[22:33])
		L2  = self._eint(line[33:44])
		N1  = self._eint(line[44:55])
		N2  = self._eint(line[55:66])
		return [ZA, AWR, L1, L2, N1, N2]

	def _SEND(self, MAT, MF):
		self._verify_MMM(self._file.readline(), [MAT, MF, 0])
	def _FEND(self, MAT):
		self._verify_MMM(self._file.readline(), [MAT, 0, 0])
	def _MEND(self):
		self._verify_MMM(self._file.readline(), [0, 0, 0])
	def _TEND(self):
		self._verify_MMM(self._file.readline(), [-1, 0, 0])
	def _DIR(self):
		raise NotImplementedError()
	def _LIST(self, MMM):
		items = self._CONT(MMM)
		NPL = items[4]

		# Read tabulated data
		B = np.zeros(NPL)
		for ln in range((NPL - 1)//6 + 1):
			line = self._file.readline()
			self._verify_MMM(line, MMM)
			for col in range(min(6, NPL - 6*ln)):
				B[6*ln+col] = self._efloat(line[:11])
				line = line[11:]

		return items, B
	def _TAB1(self, MMM):
		C1, C2, L1, L2, NR, NP = self._CONT(MMM)

		# Read interpolation region data
		NBT = np.zeros(NR, dtype=int)
		INT = np.zeros(NR, dtype=int)
		for ln in range((NR - 1)//3 + 1):
			line = self._file.readline()
			self._verify_MMM(line, MMM)
			for col in range(min(3, NR - 3*ln)):
				NBT[3*ln+col] = self._eint(line[:11])
				INT[3*ln+col] = self._eint(line[11:22])
				line = line[22:]

		# Read tabulated data
		x = np.zeros(NP)
		y = np.zeros(NP)
		for ln in range((NP - 1)//3 + 1):
			line = self._file.readline()
			self._verify_MMM(line, MMM)
			for col in range(min(3, NP - 3*ln)):
				x[3*ln+col] = self._efloat(line[:11])
				y[3*ln+col] = self._efloat(line[11:22])
				line = line[22:]

		return [C1, C2, L1, L2], TAB1(x, y, NBT, INT)
	def _TAB2(self, MMM):
		C1, C2, L1, L2, NR, NZ = self._CONT(MMM)

		# Read interpolation region data
		NBT = np.zeros(NR, dtype=int)
		INT = np.zeros(NR, dtype=int)
		for ln in range((NR - 1)//3 + 1):
			line = self._file.readline()
			self._verify_MMM(line, MMM)
			for col in range(min(3, NR - 3*ln)):
				NBT[3*ln+col] = self._eint(line[:11])
				INT[3*ln+col] = self._eint(line[11:22])
				line = line[22:]

		return [C1, C2, L1, L2, NR, NZ], \
			[NBT, INT]
	def _INTG(self):
		raise NotImplementedError()


	@staticmethod
	def _is_FEND(line):
		MAT = endf_reader._eint(line[66:70])
		MF  = endf_reader._eint(line[70:72])
		MT  = endf_reader._eint(line[72:75])
		return MAT != 0 and MF == 0 and MT == 0
	@staticmethod
	def _is_MEND(line):
		MAT = endf_reader._eint(line[66:70])
		MF  = endf_reader._eint(line[70:72])
		MT  = endf_reader._eint(line[72:75])
		return MAT == 0 and MF == 0 and MT == 0

	@staticmethod
	def _verify_MMM(line, MMM):
		if MMM == None:
			return
		MAT = endf_reader._eint(line[66:70])
		MF  = endf_reader._eint(line[70:72])
		MT  = endf_reader._eint(line[72:75])
		if MAT != MMM[0] or MF != MMM[1] or MT != MMM[2]:
			raise ValueError('Unexpected MAT/MF/MT found in file!')

	@staticmethod
	def _efloat(string):
		# The re.sub deals with "E-less numbers", which are supposed to be the
		# default; but it turns out that, sometimes, a 'D' is printed...
		fixed_string = re.sub(r'([0-9]+)([\+|-])([0-9]+)',
			r'\1E\2\3',
			string.replace('d', 'e').replace('D', 'E'))
		return float(fixed_string)
	@staticmethod
	def _eint(string):
		return int(string)

	@staticmethod
	def _print_debug(MMM):
		# print("Reading MF=%d, MT=%d" % (MMM[1], MMM[2]))
		pass



class TAB1:
	"""Represents a TAB1 record in the ENDF file format"""

	def __init__(self, x, y, NBT, INT):
		self.x = x
		self.y = y
		self.NBT = NBT
		self.INT = INT

