class reaction:
	"""Class representing a reaction, such as cross section and secondary
	distribution.

	Properties:
	 - MT
	 - cross_section (TAB1 class)
	    Energy (eV) vs cross section (barn)
	 - binding_energy
	    In eV.
	    See EPE (MF=23)
	 - fluorescence_yield
	    In eV/photoionization.
	    See EFL (MF=23)
	 - products
	    List of dicts, secondary photon and electron distributions from MF=26.
	"""

	def __init__(self, MT):
		self.MT = MT
		self.cross_section      = None
		self.binding_energy     = None # EPE (MF=23)
		self.fluorescence_yield = None # EFL (MF=23)
		self.products           = []


class atomic_relaxation:
	"""Class representing atomic relaxation data.

	Properties:
	 - shell_index
	    1-based shell index.
	    Corresponds to a reaction by MT = 533 + shell_index.
	 - shell_legacy_name
	    Legacy name of the shell, e.g. K, L1, L2.
	 - shell_modern_name
	    Modern name of the shell, e.g. 1s1/2, 2s1/2, 2p1/2.
	 - binding_energy
	    Binding energy, in eV
	 - number_electrons
	    Number of electrons in the shell when neutral (floating-point value)
	 - transitions
	    List of tuples: (SUBJ, SUBK, ETR, FTR), where
	    SUBJ, SUBK 1-based indices of the secondary and tertiary subshells (see ENDF manual)
	    ETR Energy of the transition in eV
	    FTR Fractional probability of the transition
	"""

	def __init__(self, subi):
		self.shell_index = subi
		self.shell_legacy_name = ['K',
			'L1', 'L2', 'L3',
			'M1', 'M2', 'M3', 'M4', 'M5',
			'N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7',
			'O1', 'O2', 'O3', 'O4', 'O5', 'O6', 'O7', 'O8', 'O9',
			'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10', 'P11',
			'Q1', 'Q2', 'Q3'][subi-1]
		self.shell_modern_name = ['1s1/2',
			'2s1/2', '2p1/2', '2p3/2',
			'3s1/2', '3p1/2', '3p3/2', '3d3/2', '3d5/2',
			'4s1/2', '4p1/2', '4p3/2', '4d3/2', '4d5/2', '4f5/2', '4f7/2',
			'5s1/2', '5p1/2', '5p3/2', '5d3/2', '5d5/2', '5f5/2', '5f7/2', '5g7/2', '5g9/2',
			'6s1/2', '6p1/2', '6p3/2', '6d3/2', '6d5/2', '6f5/2', '6f7/2', '6g7/2', '6g9/2', '6h9/2', '6h11/2',
			'7s1/2', '7p1/2', '7p3/2'][subi-1]
		self.binding_energy    = None
		self.number_electrons  = None
		self.transitions       = []

