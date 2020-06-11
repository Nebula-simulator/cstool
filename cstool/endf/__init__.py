# Module for loading Evaluated Nuclear Data Files (ENDF/B).
#
# Data files can be downloaded from https://www.nndc.bnl.gov/endf/project.html.
# At the time of writing, the latest version is ENDF/B-VIII.0 from 2018.
#
# For this project, we are only interested in the electro-atomic, photo-atomic,
# and atomic relaxation sublibraries. Only those can be read by the endf reader
# in this folder.
#
# The file format specification can be found at
# https://www.nndc.bnl.gov/csewg/docs/endf-manual.pdf

from .parse_endf import (
	parse_photoat,
	parse_atomic_relax,
	parse_electrons)
from .electronionization import compile_electronionization_icdf
from .photoionization import compile_photoionization_icdf
