import re
from io import StringIO
from zipfile import ZipFile
from .endf_reader import endf_reader

def parse_photoat(filename, Z):
	with ZipFile(filename) as zf:
		pattern = re.compile(
			'^photoat/photoat-{0:03d}_[A-Za-z]+_000.endf$'.format(Z))
		for fn in zf.namelist():
			if pattern.match(fn):
				return endf_reader(StringIO(zf.read(fn).decode('utf-8')))

def parse_atomic_relax(filename, Z):
	with ZipFile(filename) as zf:
		pattern = re.compile(
			'^atomic_relax/atom-{0:03d}_[A-Za-z]+_000.endf$'.format(Z))
		for fn in zf.namelist():
			if pattern.match(fn):
				return endf_reader(StringIO(zf.read(fn).decode('utf-8')))

def parse_electrons(filename, Z):
	with ZipFile(filename) as zf:
		pattern = re.compile(
			'^electrons/e-{0:03d}_[A-Za-z]+_000.endf$'.format(Z))
		for fn in zf.namelist():
			if pattern.match(fn):
				return endf_reader(StringIO(zf.read(fn).decode('utf-8')))
