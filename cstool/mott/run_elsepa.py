import os
import shutil
import glob
import subprocess
import tempfile
from .elsepa_output import dcs_parser

def run_elscata(settings, elsepa_dir=None):
	"""Run ELSCATA.

	The settings parameter should be an instance of the elscata_settings class.

	If the elsepa_dir parameter is None, this function looks for the elscata
	binary in the PATH.

	This function returns a dict, where the keys are dcs_*.dat filenames, and
	the values are corresponding instances of elsepa_output.dcs_parser.

	This function copies the entire ELSEPA directory to a temporary folder. The
	reason is that the ELSCATA and ELSCATM binaries output temporary files to
	their working directory, which may cause conflicts if multiple instances
	run concurrently. This function is therefore thread-safe.
	"""
	if elsepa_dir is None:
		try:
			elsepa_dir = os.path.dirname(shutil.which('elscata'))
		except:
			raise RuntimeError("Could not find 'elscata' binary")

	if not os.path.exists(elsepa_dir):
		raise RuntimeError("Could not find ELSEPA directory {}".format(elsepa_dir))

	with tempfile.TemporaryDirectory() as temp_dir:
		temp_elsepa_dir = os.path.join(temp_dir, 'elsepa')
		shutil.copytree(elsepa_dir, temp_elsepa_dir)

		subprocess.run(os.path.join(temp_elsepa_dir, 'elscata'),
			input=bytes(settings.generate_string(), 'utf-8'),
			cwd=temp_elsepa_dir,
			stdout=subprocess.DEVNULL)

		results = {}
		for fn in glob.glob(os.path.join(temp_elsepa_dir, 'dcs*.dat')):
			results[os.path.basename(fn)] = dcs_parser(fn)

	return results
