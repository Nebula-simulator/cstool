from distutils.core import setup

setup(
	name = 'cstool',
	version = '1.0.1',
	description = 'Computes cross sections for the Nebula simulator',
	packages = ['cstool',
		'cstool.common', 'cstool.dielectric_function', 'cstool.endf',
		'cstool.input_data', 'cstool.mott', 'cstool.phonon',
		'cstool.apps'
	],
	package_dir = {
		'cstool': 'cstool',
		'cstool.apps': 'apps'
	},
	package_data = {'cstool': ['data/endf_sources.json',
		'data/endf_data/atomic_relax.zip',
		'data/endf_data/electrons.zip',
		'data/endf_data/photoat.zip']},
	install_requires = ['numpy', 'scipy', 'pyyaml', 'h5py', 'numba', 'pint>=0.11'],
	entry_points = {'console_scripts': ['cstool = cstool.apps.cstool:main']},
)
