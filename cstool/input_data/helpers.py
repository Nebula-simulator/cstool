from cstool.common import units

def load_quantity(yaml_data, name, target_units, optional=False):
	if optional and name not in yaml_data:
		return None

	value = units(yaml_data[name])
	if not value.is_compatible_with(target_units):
		raise RuntimeError('Inconsistent units for {}'.format(name))

	return value

