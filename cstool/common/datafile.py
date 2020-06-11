import numpy as np
import h5py
from cstool.common import units

class datafile:
    """Class to store datasets, including units. It is a thin wrapper around
    h5py to store HDF5 files.

    Each "group" consists of one or more datasets. Each dimension in these
    datasets has an associated named "dimension scale", which may optionally be
    stored in the file. Each dataset or dimension scale has associated units.

    We also have "general properties". This class transparently handles string
    or float values. Internally, if the value is a string, it uses the HDF5
    attribute system, while if the value is a number, we use a special
    "properties" dataset, containing data in the form key - value - unit.
    """

    def __init__(self, filename, mode):
        """Open a file. Mode can be any of:
            r  Read-only, file must exist
            r+ Read-write, file must exist
            w  Create file, truncate if exists
            w- or x Create file, fail if exists
            a  Read-write if exists, create otherwise
        """
        self.file = h5py.File(filename, mode)
    def close(self):
        self.file.close()


    def create_group(self, name):
        return datafile_group(self.file.create_group(name))
    def get_group(self, key):
        return datafile_group(self.file[key])


    def set_property(self, key, value, unit=None):
        """Set a global property. Either a string, float or float with unit.

        If writing a float with unit, the "unit" parameter specifies the unit
        string that is written to file. If set to None, this is left to Pint.
        """
        if isinstance(value, (float, int)):
            self.file.attrs[key] = value
        elif isinstance(value, str):
            self.file.attrs[key] = value.encode('ascii')
        else:
            value, unit = _strip_unit(value, unit)

            self.file.attrs[key] = np.array((value, unit.encode('ascii')),
                dtype=np.dtype([('value', float),
                                ('unit', h5py.special_dtype(vlen=bytes))]))

    def list_properties(self):
        return self.file.attrs.keys()

    def get_property(self, key):
        value = self.file.attrs[key]
        if isinstance(value, (float, int)):
            return value
        elif isinstance(value, bytes):
            return value.decode('ascii')
        else:
            return value[0] * units(value[1].decode('ascii'))


    # Utility functions for use in a with: statement
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_value, traceback):
        self.close()




class datafile_group:
    """Helper class for datafile. Represents a group of one or more datasets
    and associated dimension scales"""
    def __init__(self, h5_group):
        self.group = h5_group
        self.scales = {}


    def add_scale(self, name, data, unit=None):
        if name in self.scales:
            raise ValueError('Scale already exists.')

        data, unit = _strip_unit(data, unit)

        h5_dset = self.group.create_dataset(name, data = data)
        h5_dset.attrs['units'] = unit.encode('ascii')
        h5py.h5ds.set_scale(h5_dset.id, name.encode('ascii'))
        self.scales[name] = h5_dset


    def add_dataset(self, name, data, scales, unit=None):
        """Add dataset, with name and data.

        The scales parameter is a tuple of names, each belonging to the
        dimension scale attached to the corresponding dimension of 'data'. May
        be None.
        """
        if scales is None:
            scales = (None,) * len(data.shape)
        else:
            scales = tuple(scales)

        if len(scales) != len(data.shape):
            raise ValueError('Wrong number of dimension scales provided'
                             'when creating dataset.')

        # Check that all scales are correct
        for dim_id, scale_name in enumerate(scales):
            if scale_name is None:
                continue
            if scale_name not in self.scales:
                raise ValueError('Using unknown dimension scale.')
            if len(self.scales[scale_name]) != data.shape[dim_id]:
                raise ValueError('Dimension has different size than its scale.')

        data, unit = _strip_unit(data, unit)

        # Create the dataset
        h5_dset = self.group.create_dataset(name, data = data)
        h5_dset.attrs['units'] = unit.encode('ascii')

        # Attach the dimension scales
        for dim_id, scale_name in enumerate(scales):
            if scale_name is None:
                continue
            h5_scale = self.scales[scale_name]
            h5_dset.dims[dim_id].attach_scale(h5_scale)

    def get_dataset(self, name):
        h5_dset = self.group[name]
        data = np.copy(h5_dset[()]) * units(h5_dset.attrs['units'].decode('ascii'))
        return data




def _strip_unit(value, unit = None):
    if unit is None:
        unit = str(value.units)
        value = value.magnitude
    else:
        value = value.to(unit).magnitude

    return value, unit

