__author__ = 'mike'

from RockPy.core import measurement
from RockPy.core.data import RockPyData


class generic_moment(measurement.Measurement):
    def __init__(self, sample_obj,
                 mtype, fpath, ftype,
                 **options):
        super(generic_moment, self).__init__(sample_obj,
                                             mtype, fpath, ftype,
                                             **options)

    def format_cryomag(self):
        data = self.ftype_data.float_data
        header = self.ftype_data.float_header
        data = RockPyData(column_names=header, data=data)
        data.define_alias('m', ('x', 'y', 'z'))
        # data = data.append_columns('mag', data.magnitude('m'))
        self._raw_data = {'data': data.append_columns('mag', data.magnitude('m'))}

    def format_sushibar(self):
        data = RockPyData(column_names=['temp', 'x', 'y', 'z', 'sm'],
                          data=self.ftype_data.out_trm(),
                          # units=['C', 'mT', 'A m^2', 'A m^2', 'A m^2']
                          )
        data.define_alias('m', ('x', 'y', 'z'))
        # data = data.append_columns('mag', data.magnitude('m'))
        self._raw_data = {'data': data.append_columns('mag', data.magnitude('m'))}

    def format_jr6(self):
        data = self.ftype_data.get_data()
        data = RockPyData(column_names=['x', 'y', 'z'],
                          data=data,
                          units=['A m^2', 'A m^2', 'A m^2'])
        data.define_alias('m', ('x', 'y', 'z'))
        self._raw_data = {'data': data.append_columns('mag', data.magnitude('m'))}


class Irm(generic_moment):
    def __init__(self, sample_obj,
                 mtype, fpath, ftype,
                 **options):
        super(Irm, self).__init__(sample_obj,
                                  mtype, fpath, ftype,
                                  **options)


class Arm(generic_moment):
    pass
