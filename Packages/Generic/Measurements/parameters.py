__author__ = 'volk'
import numpy as np

from RockPy.core.measurement import Measurement
from RockPy.utils.convert import convert2
import RockPy


class Parameter(Measurement):
    pass

class OrientationGeo(Parameter):
    """
    orientation of specimen in respect to geographic coordinate system
    """
    def __init__(self, sample_obj,
                 mtype='orientationgeo', fpath=None, ftype='generic',
                 dec=None, inc=None, sun=None,
                 time=None,
                 series=None,
                 **options):

        super(OrientationGeo, self).__init__(sample_obj=sample_obj,
                                   mtype=mtype, fpath=fpath, ftype=ftype,
                                   series=series,
                                   **options)

        self._raw_data = {'data': RockPy.RockPyData(column_names=['variable', 'dec', 'inc', 'sun', 'time'])}
        self._raw_data['data']['dec'] = dec
        self._raw_data['data']['inc'] = inc
        self._raw_data['data']['sun'] = sun
        self._raw_data['data']['time'] = time

    def format_generic(self):
        pass



class Bedding(Parameter):
    """
    orientation of bedding plane
    """
    def __init__(self, sample_obj,
                 mtype='bedding', fpath=None, ftype='generic',
                 dip_dir=None, dip=None,
                 time=None,
                 series=None,
                 **options):

        super(Bedding, self).__init__(sample_obj=sample_obj,
                                   mtype=mtype, fpath=fpath, ftype=ftype,
                                   series=series,
                                   **options)

        self._raw_data = {'data': RockPy.RockPyData(column_names=['variable', 'dip_dir', 'dip', 'time'])}
        self._raw_data['data']['dip_dir'] = dip_dir
        self._raw_data['data']['dip'] = dip
        self._raw_data['data']['time'] = time

    def format_generic(self):
        pass


class LocationGeo(Parameter):
    """
    geographic location of specimen
    lat, lon are in degrees
    altitude defaults to meter
    """
    def __init__(self, sample_obj,
                 mtype='location', fpath=None, ftype='generic',
                 lat=None, lon=None, alt=None, coordinate_system = 'wgs84',
                 time=None,
                 series=None,
                 **options):

        super(LocationGeo, self).__init__(sample_obj=sample_obj,
                                   mtype=mtype, fpath=fpath, ftype=ftype,
                                   series=series,
                                   **options)



        if coordinate_system != 'wgs84':
            self.logger.warning('support for geographic coordinate systems is not implemented yet.')


        self._raw_data = {'data': RockPy.RockPyData(column_names=['variable', 'lat', 'lon', 'alt', 'time'])}
        self._raw_data['data']['lat'] = lat
        self._raw_data['data']['lon'] = lon
        self._raw_data['data']['alt'] = alt
        self._raw_data['data']['time'] = time

    def format_generic(self):
        pass




class Mass(Parameter):
    """
    simple 1d measurement for mass
    """

    def __init__(self, sample_obj,
                 mtype='mass', fpath=None, ftype='generic',
                 value=1.0, unit='kg',
                 std=None, time=None,
                 series=None,
                 **options):
        super(Mass, self).__init__(sample_obj=sample_obj,
                                   mtype=mtype, fpath=fpath, ftype=ftype,
                                   series=series,
                                   **options)

        mass_conversion = convert2(unit, 'kg', 'mass')

        if not mass_conversion:
            self.logger.warning('mass unit << %s >> most likely not mass-compatible' %unit)
            self.logger.error('CAN NOT create Measurement')
            self.has_data = False
            return

        self._raw_data = {'data': RockPy.RockPyData(column_names=['variable', 'mass', 'time', 'std_dev'])}
        self._raw_data['data'][mtype] = value * mass_conversion
        self._raw_data['data']['time'] = time
        self._raw_data['data']['std_dev'] = std

    def format_generic(self):
        pass


class Length(Parameter):
    """
    simple 1d measurement for Length
    """

    def __init__(self, sample_obj,
                 mtype, fpath=None, ftype='generic',
                 value=1.0, unit='m',
                 direction=None,
                 std=None, time=None,
                 series=None,
                 **options):
        super(Length, self).__init__(sample_obj=sample_obj,
                                     mtype=mtype, fpath=fpath, ftype=ftype,
                                     series=series,
                                     **options)
        self.mtype = mtype
        self.ftype = ftype
        self.direction = direction

        length_conversion = convert2(unit, 'm', 'length')

        if not length_conversion:
            self.logger.warning('unit << %s >> most likely not %s-compatible' % (unit, self.__class__.get_subclass_name()))
            self.logger.error('CAN NOT create Measurement')
            self.has_data = False
            return

        self._raw_data = {'data': RockPy.RockPyData(column_names=[mtype, 'time', 'std_dev'])}
        self._raw_data['data'][mtype] = value * length_conversion
        self._raw_data['data']['time'] = time
        self._raw_data['data']['std_dev'] = std

    def format_generic(self):
        pass


class Diameter(Length):
    """
    simple 1d measurement for Length
    """
    pass


class Height(Length):
    """
    simple 1d measurement for Length
    """
    pass


class Volume(Parameter):
    """
    Combined measurement. It needs a height and a diameter measurement.
    """

    @staticmethod
    def cylinder(height, diameter):
        return np.pi * (diameter / 2) ** 2 * height

    @staticmethod
    def cube(x, y, z):
        return x * y * z

    @staticmethod
    def sphere(diameter):
        return (4 / 3) * np.pi * (diameter / 2) ** 3

    def __init__(self, sample_obj,
                 mtype, fpath=None, ftype='combined',
                 height=None, diameter=None, sample_shape='cylinder',
                 x_len=None, y_len=None, z_len=None,
                 std=None, time=None,
                 series=None,
                 **options):

        super(Volume, self).__init__(sample_obj=sample_obj,
                                     mtype=mtype, fpath=fpath, ftype='combined',
                                     **options)
        self.sample_shape = sample_shape

        if sample_shape == 'cylinder' and height and diameter:
            height_data = height.data['data']['height'].v[0]
            diameter_data = diameter.data['data']['diameter'].v[0]
            volume = self.cylinder(height_data, diameter_data)

        if x_len and y_len and z_len:
            if sample_shape != 'cube':  # check if all three dimensions but wrong/unset sample_shape
                self.logger.warning('sample_shape != cube \t but x_len, y_len, z_len provided -> assuming cube')
                sample_shape = 'cube'
            if sample_shape == 'cube':
                x = x_len.data['data']['length'].v[0]
                y = y_len.data['data']['length'].v[0]
                z = z_len.data['data']['length'].v[0]
            volume = self.cube(x, y, z)

        if diameter and not height:
            if sample_shape == 'sphere':
                diameter_data = diameter.data['data']['diameter'].v[0]
                volume = self.sphere(diameter_data)

        # store in RockPy Data object
        self._raw_data = {'data': RockPy.RockPyData(column_names=['volume', 'time', 'std_dev'])}
        self._raw_data['data'][mtype] = volume
        self._raw_data['data']['time'] = time
        self._raw_data['data']['std_dev'] = std


def test():
    Sample = RockPy.Sample(name='parameter_test',
                           mass=10., mass_unit='kg',
                           height=4.5, diameter=6., length_unit='mm')
    Sample = RockPy.Sample(name='parameter_test',
                           mass=10., mass_unit='kg', sample_shape='sphere',
                           x_len=4.5, y_len=6., z_len=6., length_unit='mm')


if __name__ == '__main__':
    test()
