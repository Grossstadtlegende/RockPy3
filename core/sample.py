import logging

import RockPy3
import core


class Sample(object):
    snum = 0

    @property
    def logger(self):
        return core.set_get_attr(self, '_logger', value=logging.getLogger('RockPy3.Sample(#%03i)[%s]' % (Sample.snum, self.name)))

    def __init__(self, name,
                 comment='',
                 mass=None, mass_unit='kg', mass_ftype='generic',
                 height=None, diameter=None,
                 x_len=None, y_len=None, z_len=None,  # for cubic samples
                 length_unit='mm', length_ftype='generic',
                 sample_shape='cylinder',
                 coord=None,
                 samplegroup=None, study=None,
                 ):

        """
        Parameters
        ----------
           name: str
              name of the sample.
           mass: float
              mass of the sample. If not kg then please specify the mass_unit. It is stored in kg.
           mass_unit: str
              has to be specified in order to calculate the sample mass properly.
           height: float
              sample height - stored in 'm'
           diameter: float
              sample diameter - stored in 'm'
           length_unit: str
              if not 'm' please specify
           length_machine: str
              if not 'm' please specify
           sample_shape: str
              needed for volume calculation
              cylinder: needs height, diameter
              cube: needs x_len, y_len, z_len
              sphere: diameter
           coord: str
              coordinate system
              can be 'core', 'geo' or 'bed'
           color: color
              used in plots if specified
        """
        self.name = name #unique name, only one per study
        self.comment = comment
        self.idx = Sample.snum

        Sample.snum += 1
        # create a study if none provided
        if not study or not isinstance(study, RockPy3.Study):
            name = None #initialize
            self.logger.warning('A Sample needs a Study to be contained in. RockPy now creates a study and adds the sample')

            if isinstance(study, str):
                name = study

            study = RockPy3.Study(name=name)
            study._add_sample(sobj=self)



        self._study = study
        self._sample_groups = samplegroup


        # coordinate system
        self._coord = coord

        RockPy3.logger.info('CREATING\t new sample << %s >>' % self.name)

        self.raw_measurements = []
        self.measurements = []
        self.results = None

        ''' is sample is a mean sample from samplegroup ect... '''
        # self.is_mean = False  # if a calculated mean_sample #todo needed?
        # self.mean_measurements = []
        # self._mean_results = None

        # # dictionaries
        # self._mdict = self._create_mdict()
        # self._mean_mdict = self._create_mdict()
        # self._rdict = self._create_mdict()

        # if mass is not None:
        #     mass = self.add_measurement(mtype='mass', fpath=None, ftype=mass_ftype,
        #                                 value=float(mass), unit=mass_unit)
        # if diameter is not None:
        #     diameter = self.add_measurement(mtype='diameter', fpath=None, ftype=length_ftype,
        #                                     diameter=float(diameter), length_unit=length_unit)
        # if height is not None:
        #     height = self.add_measurement(mtype='height', fpath=None, ftype=length_ftype,
        #                                   height=float(height), length_unit=length_unit)
        #
        # if x_len:
        #     x_len = self.add_measurement(mtype='length', fpath=None, ftype=length_ftype,
        #                                  value=float(x_len), unit=length_unit, direction='x')
        # if y_len:
        #     y_len = self.add_measurement(mtype='length', fpath=None, ftype=length_ftype,
        #                                  value=float(y_len), unit=length_unit, direction='y')
        # if z_len:
        #     z_len = self.add_measurement(mtype='length', fpath=None, ftype=length_ftype,
        #                                  value=float(z_len), unit=length_unit, direction='z')
        #
        # if height and diameter:
        #     self.add_measurement(mtype='volume', sample_shape=sample_shape, height=height, diameter=diameter)
        # if x_len and y_len and z_len:
        #     self.add_measurement(mtype='volume', sample_shape=sample_shape, x_len=x_len, y_len=y_len, z_len=z_len)