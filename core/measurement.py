__author__ = 'volk'

import logging
import inspect
import itertools
import xml.etree.ElementTree as etree
from copy import deepcopy
import numpy as np
import os
import os.path
import decorator
import RockPy3.core.io
import RockPy3.core.utils

try:
    from pylatex import Document, Section, Subsection, Tabular, Description, Math, TikZ, Axis, Plot, Figure, Package
    from pylatex.utils import italic, escape_latex
    import RockPy3.utils.latex

    pylatex_available = True

except ImportError:
    pylatex_available = False
    logger = logging.getLogger('RockPy')
    logger.warning('Please install module pylatex for latex output.')


class Measurement(object):
    """
    """
    n_created = 0
    log = logging.getLogger('RockPy3.MEASUREMENT')

    possible_plt_props = ['agg_filter', 'alpha', 'animated', 'antialiased', 'axes', 'clip_box', 'clip_on', 'clip_path',
                          'color', 'contains', 'dash_capstyle', 'dash_joinstyle', 'dashes', 'drawstyle', 'figure',
                          'fillstyle', 'gid', 'label', 'linestyle', 'linewidth', 'lod', 'marker', 'markeredgecolor',
                          'markeredgewidth', 'markerfacecolor', 'markerfacecoloralt', 'markersize', 'markevery',
                          'path_effects', 'picker', 'pickradius', 'rasterized', 'sketch_params', 'snap',
                          'solid_capstyle', 'solid_joinstyle', 'transform', 'url', 'visible', 'xdata', 'ydata',
                          'zorder']

    _mcp = None  # mtype calculation parameter cache for all measurements implemented as a dict(measurement:method:parameter)
    _cmcp = None  # mtype calculation parameter cache for all measurements implemented as a dict(measurement: parameter)
    _pcp = None  # possible calculation parameters cache for a single measurement
    _gcp = None  # all possible parameters combined for all measurements and all methods
    _mpcp = None  # all possible parameters for each method of a specific mtype as a dict(method:parameters)

    _scp = None  # standard_calculation parameter
    _rm = None  # result methods

    # xml tags
    MEASUREMENT = 'measurement'
    DATA = 'data'
    RAW_DATA = 'raw_data'
    BASE_MEASUREMENTS = 'base_measurements'
    BID = 'bid'

    # xml attributes
    NAME = 'name'

    @classmethod
    def _mtype(cls):
        return cls.__name__.lower()

    ####################################################################################################################
    ''' calculation parameter methods '''

    @classmethod
    def collected_mtype_calculation_parameter(cls):
        """
        searches through all implemented calculation methods and collects the parameters used in the method.

        Returns
        -------
            dictionary
             key == mtype
             value == a sorted list of unique parameters
        """
        if not cls._cmcp:
            cls._cmcp = {}
            for mname, measurement in RockPy3.implemented_measurements.items():
                cls._cmcp.setdefault(
                    mname,
                    sorted(
                        list(set([i for j in measurement.mtype_possible_calculation_parameter().values() for i in j]))))
        return cls._cmcp

    @classmethod
    def mtype_calculation_parameter(cls):
        """
        searches through all implemented calculation methods and collects the parameters used in the method.

        Returns
        -------
            a sorted list of unique parameters
        """
        if not cls._mcp:
            cls._mcp = {}
            for mname, measurement in RockPy3.implemented_measurements.items():
                cls._mcp.setdefault(mname, measurement.mtype_possible_calculation_parameter())
        return cls._mcp

    @classmethod
    def method_calculation_parameter_list(cls):
        """
        searches through all implemented calculation methods and collects the parameters used in the method.

        Returns
        -------
            a dictionary with:
            key: calculation_method name
            value: list of unique parameters
        """
        if not cls._pcp:
            cls.pcp = {}
            for mname, measurement in RockPy3.implemented_measurements.items():
                for method, parameters in measurement.mtype_possible_calculation_parameter().items():
                    cls.pcp.setdefault(method, parameters)
        return cls.pcp

    @classmethod
    def global_calculation_parameter(cls):
        """
        searches through all implemented calculation methods and collects the parameters used in the method.

        Returns
        -------
            a sorted list of unique parameters
        """
        if not cls._gcp:
            gpcp = []
            for mname, measurement in RockPy3.implemented_measurements.items():
                for method, parameters in measurement.collected_mtype_calculation_parameter().items():
                    gpcp.extend(parameters)
        return sorted(list(set(gpcp)))

    @classmethod
    def mtype_possible_calculation_parameter(cls):
        """
        All possible calculation methods of a measurement. It first creates a dictionary of all calculate_methods and
        their signature.
        Then it looks through all result_ methods and checks if any is not added yet and needs a recipe
        Note
        ----
            This adds a pseudo_method to the dictionary for each result_method with a recipe.
            Therefor this dict can not be used to check for methods!
        """

        if not cls._mpcp:
            # get all parameters from all measurements
            cls._mpcp = {
                i: set(arg for arg, value in inspect.signature(getattr(cls, 'calculate_' + i)).parameters.items()
                       if not arg in ['self', 'non_method_parameters'])
                for i in cls.calculate_methods()}

            # methods with recipe need 'recipe' to be added to the calculation_parameters
            for res in cls.result_methods():
                if res not in cls._mpcp:
                    methods = cls.get_calculate_methods(res)
                    if cls.result_category(res) == 'indirect':
                        cls._mpcp.setdefault(res, set())
                        cls._mpcp[res].update(set(param for recipe in methods for param in cls._mpcp[recipe]))
                    if 'recipe' in cls.result_category(res):
                        cls._mpcp.setdefault(res, set())
                        cls._mpcp[res].update(set(param for recipe in methods for param in cls._mpcp[recipe]))
                        cls._mpcp[res].update(['recipe'])
        return cls._mpcp

    def calculation_methods(self):
        # dynamically generating the calculation and standard parameters for each calculation method.
        # This just sets the values to non, the values have to be specified in the class itself
        return {i: getattr(self, i) for i in dir(self)
                if i.startswith('calculate_')
                if not i.endswith('generic')
                if not i.endswith('result')
                }

    @classmethod
    def standards_result(cls):
        """
        generates a dictionary of results:calculate_methods
        """
        scp = {r: {'recipe': False, 'indirect': False, 'secondary': False, 'dependent': False} for r in
               cls.result_methods()}
        for res in scp:
            res_method = getattr(cls, 'result_' + res)
            result_sig = {k.name: k.default for k in inspect.signature(res_method).parameters.values() if
                          not k.name == 'self' if not k.name == 'non_method_parameters'}

            if 'recipe' in result_sig:
                scp[res]['recipe'] = True
            if 'calculation_method' in result_sig:
                scp[res]['indirect'] = True
            if 'secondary' in result_sig:
                scp[res]['secondary'] = True
            if 'dependencies' in result_sig:
                scp[res]['dependent'] = True

            scp[res].setdefault('signature', result_sig)
            result_sig.setdefault('recipe', 'default')
        return scp

    @classmethod
    def standards_calculate(cls):
        scp = {}
        for c, method in cls.calculate_methods().items():
            method_sig = {k.name: k.default for k in inspect.signature(method).parameters.values() if
                          not k.name == 'self' if not k.name == 'non_method_parameters'}
            scp[c] = method_sig
        return scp

    """
    ####################################################################################################################
    RESULT/CALCULATE METHOD RELATED
    """

    @classmethod
    def has_recipe(cls, res):
        """
        checks if result has parameter recipe in signature. if it does this means that the result_method has
        different
        """
        if 'recipe' in inspect.signature(getattr(cls, 'result_' + res)).parameters:
            return True
        else:
            return False

    @classmethod
    def has_calculation_method(cls, res):
        """
        checks if result has parameter calculation_method in argspec. if it does this means that the result_method has
        different

        Returns
        -------
            bool
        """
        if 'calculation_method' in inspect.signature(getattr(cls, 'result_' + res)).parameters:
            return True
        else:
            return False

    @classmethod
    def has_secondary(cls, res):
        """
        checks if result has parameter calculation_method in argspec. if it does this means that the result_method has
        different

        Returns
        -------
            bool
        """
        if 'secondary' in inspect.signature(getattr(cls, 'result_' + res)).parameters:
            return True
        else:
            return False

    @classmethod
    def result_category(cls, res):
        """
        Takes a result_method name as argument and checks which category it belongs to.

            1. direct methods:
                :code:`result_a` calls :code:`calculate_a`

            2. direct recipe methods:
                :code:`result_b` calls :code:`calculate_b_RECIPE1` or :code:`calculate_b_RECIPE2`

            3. indirect methods:
                :code:`result_d` calls :code:`calculate_c` via calculation_method parameter

            4. indirect recipe methods:
                :code:`result_f` calls :code:`calculate_e_RECIPE1` or :code:`calculate_e_RECIPE2` via
                calculation_method parameter

            extra: dependent calculation_methods
                :code:`result_g` need a secondary measurement to be called

        Parameter
        ---------

            result: str
                the name of the result_method without the :code:`'result_'`

        Returns
        -------

            category: str
                'direct' if case 1
                'direct_recipe' if case 2
                'indirect' if case 3
                'indirect_recipe' if case 4
            if secondary neasurement is needed, '_dependent' will be added:
                -> e.g. 'direct_dependent'
        """
        res = res.replace('result_', '')
        out = None
        if cls.has_recipe(res) and cls.has_calculation_method(res):
            out = 'indirect_recipe'
        elif cls.has_calculation_method(res) and not cls.has_recipe(res):
            out = 'indirect'
        elif cls.has_recipe(res) and not cls.has_calculation_method(res):
            out = 'direct_recipe'
        elif res in cls.calculate_methods():
            out = 'direct'
        if cls.has_secondary(res):
            out += '_dependent'

        return out

    @classmethod
    def get_calculate_methods(cls, res):
        """
        takes a result_method name as input and returns list of matching calculate_methods
        """

        if cls.result_category(res) == 'direct':
            return [res]
        elif cls.result_category(res) == 'direct_recipe':
            return [i for i in cls.calculate_methods()
                    if i.split('_')[-1].isupper
                    if ''.join(i.split('_')[:-1]) == res]

        calculation_method = inspect.signature(getattr(cls, 'result_' + res)).parameters['calculation_method'].default
        if cls.result_category(res) == 'indirect':
            return [calculation_method]
        elif cls.result_category(res) == 'indirect_recipe':
            return [i for i in cls.calculate_methods()
                    if i.split('_')[-1].isupper
                    if ''.join(i.split('_')[:-1]) == calculation_method]

    @classmethod
    def implemented_ftypes(cls):
        # setting implemented machines
        # looking for all subclasses of RockPy3.io.base.Machine
        # generating a dictionary of implemented machines : {implemented out_* method : machine_class}
        implemented_ftypes = {cl.__name__.lower(): cl for cl in RockPy3.core.io.ftype.__subclasses__()}
        return implemented_ftypes

    @classmethod
    def subclasses(cls):
        """
        Returns a list of the implemented_visuals names
        """
        return [i.__name__.lower() for i in cls.inheritors()]

    @classmethod
    def inheritors(cls):
        """
        Method that gets all children and childrens-children ... from a class

        Returns
        -------
           list
        """
        subclasses = set()
        work = [cls]
        while work:
            parent = work.pop()
            for child in parent.__subclasses__():
                if child not in subclasses:
                    subclasses.add(child)
                    work.append(child)
        return subclasses

    @classmethod
    def measurement_formatters(cls):
        # measurement formatters are important!
        # if they are not inside the measurement class, the measurement has not been implemented for this machine.
        # the following machine formatters:
        # 1. looks through all implemented measurements
        # 2. for each measurement stores the machine and the applicable readin class in a dictionary
        measurement_formatters = {
            i.replace('format_', '').lower(): getattr(cls, i) for i in dir(cls) if i.startswith('format_')
            }
        return measurement_formatters

    ####################################################################################################################
    # builtin methods
    @classmethod
    def result_methods(cls):
        """
        Searches through all :code:`result_*` methods and creates a dictionary with:

            result_name : result_method

        where result_name is the name without the result_ prefix
        """
        if not cls._rm:
            cls._rm = {i[7:]: getattr(cls, i) for i in dir(cls) if i.startswith('result_')
                       if not i.endswith('generic')
                       if not i.endswith('methods')
                       if not i.endswith('category')
                       }
        return cls._rm

    @classmethod
    def calculate_methods(cls):
        # dynamically generating the calculation and standard parameters for each calculation method.
        # This just sets the values to non, the values have to be specified in the class itself
        calculate_methods = {i.replace('calculate_', ''): getattr(cls, i) for i in dir(cls)
                             if i.startswith('calculate_')
                             if not i.endswith('generic')
                             if not i.endswith('result')
                             if not i.endswith('methods')
                             }
        return calculate_methods

    @classmethod
    def correct_methods(cls):
        """
        Dynamically searches through the class and finds all correction methods

        """
        methods = {i.replace('correct_', ''): getattr(cls, i) for i in dir(cls)
                   if i.startswith('correct_')
                   }
        # for name, method in methods.items():
        setattr(RockPy3.Measurement.correct_methods, '__doc__', 'test')
        return methods

    ####################################################################################################################

    @classmethod
    def get_subclass_name(cls):
        return cls.__name__

    @property
    def study(self):
        """
        find study to which the measurement belongs

        :return:
        """
        if self.sobj:
            return self.sobj.study
        return None

    ####################################################################################################################
    # plotting / legend properties
    @property
    def plt_props(self):
        return self._plt_props

    def set_plt_prop(self, prop, value):
        """
        sets the plt_props for the measurement.

        raises
        ------
            KeyError if the plt_prop not in the matplotlib.lines.Line2D
        """
        if value is None:
            return
        if prop not in Measurement.possible_plt_props:
            raise KeyError
        self._plt_props.setdefault(prop, None)
        self.log.debug('SETTING {} from {} -> {}'.format(prop, self._plt_props[prop], value))
        self._plt_props[prop] = value

    ####################################################################################################################

    @property
    def coord(self):  # todo remove?
        """
        find coordinate system of sample
        """
        print("returning" + str(self.sobj.coord))
        return self.sobj.coord

    """
    ####################################################################################################################
    measurement creation through function
    """

    @classmethod
    def empty(cls, sobj,
              fpath=None, ftype='generic',  # file path and file type
              idx=None,
              # for special import of pure data (needs to be formatted as specified in data of measurement class)
              series=None,
              ):

        mdata = {'data': RockPy3.Data(column_names=[])}

        return cls(sobj=sobj, fpath='', ftype='empty', mdata=mdata, series=series, idx=idx)

    @classmethod
    def from_mdata(cls):
        pass

    @classmethod
    def from_file(cls, sobj,
                  fpath=None, ftype='generic',  # file path and file type
                  idx=None, sample_name=None,
                  # for special import of pure data (needs to be formatted as specified in data of measurement class)
                  series=None,
                  **options
                  ):

        if ftype in cls.implemented_ftypes():
            ftype_data = cls.implemented_ftypes()[ftype](fpath, sobj.name)
        else:
            cls.log.error('CANNOT IMPORT ')

        if ftype in cls.measurement_formatters():
            cls.log.debug('ftype_formatter << %s >> implemented' % ftype)
            mdata = cls.measurement_formatters()[ftype](ftype_data, sobj_name=sobj.name)
            if not mdata:
                return
        else:
            cls.log.error('UNKNOWN ftype: << %s >>' % ftype)
            cls.log.error(
                'most likely cause is the \"format_%s\" method is missing in the measurement << %s >>' % (
                    ftype, cls.__name__))
            return

        return cls(sobj=sobj, fpath=fpath, ftype=ftype, mdata=mdata, series=series, idx=idx, **options)

    @classmethod
    def from_simulation(cls, sobj=None, idx=None, **parameter):
        """
        pseudo abstract method that should be overridden in subclasses to return a simulated measurement
        based on given parameters
        """
        return None

    @classmethod
    def from_measurements_create_mean(cls, sobj, mlist,
                                      interpolate=False, recalc_mag=False,
                                      substfunc='mean', ignore_series=False,
                                      color=None, marker=None, linestyle=None):
        """
        Creates a new measurement from a list of measurements
        :param sobj:
        :param mlist:
        :param interpolate:
        :param recalc_mag:
        :param substfunc:
        :param ignore_series:
        :return:
        """
        # convert to single measurement
        mlist = RockPy3.core.utils.to_list(mlist)
        if any(m.mtype != cls.__name__.lower() for m in mlist):
            cls.log.error('Some measurements have wrong mtype. They will be ignored')
            mlist = [m for m in mlist if m.mtype == cls.__name__.lower()]

        # use first measurement as base
        dtypes = RockPy3.core.utils.get_common_dtypes_from_list(mlist=mlist)

        if len(mlist) == 1:
            mdata = deepcopy(mlist[0].data)

        else:
            mdata = {}

            for dtype in dtypes:  # cycle through all dtypes e.g. 'down_field', 'up_field' for hysteresis
                mdata.setdefault(dtype)
                dtype_list = [m.data[dtype] for m in mlist if m.data[dtype]]  # get all data for dtype in one list
                if dtype_list:
                    if interpolate:
                        varlist = cls._get_variable_list(rpdata_list=dtype_list)
                        if len(varlist) > 1:
                            dtype_list = [m.interpolate(varlist) for m in dtype_list]

                if len(dtype_list) > 1:  # for single measurements
                    mdata[dtype] = RockPy3.condense(dtype_list, substfunc=substfunc)
                    mdata[dtype] = mdata[dtype].sort('variable')

                if recalc_mag:
                    mdata[dtype].define_alias('m', ('x', 'y', 'z'))
                    mdata[dtype]['mag'].v = mdata[dtype].magnitude('m')

        ################################################################################################################
        # initial state
        initial = None

        if all(m.has_initial_state for m in mlist):
            init_list = [m.initial_state for m in mlist if m.has_initial_state]
            initial = cls.from_measurements_create_mean(sobj=sobj, mlist=init_list, interpolate=interpolate,
                                                        recalc_mag=recalc_mag, substfunc=substfunc)
        # add series if needed
        series = None
        if not ignore_series:
            slist = (set(s.data for s in m.series) for m in mlist)
            series = set()
            for s in slist:
                if not series:
                    series = s
                else:
                    series = series & s
            if series:
                series = list(series)

        return cls(sobj=sobj, ftype='from_measurements_create_mean', mdata=mdata,
                   initial_state=initial, series=series, ismean=True, base_measurements=mlist,
                   color=color, marker=marker, linestyle=linestyle)

    @classmethod
    def from_measurement(cls):
        """
        creates a measurement from a different type

        e.g. pARM spectra -> ARM acquisition
        """
        pass

    @classmethod
    def from_result(cls, **parameter):
        """
        pseudo abstract method that should be overridden in subclasses to return a measurement created from results
        """
        return None

    def set_initial_state(self,
                          mtype=None, fpath=None, ftype=None,  # standard
                          mobj=None, series=None,
                          ):
        """
        creates a new measurement (ISM) as initial state of base measurement (BSM).
        It dynamically calls the measurement _init_ function and assigns the created measurement to the
        self.initial_state value. It also sets a flag for the ISM to check if a measurement is a MIS.

        if a measurement object is passed, the initial state will be created from _measurements.
        Parameters
        ----------
           mtype: str
              measurement type
           mfile: str
              measurement data file
           machine: str
              measurement machine
            mobj: RockPy3.MEasurement object
           options:
        """

        with RockPy3.ignored(AttributeError):
            mtype = mtype.lower()
            ftype = ftype.lower()

        self.log.info('CREATING << %s >> initial state measurement << %s >> data' % (mtype, self.mtype))

        # can only be created if the measurement is actually implemented
        if all([mtype, ftype, fpath]) or fpath or mobj:
            self.initial_state = self.sobj.add_measurement(
                mtype=mtype, ftype=ftype, fpath=fpath, series=series, mobj=mobj)
            self.initial_state.is_initial_state = True
            return self.initial_state
        else:
            self.log.error('UNABLE to find measurement << %s >>' % mtype)

    def __init__(self,
                 sobj,
                 fpath=None, ftype=None,
                 mdata=None,
                 series=None,
                 idx=None,
                 initial_state=None,
                 ismean=False, base_measurements=None,
                 color=None, marker=None, linestyle=None, mid=None,
                 **options
                 ):
        """
        Constructor of the measurement class.

        Several checks have to be done:
            1. is the measurement implemented:
                this is checked by looking if the measurement is in the RockPy3.implemented_measurements
            2. if mdata is given, we can directly create the measurement #todo from_mdata?
            3. if the file format (ftype) is implemented #todo from_file
                The ftype has to be given. This is how RockPy can format data from different sources into the same
                format, so it can be analyzed the same way.

        Parameters
        ----------
            sobj: RockPy3.Sample
                the sample object the measurement belongs to. The constructor is usually called from the
                Sample.add_measurement method
            mtype: str
                MANDATORY: measurement type to be imported. Implemented measurements can be seen when calling
                >>> print Measurement.measurement_formatters()
            fpath: str
                path to the file including filename
            ftype: str
                file type. e.g. vsm
            mdata: dict
                when mdata is set, this will be directly used as measurement data without formatting from file
            initial_state:
                RockPy3.Measurement obj

        """
        self._cm = None

        if mid is None:
            self.id = id(self)
        else:
            self.id = int(mid)

        self.sobj = sobj
        self._plt_props = {'label': ''}

        self.log = logging.getLogger('RockPy3.MEASURMENT.' + self.get_subclass_name())

        if mdata is None:
            mdata = {}

        # the data that is used for calculations and corrections
        self._data = mdata

        # _raw_data is a backup deepcopy of _data it will be used to reset the _data if reset_data() is called
        self._raw_data = deepcopy(mdata)

        # coordinate system that is currently used in data; _raw_data is always assumed to be in core coordiantes
        self._actual_data_coord = 'core'

        self.is_mean = ismean  # flag for mean measurements
        self.base_measurements = base_measurements  # list with all measurements used to generate the mean

        self.ftype = ftype
        self.fpath = fpath

        ''' initial state '''
        self.is_initial_state = False
        self.initial_state = initial_state

        ''' calibration, correction and holder'''
        self.calibration = None
        self.holder = None
        self._correction = []

        self.results = None
        self.__initialize()

        # normalization
        self.is_normalized = False  # normalized flag for visuals, so its not normalized twice
        self.norm = None  # the actual parameters

        # add series if provided
        ''' series '''
        self._series = []
        if series:
            self.add_series(series=series)
        else:
            if self.study:
                self.study._series.setdefault('none', []).append(self)

        if not idx:
            idx = len(self.sobj.measurements)

        self.idx = idx
        self.__class__.n_created += 1

        #### automatically set the plt_props for the measurement according to the
        self.set_plt_prop(prop='color', value=RockPy3.colorscheme[self.idx])
        self.set_plt_prop(prop='marker', value=RockPy3.marker[self.sobj.idx])
        self.set_plt_prop(prop='linestyle', value='-')

        if any([color, marker, linestyle]) or marker == '':
            self.set_plt_prop('color', color)
            self.set_plt_prop('marker', marker)
            self.set_plt_prop('linestyle', linestyle)

    def set_recipe(self, result, recipe):
        """
        changes the recipe for a result to a new value. Changes the standard parameter dictionary to the new values

        Parameter
        ---------
            result:
            recipe:

        Note:
            if the result is indirect (e.g. hf_sus is calculated through ms) the result will overwrite the method for both

        """
        direct_result = result
        indirect_result = None

        if not direct_result in self.result_recipe:
            self.log.error('Measurement << %s >> has no result << %s >>' % (self.mtype, direct_result))
            return

        if not self.standards_result()[direct_result]['recipe']:
            self.log.error('Result << %s >> has no recipe << %s >>' % (direct_result, recipe))
            return

        if self.standards_result()[direct_result]['indirect']:
            direct_result = self.standards_result()[direct_result]['signature']['calculation_method']
            indirect_result = result

        if not recipe in self.get_recipes(direct_result):
            self.log.error('Recipe not found, these are implemented: %s' % self.get_recipes(direct_result))
            return

        self.result_recipe[direct_result] = recipe.upper()

        if indirect_result:
            self.result_recipe[indirect_result] = self.result_recipe[direct_result]

        self.log.warning('Calculation parameter changed from:')
        self.log.warning('{}'.format(self.calculation_parameter[direct_result]))
        self.calculation_parameter[direct_result] = self.standards_calculate()[self.get_calculate_method(direct_result)]

        # change the method that is called
        if recipe == 'default':
            recipe = ''

        self.methods[direct_result] = getattr(self, '_'.join(['calculate', self.get_calculate_method(result)]))
        self.log.warning('{}'.format(self.calculation_parameter[direct_result]))

        if indirect_result:
            self.log.warning(
                'The result << {} >> is an in direct result and is calculated through the method << {} >>'.format(
                    indirect_result, direct_result))
            self.log.warning('The recipe for all connected results has been changed to << {} >>'.format(recipe))
            # change the method that is called for the indirect result
            self.calculation_parameter[indirect_result] = self.calculation_parameter[direct_result]
            self.methods[indirect_result] = self.methods[direct_result]

    def get_recipes(self, result):
        """
        returns all result recipes for a given result
        :param result:
        :return:
        """
        recipes = ['default'] + [r.split('_')[-1].lower() for r in self.standards_calculate() if
                                 result in r and r.split('_')[-1].isupper()]
        return set(recipes)

    def get_calculate_method(self, result):
        """
        Takes a result, looks up the defined method and returns the calculation_method_name

        Parameter
        ---------
            result: str
                The result name

        Returns
        -------
            returns the calculate_method name.
            For direct methods without a recipe this will be: resultname
            For direct methods with recipe this will be: resultname_RECIPE
            For indirect methods this will be: calculationmethod
            For indirect methods with a recipe this will be: calculationmethod

        """
        # for indirect methods the result name must be the calculation name
        if self.standards_result()[result]['indirect']:
            result = self.standards_result()[result]['signature']['calculation_method']

        # get the recipe
        recipe = self.result_recipe[result]
        # add the suffix for non-default mewthods
        if recipe != 'DEFAULT':
            method_name = '_'.join([result, recipe.upper()])
        # default recipes do not need a suffix
        else:
            method_name = result
        return method_name

    def get_dependent_results(self, result):
        """
        gets all result names that are equally independent and the independent result they are based on

        Example
        -------
            hf_sus(hysteresis) is always dependent on the calculation of ms.
                -> returns [ms, hf_sus]
            mrs_ms(hysteresis) is always dependent on ms & mrs and dependencies on each of these
                -> returns [ms, mrs, mrs_ms, hf_sus]
        :param result:
        :return:
        """

        # if not self.standards_result()[result]['indirect'] and not self.standards_result()[result]['dependent']:
        #     return [result]

        if 'calculation_method' in self.standards_result()[result]['signature']:
            cm = self.standards_result()[result]['signature']['calculation_method']
        else:
            cm = result

        indirect_methods = [res for res in self.standards_result() if self.standards_result()[res]['indirect']]
        dependent_on_cm = [res for res in indirect_methods if
                           self.standards_result()[res]['signature']['calculation_method'] == cm]

        if self.standards_result()[result]['dependent']:
            for res in self.standards_result()[result]['signature']['dependencies']:
                dependent_on_cm.extend(self.get_dependent_results(res))

        return sorted(set([cm] + dependent_on_cm))

    @property
    def mtype(self):
        return self._mtype()

    @property
    def base_ids(self):
        """
        returns a list of ids for all base measurements
        """
        return [m.id for m in self.base_measurements]

    def __repr__(self):
        if self.is_mean:
            add = 'mean_'
        else:
            add = ''
        return '<<RockPy3.{}.{}{}{} at {}>>'.format(self.sobj.name, add, self.mtype,
                                                    self.stype_sval_tuples,
                                                    hex(id(self)))

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return self.id == other.id

    def __add__(self, other):
        first = deepcopy(self)

        for dtype in first.data:
            vars1 = set(first.data[dtype]['variable'].v)
            vars2 = set(other.data[dtype]['variable'].v)
            diff = vars1 ^ vars2
            if diff:
                vars = vars1 | vars2
                first.data[dtype] = first.data[dtype].interpolate(new_variables=vars)
                other.data[dtype] = other.data[dtype].interpolate(new_variables=vars)
            first.data[dtype] = first.data[dtype].eliminate_duplicate_variable_rows() + other.data[
                dtype].eliminate_duplicate_variable_rows()
            first.data[dtype] = first.data[dtype].sort()
        return self.sobj.add_measurement(mtype=first.mtype, mdata=first.data)

    def __sub__(self, other):
        first = deepcopy(self)

        for dtype in first.data:
            vars1 = set(first.data[dtype]['variable'].v)
            vars2 = set(other.data[dtype]['variable'].v)
            diff = vars1 ^ vars2
            if diff:
                vars = vars1 | vars2
                first.data[dtype] = first.data[dtype].interpolate(new_variables=vars)
                other.data[dtype] = other.data[dtype].interpolate(new_variables=vars)
            first.data[dtype] = first.data[dtype].eliminate_duplicate_variable_rows() - other.data[
                dtype].eliminate_duplicate_variable_rows()
            first.data[dtype] = first.data[dtype].sort()
        return self.sobj.add_measurement(mtype=first.mtype, mdata=first.data)

    def __getstate__(self):
        """
        returned dict will be pickled
        :return:
        """
        pickle_me = {k: v for k, v in self.__dict__.items() if k in
                     (
                         'id',
                         'ftype', 'fpath',
                         # plotting related
                         '_plt_props',
                         'calculation_parameters',
                         # data related
                         '_raw_data', '_data',
                         'initial_state', 'is_initial_state',
                         'is_mean', 'base_measurements',
                         # sample related
                         'sobj',
                         '_series',
                         'calibration', 'holder', 'correction',
                     )
                     }
        return pickle_me

    def __setstate__(self, d):
        """
        d is unpickled data
           d:
        :return:
        """
        self.__dict__.update(d)
        self.__initialize()

    def __initialize(self):
        """
        Initialize function is called inside the __init__ function, it is also called when the object is reconstructed
        with pickle.

        :return:
        """

        # calculate the standards for results, calculation and the appropriate methods
        self.result_recipe = {result: self.standards_result()[result]['signature']['recipe'].upper() for result in
                              self.standards_result()}

        self.calculation_parameter = {
            result: self.standards_calculate()[self.get_calculate_method(result)] for result in self.result_recipe
            if not self.standards_result()[result]['secondary']}

        self.methods = {result: getattr(self, 'calculate_' + self.get_calculate_method(result)) for result in
                        self.result_recipe if not self.standards_result()[result]['secondary']}

        self._info_dict = self.__create_info_dict()

    @property
    def stype_sval_tuples(self):
        if self.get_series():
            return [(s.stype, s.value) for s in self.series]
        else:
            return []

    @property
    def m_idx(self):
        return self.sobj.measurements.index(self)  # todo change could be problematic if changing the sobj

    @property
    def fname(self):
        """
        Returns only filename from self.file

        Returns
        -------
           str: filename from full path
        """
        return os.path.basename(self.fpath)

    @property
    def has_initial_state(self):
        """
        checks if there is an initial state
        """
        return True if self.initial_state else False

    ####################################################################################################################
    ''' INFO DICTIONARY '''

    @property
    def info_dict(self):
        if not hasattr(self, '_info_dict'):
            self._info_dict = self.__create_info_dict()
        if not all(i in self._info_dict['series'] for i in self.series):
            self._populate_info_dict()
        return self._info_dict

    def __create_info_dict(self):
        """
        creates all info dictionaries

        Returns
        -------
           dict
              Dictionary with a permutation of ,type, stype and sval.
        """
        d = ['stype', 'sval']
        keys = ['_'.join(i) for n in range(3) for i in itertools.permutations(d, n) if not len(i) == 0]
        out = {i: {} for i in keys}
        out.update({'series': []})
        return out

    def _populate_info_dict(self):
        """
        Removes old info_dict and then Re-calculates the info_dictionary for the measurement
        """
        self._info_dict = self.__create_info_dict()
        map(self.add_s2_info_dict, self.series)

    def add_s2_info_dict(self, series):
        """
        adds a measurement to the info dictionary.

        Parameters
        ----------
           series: RockPy3.Series
              Series to be added to the info_dictionary
        """

        if not series in self._info_dict['series']:
            self._info_dict['stype'].setdefault(series.stype, []).append(self)
            self._info_dict['sval'].setdefault(series.value, []).append(self)

            self._info_dict['sval_stype'].setdefault(series.value, {})
            self._info_dict['sval_stype'][series.value].setdefault(series.stype, []).append(self)
            self._info_dict['stype_sval'].setdefault(series.stype, {})
            self._info_dict['stype_sval'][series.stype].setdefault(series.value, []).append(self)

            self._info_dict['series'].append(series)

    def remove_s_from_info_dict(self, series):
        """
        removes a measurement to the info dictionary.

        Parameters
        ----------
           series: RockPy3.Series
              Series to be removed from the info_dictionary
        """
        if series in self._info_dict['series']:
            self.series.remove(series)
            self._info_dict['stype'].setdefault(series.stype, []).remove(self)
            self._info_dict['sval'].setdefault(series.value, []).remove(self)
            self._info_dict['sval_stype'][series.value].setdefault(series.stype, []).remove(self)
            self._info_dict['stype_sval'][series.stype].setdefault(series.value, []).remove(self)
            self._info_dict['series'].remove(series)
        self._info_dict_cleanup()

    def remove_all_series(self):
        """
        removes all series from the measurement
        """
        for s in self.series:
            self.series.remove(s)
            self.remove_s_from_info_dict(s)

    def _info_dict_cleanup(self):
        """
        recursively removes all empty lists from dictionary
        :param empties_list:
        :return:
        """

        mdict = getattr(self, 'info_dict')

        # cycle through level 0
        for k0, v0 in sorted(mdict.items()):
            if isinstance(v0, dict):
                # cycle through level 1
                for k1, v1 in sorted(v0.items()):
                    if isinstance(v1, dict):
                        for k2, v2 in sorted(v1.items()):
                            if not v2:
                                v1.pop(k2)
                            if not v1:
                                v0.pop(k1)
                            else:
                                if not v1:
                                    v0.pop(k1)
                    else:
                        if not v1:
                            v0.pop(k1)

    @property
    def stypes(self):
        """
        list of all stypes
        """
        stypes = set(s.stype for s in self.series)
        return list(stypes)

    @property
    def svals(self):
        """
        list of all stypes
        """
        return self.info_dict['sval'].keys()

    @property
    def data(self):
        if self._data == {}:
            self._data = deepcopy(self._raw_data)

        # transform vectorial x,y,z data to new coordinate system when needed
        # self.transform_data_coord(final_coord=self.coord)
        # TODO: fails ->
        # print self.coord

        return self._data

    def transform_data_coord(self, final_coord):
        """
        transform columns x,y,z in data object to final_coord coordinate system
        """
        if self._actual_data_coord != final_coord:
            # check if we have x,y,z columns in data
            # TODO look for orientation measurements
            # extract coreaz, coredip, bedaz and beddip
            # transform x, y, z columns in _data['data']
            # final_xyz = RockPy3.utils.general.coord_transform(initial_xyz, self._actual_data_coord, final_coord)
            self.log.warning('data needs to be transformed from %s to %s coordinates. NOT IMPLEMENTED YET!' % (
                self._actual_data_coord, final_coord))
            # return self._data
        else:
            self.log.debug('data is already in %s coordinates.' % final_coord)
            # return self._data

    @property
    def correction(self):
        """
        If attribute _correction does not exist, it is created and returned. After that the attribute always exists.
        This is a way so any _attribute does not have to be created in __init__
        A string of the used calculation method has to be appended for any corrections.
        This way we always know what has been corrected and in what order it has been corrected
        """
        return self.set_get_attr('_correction', value=list())

    def reset_data(self):
        """
        Resets all data back to the original state. Deepcopies _raw_data back to _data and resets correction
        """
        self._data = deepcopy(self._raw_data)
        # create _correction if not exists
        self.set_get_attr('_correction', value=list())
        self._correction = []

    ####################################################################################################################
    ''' DATA RELATED '''

    ''' Calculation and parameters '''

    def calculate_result(self, result, **parameter):
        """
        Helper function to dynamically call a result. Used in Visualize

        Parameters
        ----------
           result:
           parameter:
        """

        if not self.has_result(result):
            self.log.warning('%s does not have result << %s >>' % self.mtype, result)
            return
        else:
            # todo figure out why log wrong when called from Visualize
            self.log = logging.getLogger('RockPy3.MEASURMENT.' + self.mtype + '[%s]' % self.sobj.name)
            self.log.info('CALCULATING << %s >>' % result)
            out = getattr(self, 'result_' + result)(**parameter)
        return out

    def calc_all(self, recalc=False, **parameter):
        # get possible calculation parameters and put them in a dictionary
        calculation_parameter, kwargs = RockPy3.core.utils.kwargs_to_calculation_parameter(rpobj=self, **parameter)
        for result_method in self.result_methods():
            calc_param = calculation_parameter.get(self.mtype, {})
            calc_param = calc_param.get(result_method, {})
            getattr(self, 'result_' + result_method)(recalc=recalc, **calc_param)
        if kwargs:
            self.log.warning('--------------------------------------------------')
            self.log.warning('| %46s |' % 'SOME PARAMETERS COULD NOT BE USED')
            self.log.warning('--------------------------------------------------')
            for i, v in kwargs.items():
                self.log.warning('| %22s: %22s |' % (i, v))
            self.log.warning('--------------------------------------------------')
        if calculation_parameter:
            self.log.info('--------------------------------------------------')
            self.log.info('| %46s |' % 'these parameters were used')
            self.log.info('--------------------------------------------------')
            for i, v in calculation_parameter.items():
                self.log.info('| %46s |' % i)
                for method, parameter in v.items():
                    self.log.info('| %22s: %22s |' % (method, parameter))
            self.log.info('--------------------------------------------------')

        return self.results

    def delete_dtype_var_val(self, dtype, var, val):
        """
        deletes step with var = var and val = val

           dtype: the step type to be deleted e.g. th
           var: the variable e.g. temperature
           val: the value of that step e.g. 500

        example: measurement.delete_step(step='th', var='temp', val=500) will delete the th step where the temperature is 500
        """
        idx = self._get_idx_dtype_var_val(dtype=dtype, var=var, val=val)
        self.data[dtype] = self.data[dtype].filter_idx(idx, invert=True)
        return self

    def check_parameters(self, caller, parameter):
        """
        Checks if previous calculation used the same parameters, if yes returns the previous calculation
        if no calculates with new parameters

        Parameters
        ----------
           caller: str
               name of calling function ('result_generic' should be given as 'generic')
           parameter:
        Returns
        -------
           bool
              returns true is parameters are not the same
        """
        if self.calculation_parameter[caller]:
            # parameter for new calculation
            a = []
            for key in self.calculation_parameter[caller]:
                if key in parameter:
                    a.append(parameter[key])
                else:
                    a.append(self.calculation_parameter[caller][key])
                    # a = [parameter[i] for i in self.calculation_parameter[caller]]
            # get parameter values used for calculation
            b = [self.calculation_parameter[caller][i] for i in self.calculation_parameter[caller]]
            if a != b:
                return True
            else:
                return False
        else:
            return True

    def has_result(self, result):
        """
        Checks if the measurement contains a certain result

        Parameters
        ----------
           result: str
              the result that should be found e.g. result='ms' would give True for 'hys' and 'backfield'
        Returns
        -------
           out: bool
              True if it has result, False if not
        """
        if result in self.result_methods():
            return True
        else:
            return False

    @classmethod
    def _get_variable_list(self, rpdata_list, var='variable'):
        """
        takes a list of rpdata objects. it checks for all steps, the size of the step and min and max values of the
        variable. It then generates a list of new variables from the max(min) -> min(max) with the mean step size


        """
        min_vars = []
        max_vars = []
        stepsizes = []
        for rp in rpdata_list:
            stepsizes.append(np.diff(rp[var].v))
            min_vars.append(min(rp[var].v))
            max_vars.append(max(rp[var].v))
        idx, steps = max(enumerate(stepsizes), key=lambda tup: len(tup[1]))
        new_variables = np.arange(max(min_vars), min(max_vars), np.mean(steps))
        return sorted(list(set(new_variables)))

    """
    ####################################################################################################################

    SERIES related
    """

    @property
    def series(self):
        if self._series:
            return self._series
        else:
            series = RockPy3.Series(stype='none', value=np.nan, unit='')
            return [series]

    def get_series(self, stype=None, sval=None, series=None):
        """
        searches for given stypes and svals in self.series and returns them

        Parameters
        ----------
            series: list(tuple)
                list of tuples to avoid problems wit separate series and same sval
            stypes: list, str
                stype or stypes to be looked up
            svals: float
                sval or svals to be looked up

        Returns
        -------
            list
                list of series that fit the parameters
                if no parameters - > all series
                empty if none fit

        Note
        ----
            m = measurement with [<RockPy3.series> pressure, 0.00, [GPa], <RockPy3.series> temperature, 0.00, [C]]
            m.get_series('pressure', 0) -> [<RockPy3.series> pressure, 0.00, [GPa]]
            m.get_series(0) -> [<RockPy3.series> pressure, 0.00, [GPa], <RockPy3.series> temperature, 0.00, [C]]
        """
        out = self.series
        if stype is not None:
            stype = RockPy3.core.utils.to_list(stype)
            stype = [stype.lower() for stype in stype]
            out = [i for i in out if i.stype in stype]
        if sval is not None:
            sval = RockPy3.core.utils.to_list(sval)
            out = [i for i in out if i.value in sval]

        if series:  # todo series

            out = [i for i in out if i.data in sval]

        return out

    def get_sval(self, stype):
        """
        Searches for stype and returns sval
        """
        s = self.get_series(stype=stype)
        return s[0].value if s else None

    def add_series(self, stype=None, sval=0, unit='', series_obj=None, series=None):
        """
        adds a series to measurement.series, then adds is to the data and results datastructure

        Parameters
        ----------
           stype: str
              series type to be added
           sval: float or int
              series value to be added
           unit: str
              unit to be added. can be None #todo change so it uses Pint
            series_obj: RockPy3.series
                if a previously created object needs to be passed
            series: list(tuples)
                default: None
                Series object gets created for a list of specified series

        Returns
        -------
           [RockPy3.Series] list of RockPy series objects

        Note
        ----
            If the measurement previously had no series, the (none, 0 , none) standard series will be removed first
        """
        # if a series object is provided other wise create series object
        if not any(i for i in [stype, sval, unit, series_obj, series]):
            return
        elif series_obj:
            series = series_obj

        elif series:
            if type(series) == tuple:
                aux = []
                aux.append(series)
                slist = aux
            else:
                slist = series
            series = []
            for stup in slist:
                series.append(RockPy3.Series.from_tuple(series=stup))
        else:
            series = RockPy3.Series(stype=stype, value=sval, unit=unit)

        series = RockPy3.core.utils.to_list(series)

        for s in series:
            self.study._series.setdefault(s.data, []).append(self)  # todo see if better
            with RockPy3.ignored(ValueError):
                self.study._series['none'].remove(self)  # todo see if better

        # remove default series from sobj.mdict if non series exists previously
        if not self._series:
            self.sobj._remove_series_from_mdict(mobj=self, series=self.series[0],
                                                mdict_type='mdict')  # remove default series

        # turn into list to add multiples
        series = RockPy3.core.utils.to_list(series)
        for sobj in series:
            if not any(sobj == s for s in self._series):
                self._series.append(sobj)
                self._add_sval_to_data(sobj)
                self._add_sval_to_results(sobj)

            # add the measurement to the mdict of the sobj
            if not self.is_mean:
                self.sobj._add_series2_mdict(series=sobj, mobj=self)
        return series

    def _add_sval_to_data(self, sobj):
        """
        Adds stype as a column and adds svals to data. Only if stype != none.

        Parameter
        ---------
           sobj: series instance
        """
        if sobj.stype != 'none':
            for dtype in self._raw_data:
                if self._raw_data[dtype]:
                    data = np.ones(len(self.data[dtype]['variable'].v)) * sobj.value
                    if not 'stype ' + sobj.stype in self.data[dtype].column_names:
                        self.data[dtype] = self.data[dtype].append_columns(column_names='stype ' + sobj.stype,
                                                                           data=data)  # , unit=sobj.unit) #todo add units

    def _add_sval_to_results(self, sobj):
        """
        Adds the stype as a column and the value as value to the results. Only if stype != none.

        Parameter
        ---------
           sobj: series instance
        """
        if sobj.stype != 'none':
            # data = np.ones(len(self.results['variable'].v)) * sobj.value
            if not self.results:
                self.results = RockPy3.Data(column_names='stype ' + sobj.stype,
                                            data=[sobj.value])
            if not 'stype ' + sobj.stype in self.results.column_names:
                self.results = self.results.append_columns(column_names='stype ' + sobj.stype,
                                                           data=[sobj.value])  # , unit=sobj.unit) #todo add units

    def __sort_list_set(self, values):
        """
        returns a sorted list of non duplicate values
           values:
        :return:
        """
        return sorted(list(set(values)))

    def _get_idx_dtype_var_val(self, dtype, var, val):
        """
        returns the index of the closest value with the variable(var) and the step(step) to the value(val)

        option: inverse:
           returns all indices except this one

        """
        out = [np.argmin(abs(self.data[dtype][var].v - val))]
        return out

    def equal_series(self, other):
        if all(i in other.series for i in self.series):
            return True
        if not self.series and not other.series:
            return True
        else:
            return False

    """
    Normalize functions
    +++++++++++++++++++
    """

    def normalize(self,
                  reference='data', ref_dtype='mag', norm_dtypes='all', vval=None,
                  norm_method='max', norm_factor=None, result=None,
                  normalize_variable=False, dont_normalize=None,
                  norm_initial_state=True, **options):
        """
        normalizes all available data to reference value, using norm_method

        Parameter
        ---------
            reference: str
                reference state, to which to normalize to e.g. 'NRM'
                also possible to normalize to mass
            ref_dtype: str
                component of the reference, if applicable. standard - 'mag'
            norm_dtypes: list
                default: 'all'
                dtypes to be normalized, if dtype = 'all' all columns will be normalized
            vval: float
                variable value, if reference == value then it will search for the point closest to the vval
            norm_method: str
                how the norm_factor is generated, could be min
            normalize_variable: bool
                if True, variable is also normalized
                default: False
            result: str
                default: None
                normalizes the values in norm_dtypes to the result value.
                e.g. normalize the moment to ms (hysteresis measuremetns)
            dont_normalize: list
                list of dtypes that will not be normalized
                default: None
            norm_initial_state: bool
                if true, initial state values are normalized in the same manner as normal data
                default: True
        """

        # dont normalize parameter measurements
        if isinstance(self, RockPy3.Parameter):
            return

        # separate the calc from non calc parameters
        calculation_parameter, options = RockPy3.core.utils.separate_calculation_parameter_from_kwargs(rpobj=self,
                                                                                                       **options)

        # getting normalization factor
        if not norm_factor:  # if norm_factor specified
            norm_factor = self._get_norm_factor(reference=reference, rtype=ref_dtype,
                                                vval=vval,
                                                norm_method=norm_method,
                                                result=result,
                                                **calculation_parameter)

        norm_dtypes = RockPy3.core.utils.to_list(norm_dtypes)  # make sure its a list/tuple

        for dtype, dtype_data in self.data.items():  # cycling through all dtypes in data
            if dtype_data:
                if 'all' in norm_dtypes:  # if all, all non stype data will be normalized
                    norm_dtypes = [i for i in dtype_data.column_names if not 'stype' in i]

                ### DO not normalize:
                # variable
                if not normalize_variable:
                    variable = dtype_data.column_names[dtype_data.column_dict['variable'][0]]
                    norm_dtypes = [i for i in norm_dtypes if not i == variable]

                if dont_normalize:
                    dont_normalize = RockPy3.core.utils._to_tuple(dont_normalize)
                    norm_dtypes = [i for i in norm_dtypes if not i in dont_normalize]

                for ntype in norm_dtypes:  # else use norm_dtypes specified
                    try:
                        dtype_data[ntype] = dtype_data[ntype].v / norm_factor
                    except KeyError:
                        self.log.warning(
                            'CAN\'T normalize << %s, %s >> to %s' % (self.sobj.name, self.mtype, ntype))

                if 'mag' in dtype_data.column_names:
                    try:
                        self.data[dtype]['mag'] = self.data[dtype].magnitude(('x', 'y', 'z'))
                    except KeyError:
                        self.log.debug('no (x,y,z) data found keeping << mag >>')

        self.log.debug('NORMALIZING << %s >> with << %.2e >>' % (', '.join(norm_dtypes), norm_factor))

        if self.initial_state and norm_initial_state:
            for dtype, dtype_rpd in self.initial_state.data.items():
                self.initial_state.data[dtype] = dtype_rpd / norm_factor
                if 'mag' in self.initial_state.data[dtype].column_names:
                    self.initial_state.data[dtype]['mag'] = self.initial_state.data[dtype].magnitude(('x', 'y', 'z'))
        return self

    def _get_norm_factor(self, reference, rtype, vval, norm_method, result, **calculation_parameter):
        """
        Calculates the normalization factor from the data according to specified input

        Parameter
        ---------
           reference: str
              the type of data to be referenced. e.g. 'NRM' -> norm_factor will be calculated from self.data['NRM']
              if not given, will return 1
           rtype:
           vval:
           norm_method:

        Returns
        -------
           normalization factor: float
        """
        norm_factor = 1  # inititalize
        # print('measurement:', locals())
        if reference and not result:
            if reference == 'nrm' and reference not in self.data and 'data' in self.data:
                reference = 'data'

            if reference in self.data:
                norm_factor = self._norm_method(norm_method, vval, rtype, self.data[reference])

            if reference in ['is', 'initial', 'initial_state']:
                if self.initial_state:
                    norm_factor = self._norm_method(norm_method, vval, rtype, self.initial_state.data['data'])
                if self.is_initial_state:
                    norm_factor = self._norm_method(norm_method, vval, rtype, self.data['data'])

            if reference == 'mass':
                m = self.get_mtype_prior_to(mtype='mass')
                if not m:
                    raise KeyError('CANT find mass measurement')
                return m.data['data']['mass'].v[0]

            if isinstance(reference, float) or isinstance(reference, int):
                norm_factor = float(reference)

        elif result:
            norm_factor = getattr(self, 'result_' + result)(**calculation_parameter)[0]
        else:
            self.log.warning('NO reference specified, do not know what to normalize to.')
        return norm_factor

    def _norm_method(self, norm_method, vval, rtype, data):
        methods = {'max': max,
                   'min': min,
                   # 'val': self.get_val_from_data,
                   }
        if not vval:
            if not norm_method in methods:
                raise NotImplemented('NORMALIZATION METHOD << %s >>' % norm_method)
                return
            else:
                return methods[norm_method](data[rtype].v)

        if vval:
            idx = np.argmin(abs(data['variable'].v - vval))
            out = data.filter_idx([idx])[rtype].v[0]
            return out

    def get_mtype_prior_to(self, mtype):
        """
        search for last mtype prior to self

        Parameters
        ----------
           mtype: str
              the type of measurement that is supposed to be returned

        Returns
        -------
           RockPy3.Measurement
        """

        measurements = self.sobj.get_measurement(mtype=mtype)

        if measurements:
            out = [i for i in measurements if i.m_idx <= self.m_idx]
            return out[-1]

        else:
            return None

    def _add_stype_to_results(self):
        """
        adds a column with stype stype.name to the results for each stype in measurement.series
        :return:
        """
        if self._series:
            for t in self.series:
                if t.stype:
                    if t.stype not in self.results.column_names:
                        self.results.append_columns(column_names='stype ' + t.stype,
                                                    data=t.value,
                                                    # unit = t.unit      # todo add units
                                                    )

    def get_series_labels(self, stypes=True, add_stype=True, add_sval=True, add_unit=True):
        """
        takes a list of stypes or stypes = True and returns a string with stype_sval_sunit; for each stype

        Parameters
        ----------
            stypes: list / bool
                default: True
                if True all series will be returned
                if a list of strings is provided the ones where a matching stype can be found are appended
            add_stype: bool
                default: True
                if True: stype is first part of label
                if False: stype not in label
            add_unit: bool
                default: True
                if True: unit is last part of label
                if False: unit not in label
        """
        out = []
        if stypes is True or stypes is None:
            stypes = list(self.stypes)

        stypes = RockPy3.core.utils.to_list(stypes)

        for stype in stypes:
            stype = self.get_series(stype=stype)[0]
            if stype:
                aux = []
                if add_stype:
                    aux.append(stype.stype)
                if add_sval:
                    aux.append(str(np.round(stype.value, 2)))
                if add_unit:
                    aux.append(stype.unit)
                stype_label = ' '.join(aux)
                if not stype_label in out:
                    out.append(stype_label)
            else:
                self.log.warning('CANT find series << %s >>' % stype)
                self.log.warning('\tonly one of these are possible:\t%s' % self.stypes)
        return '; '.join(out)

    def has_mtype_stype_sval(self, mtype=None, stype=None, sval=None):
        """
        checks if the measurement type is mtype, if it has a series with stype and the sval and if it is a mean measurement

        Parameters
        ----------

            mtype: str
                checks if it is the correct mtype
            stype: str
                checks if it has series with the correct stype
            sval: float
                checks if it has series with the correct sval
        Note
        ----
            only takes a single str/float for each argument
        """
        if mtype and self.mtype != mtype:
            return False
        if stype is not None or sval is not None:
            if not self.get_series(stype=stype, sval=sval):
                return False
        return True

    """
    CORRECTIONS
    +++++++++++
    """

    def correct_dtype(self, dtype='th', var='variable', val='last', initial_state=True):
        """
        corrects the remaining moment from the last th_step

           dtype:
           var:
           val:
           initial_state: also corrects the initial state if one exists
        """

        try:
            calc_data = self.data[dtype]
        except KeyError:
            self.log.error('REFERENCE << %s >> can not be found ' % (dtype))

        if val == 'last':
            val = calc_data[var].v[-1]
        if val == 'first':
            val = calc_data[var].v[0]

        idx = self._get_idx_dtype_var_val(step=dtype, var=var, val=val)

        correction = self.data[dtype].filter_idx(idx)  # correction step

        for dtype in self.data:
            # calculate correction
            self._data[dtype]['m'] = self._data[dtype]['m'].v - correction['m'].v
            # recalc mag for safety
            self.data[dtype]['mag'] = self.data[dtype].magnitude(('x', 'y', 'z'))
        self.reset__data()

        if self.initial_state and initial_state:
            for dtype in self.initial_state.data:
                self.initial_state.data[dtype]['m'] = self.initial_state.data[dtype]['m'].v - correction['m'].v
                self.initial_state.data[dtype]['mag'] = self.initial_state.data[dtype].magnitude(('x', 'y', 'z'))
        return self

    def set_calibration_measurement(self,
                                    fpath=None,  # file path
                                    mdata=None,
                                    mobj=None,  # for special import of a measurement instance
                                    ):
        """
        creates a new measurement that can be used as a calibration for self. The measurement has to be of the same
        mtype and has to have the same ftype

        Parameters
        ----------
        fpath: str
            the full path and filename where the file is located on the hard disk
        mdata: RockPyData
        mobj: RockPy3.Measurement
        """

        cal = self.sobj.add_measurement(mtype=self.mtype, ftype=self.ftype, fpath=fpath,
                                        mobj=mobj, mdata=mdata,
                                        create_only=True)
        self.calibration = cal

    '''' PLOTTING '''''

    def label_add_sname(self):
        """
        adds the corresponding sample_name to the measurement label
        """
        if isinstance(self.sobj, RockPy3.MeanSample):
            mean = 'mean '
        else:
            mean = ''
        self.plt_props['label'] = ' '.join([self.plt_props['label'], mean, self.sobj.name])

    def label_add_stype(self, stypes=None, add_stype=True, add_sval=True, add_unit=True):
        """
        adds the corresponding sample_name to the measurement label
        """
        text = self.get_series_labels(stypes=stypes, add_stype=add_stype, add_sval=add_sval, add_unit=add_unit)
        self.plt_props['label'] = ' '.join([self.plt_props['label'], text])

    def label_add_text(self, text):
        self.plt_props['label'] = ' '.join([self.plt_props['label'], text])

    @property
    def plottable(self):
        """
        returns a list of all possible Visuals for this measurement
        :return:
        """
        out = {}
        for name, visual in RockPy3.Visualize.base.Visual.implemented_visuals().items():
            if visual._required == [self.mtype]:
                out.update({visual.__name__.lower(): visual})
        return out

    def show_plots(self):
        for visual in self.plottable:
            self.plottable[visual](self, show=True)

    def set_get_attr(self, attr, value=None):
        """
        checks if attribute exists, if not, creates attribute with value None
           attr:
        :return:
        """
        if not hasattr(self, attr):
            setattr(self, attr, value)
        return getattr(self, attr)

    def series_to_color(self, stype, reverse=False):
        # get all possible svals in the hierarchy
        svals = sorted(self.sobj.mdict['stype_sval'][stype].keys())

        # create colormap from svals
        color_map = RockPy3.core.utils.create_heat_color_map(value_list=svals, reverse=reverse)

        # get the index and set the color
        sval = self.get_series(stype=stype)[0].value
        sval_index = svals.index(sval)
        self.color = color_map[sval_index]

    def plt_all(self, **plt_props):
        fig = RockPy3.Figure()
        calculation_parameter, non_calculation_parameter = core.utils.separate_calculation_parameter_from_kwargs(
            self, **plt_props)
        for visual in self.plottable:
            fig.add_visual(visual=visual, visual_input=self, **plt_props)
        fig.show(**non_calculation_parameter)

    ####################################################################################################################
    ''' REPORT '''

    # todo report sheet

    ####################################################################################################################
    ''' XML io'''

    @property
    def etree(self):
        """
        Returns the content of the measurement as an xml.etree.ElementTree object which can be used to construct xml
        representation

        Returns
        -------
             etree: xml.etree.ElementTree
        """

        measurement_node = etree.Element(type(self).MEASUREMENT, attrib={'id': str(self.id), 'mtype': str(self.mtype),
                                                                         'is_mean': str(self.is_mean)})

        # store _data dictionary
        for name, data in self._data.items():
            de = etree.SubElement(measurement_node, type(self).DATA, attrib={type(self).NAME: name})
            if data is not None:
                de.append(data.etree)

        # store _raw_data dictionary
        for name, data in self._raw_data.items():
            de = etree.SubElement(measurement_node, type(self).RAW_DATA, attrib={type(self).NAME: name})
            if data is not None:
                det = data.etree
                de.append(det)

        if self.is_mean:
            # store ids of base measurements
            base_measurements_node = etree.SubElement(measurement_node, type(self).BASE_MEASUREMENTS, attrib={})
            for id in self.base_ids:
                etree.SubElement(base_measurements_node, type(self).BID, attrib={}).text = str(id)

        return measurement_node

    @classmethod
    def from_etree(cls, et_element, sobj):
        """

        :param et_element: ElementTree.Element containing the xml data
        :param sobj: sample object to which this measurement will belong
        :return:
        """
        if et_element.tag != cls.MEASUREMENT:
            log.error('XML Import: Need {} node to construct object.'.format(cls.MEASUREMENT))
            return None

        # readin data
        mdata = {}
        for data in et_element.findall(cls.DATA):
            mdata[data.attrib[cls.NAME]] = None

        # readin rawdata
        raw_data = {}
        for data in et_element.findall(cls.RAW_DATA):
            raw_data[data.attrib[cls.NAME]] = None

        is_mean = (et_element.attrib['is_mean'].upper == 'TRUE')
        if is_mean:
            # TODO: readin base measurements
            pass

        m = cls(sobj=sobj, id=et_element.attrib['id'], mtype=et_element.attrib['mtype'], ismean=is_mean)

        return m


###################################################################
# decorators
###################################################################

@decorator.decorator
def result_new(func, *args, **kwargs):
    result_name = '_'.join(func.__name__.split('_')[1:])

    # get the signature of the function: meaning all parameters taht have been passed and the variable names
    signature = {k:args[i] for i,k in enumerate(inspect.signature(func).parameters) if i< len(args)}
    recalc = signature.pop('recalc', False)
    self = signature.pop('self')
    cmethod = signature.pop('calculation_method', None)

    kwargs.update(signature)
    # look through the kwargs if there are possible calculation_parameter
    changed_params = set(self.calculation_parameter[result_name]) & set(kwargs)
    unused_params = set(kwargs) - set(self.calculation_parameter[result_name])

    if not hasattr(self, 'results'):
        setattr(self, 'results', None)
    # check if result needs to be called again
    # if no results have been calculated set recalc to true
    if self.results is None:
        self.log.debug('No results calculated, yet. Calculating')
        recalc = True

    # if this result has not been calculated
    elif result_name not in self.results.column_names:
        recalc = True

    # if any parameter changed after a result has been calculated, it has to be calculated again
    elif result_name in self.results.column_names:
        if any(self.calculation_parameter[result_name][param] != kwargs[param] for param in changed_params):
            self.log.debug('Calculation parameters have changed. Calculating')
            recalc = True

    if unused_params:
        self.log.warning('Found calculation_parameters that have not been used:')
        self.log.warning('{}'.format(unused_params))

    for param in changed_params:
        self.calculation_parameter[result_name][param] = kwargs[param]

    # call calculation method, all args per passed
    if recalc:
        self.log.debug('calling {}'.format(self.methods[result_name].__name__))
        self.methods[result_name](result_name=result_name, **self.calculation_parameter[result_name])
    return self.results[result_name].v[0], self.results[result_name].e[0]


@decorator.decorator
def result(func, *args, **kwargs):
    """
    Result decorator. This decorator calls the right :code:`calculation_method` from the pool of possible
    methods/recipes of the measurement class.

    it first builds the correct name for the :code:`calculation_method` then calls the method and directly returns the
    results without actually calling the result_method

    Parameters
    ----------
        recipe: str
            the default calculation recipe (e.g linear or nonlinear) for the given result_method
            some results have more than one calculation method implemented these can be used with this
        calculation_method: str
            default: None
            if this is not none, but a string, the calculation of the result will be done using a different
            result method than the specified.
            Some Results (e.g. Thellier.result_sigma) is calculated in a different method (in this case it
            is calculated by Thellier.result_slope) and therefore 'slope' has to be passed
    """
    result_name = '_'.join(func.__name__.split('_')[1:])

    # compute the parameter dictionary from the functions argspecs
    parameters = RockPy3.core.utils.get_full_argspec(func=func, args=args, kwargs=kwargs)
    # get the measurement object, equivalent to self in class = args[0]
    self = parameters.pop('self')

    parameters.setdefault('result', result_name)
    # calculation_parameters, p = RockPy3.core.utils.separate_calculation_parameter_from_kwargs(rpobj=self, **parameters)

    # calculation method has to be popped from dictionary, otherwise it is stored in calculation parameters
    # when the calculation_method is called
    if 'calculation_method' in parameters:
        calculation_method = parameters.pop('calculation_method')
    else:
        calculation_method = None

    # recipes are the different possible calculation paths
    recipe = parameters.get('recipe', '').upper()  # not pop because needed for calculation_parameter check

    # add called from to parameters so output will give correct result otherwise:
    # results that are calculated using calculation_method with show calculation method in output instead of
    # result

    # get a list of all possible calculation methods
    recipes = [i.split('_')[-1] for i in self.calculation_methods().keys()
               if len(i.split('_')) >= 3
               if calculation_method in i.split('_') or result_name in '_'.join(i.split('_')[1:-1])
               if i.split('_')[-1].isupper()]

    if recipe and recipe not in recipes:
        raise NotImplementedError(
            'RECIPE << {} >> not implemented for {} chose from {}'.format(recipe, result_name, recipes))

    # building the name of the calculation method to be called
    # if a calculation method is given, a different method has to be called.
    # e.g. result_eigenvalue1 needs to call calculate_tensor -> calculation_method needs to be 'tensor
    if calculation_method:
        calc_name = 'calculate_' + calculation_method
    # otherwise we can call the calculation_method with the same name
    else:
        calc_name = 'calculate_' + result_name

    # we need to add any possible recipes
    if recipe:
        calc_name += '_' + str(recipe)

    if calc_name not in self.calculation_methods():
        raise NotImplementedError('CALCULATION METHOD << %s >> not implemented' % calc_name)

    # call calculation method, all args per passed
    self.calculation_methods()[calc_name](**parameters)
    return self.results[result_name].v[0], self.results[result_name].e[0]


@decorator.decorator
def calculate_new(func, *args, **kwargs):
    """
    Calculation decorator, checks if a column exists in the self.results.
    :param func:
    :param args:
    :param kwargs:
    :return:
    """
    self = args[0]
    result_name = kwargs.pop('result_name')

    # get all results that are dependent of or a dependency of result_name
    dependent_results = self.get_dependent_results(result_name)

    # if no Results have been calculated - create a new RPdata object
    if self.results is None:
        self.results = RockPy3.Data(column_names=dependent_results, data=[np.nan for i in dependent_results])

    # create new columns for all dependent results #todo change if RPdata['not_yet_a_column']=5 gives no error
    for result in dependent_results:
        if not result in self.results.column_names:
            self.results = self.results.append_columns(column_names=result, data=[[[np.nan, np.nan]]])
    return func(self, **self.calculation_parameter[result_name])


@decorator.decorator
def calculate(func, *args, **kwargs):
    """
    Compares the parameters of calculated result with parameter used for new calculation.

    There are three possible, where the result has to be calculated

    #. the result has not been calculated before:
        this can be seen in the self.calculation_parameters dictionary because it will not have a key for the
        calculation_method_name
    #. the result is forced to be recalculated:
        parameters also has an entry for recalc, which is :code:`True`
    #. at least one parameter differs from the calculation_parameters

    If any of the above is true, the method then updates the calcualtion_parameters with the new parameters and
    returns True

    Parameters
    ----------
        calculation_method_name: str
            the name of the result that should be calculated. Used for looking up the previous calculation_parameters
            in the :code:`self.calcluation_parameter` dictionary.
        method_name: str
            the method that is supposed to be used to compute the result. this comes directly from
            :code:`self.calculate_result_name_MethodName`
        parameters: dict
            the calculation parameters used for the new calculation, these will be returned if the calculation should
            be done
        recalc: bool
            True if the calculation is forced

    Returns
    -------
        False
            if paramters are equal -> no new calculation necessary
        True
            if parameters are not equal -> new calculation with new parameters
    """
    parameters = {arg: args[i] for i, arg in enumerate(decorator.getfullargspec(func).args)}
    calculation_method_name, recipe_name = get_result_recipe_name(func_name=func.__name__)

    if 'called_from' in kwargs:
        called_from = kwargs.pop('called_from')
    else:
        called_from = calculation_method_name

    # only add recipe to name if recipe is given, decluttering calculation_parameter dict
    if recipe_name:
        parameters.update(dict(recipe=recipe_name))

    parameters.pop('check',
                   None)  # pop check from parameters in case check is passed for calculations, None ignores error

    # get recalc value from locals dictionary passed from method
    recalc = kwargs.pop('recalc', False)

    # get the measurement object, equivalent to self in class
    self = parameters.pop('self')

    self.log.debug('CHECKING if << %s >> needs to be called' % func.__name__)

    do_calc = False  # flag, that shows if result should be returned or the result
    # recalc forces a new calculation
    if recalc:
        self.calculation_parameter.setdefault(calculation_method_name, parameters)
        self.log.debug('FORCED recalculation of << %s >>' % called_from)
        self.log.debug('\t with: %s' % parameters)
        do_calc = True
    # if result has not been calculated so far,
    # self.calculation_parameter[calculation_method_name] is an empty dictionary
    elif calculation_method_name not in self.calculation_parameter:
        self.calculation_parameter.setdefault(calculation_method_name, parameters)
        self.log.debug('RESULT << %s >> not calculated yet' % called_from)
        self.log.debug('\tcalculation with: %s' % parameters)
        do_calc = True
    elif 'recipe' in parameters \
            and not parameters['recipe'] == self.calculation_parameter[calculation_method_name]['recipe']:
        self.log.debug('RESULT << %s >> calculated with different recipe' % calculation_method_name)
        do_calc = True
    # # if only one of the parameters is different result has to be recalculated
    elif any(vold != parameters[key] for key, vold in self.calculation_parameter[calculation_method_name].items()):
        self.log.debug('RESULT << %s >> parameters have changed' % called_from)
        self.log.debug('\told parameters: %s' % self.calculation_parameter[calculation_method_name])
        self.log.debug('\tnew parameters: %s' % parameters)
        self.calculation_parameter[calculation_method_name].update(parameters)
        do_calc = True
    if do_calc:
        self.log.info('CALCULATING << %s >>' % called_from)
        return func(*args, **kwargs)
    else:
        self.log.debug('RESULT << %s >> already calculated' % called_from)
        self.log.debug('\twith parameters: %s' % self.calculation_parameter[calculation_method_name])


@decorator.decorator
def correction(func, *args, **kwargs):
    """
    automatically adds the called correction_function to self._correct
    """
    self = args[0]
    if func.__name__ in self.correction:
        self.log.warning('CORRECTION {} has already been applied'.format(func.__name__))
    self.correction.append(func.__name__)
    return func(*args, **kwargs)


def get_result_recipe_name(func_name):
    """
    Takes the name of a result or a calculation method and extracts the result name and the method ame from it

    Returns
    -------
        result_name: str
        method_name: str
    """

    full_name = func_name.replace('calculate_', '').replace('result_', '')
    split = full_name.split('_')

    if split[-1].isupper():
        recipe = split[-1].lower()
        result_name = '_'.join(split[:-1])
    else:
        recipe = 'none'
        result_name = full_name

    return result_name, recipe


if __name__ == '__main__':
    import RockPy3.Packages.Mag.Measurements.hysteresis
    from pprint import pprint

    S = RockPy3.RockPyStudy()
    s = S.add_sample(name='test')
    # S.import_folder('/Users/mike/Google Drive/__code/RockPy3/mike_testing/auto_import')
    # S.info(sample_info=False)
    # m = S.get_measurement(mtype='hysteresis')[0]
    # m = RockPy3.Packages.Mag.Measurements.hysteresis.Hysteresis(sobj=s)
    m = s.add_simulation(mtype='hysteresis', ms=1)
    m.set_recipe('hf_sus', 'app2sat')
    # print(m.standards_result())
    # print(m.standards_calculate())
    # print(m.get_dependent_results(result='ms'))
    # print(m.get_dependent_results(result='mrs_ms'))
    m.calc_all()
    # print(m)
    # print(m.result_hf_sus(recalc=True, saturation_percent=90, ommit_last_n=1, check=True))
    # print(m.result_mrs_ms())
    print(m.results)
