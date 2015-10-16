import inspect

__author__ = 'mike'
from contextlib import contextmanager
from copy import deepcopy
import RockPy3
import logging
from functools import wraps


def colorscheme(scheme='simple'):
    colors = {'simple': ['r', 'g', 'b', 'c', 'm', 'y', 'k'] * 100,
              }
    return colors[scheme]


def create_logger(name):
    log = logging.getLogger(name=name)
    log.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(levelname)-10s %(name)-20s %(message)s', "%H:%M:%S")
    # formatter = logging.Formatter('%(asctime)s: %(levelname)-10s %(name)-20s %(message)s')
    # fh = logging.FileHandler('RPV3.log')
    # fh.setFormatter(formatter)
    ch = logging.StreamHandler()
    # ch.setLevel(logging.WARNING)
    ch.setLevel(logging.NOTSET)
    ch.setFormatter(formatter)
    # log.addHandler(fh)
    log.addHandler(ch)

    return log  # ch#, fh


def to_list(oneormoreitems):
    """
    convert argument to tuple of elements
    :param oneormoreitems: single number or string or list of numbers or strings
    :return: list of elements
    """
    if type(oneormoreitems) == list:
        return oneormoreitems
    elif type(oneormoreitems) == tuple:
        return [i for i in oneormoreitems]
    else:
        return [oneormoreitems]


def set_get_attr(obj, attr, value=None):
    """
    checks if attribute exists, if not, creates attribute with value None

    Parameters
    ----------
        obj: object
        attr: str

    Returns
    -------
        value(obj.attr)
    """
    if not hasattr(obj, attr):
        setattr(obj, attr, value)
    return getattr(obj, attr)


def append_if_not_exists(elist, element, operation):
    """
    appends an element to a list if it does not exist in list
    :param elist:
    :param element:
    :param operation:
    :return:
    """
    if operation == 'append':
        if not element in elist:
            elist.append(element)
    if operation == 'remove':
        if element in elist:
            elist.remove(element)
    return elist


def get_common_dtypes_from_list(mlist):
    """
    Takes a list of measurements and returns a list of common dtypes.

    Example
    -------
       mlist = [hys(down_field, up_field), hys(down_field, up_field, virgin)]
       returns ['down_field', 'up_field']

    Parameter
    ---------
       mlist: list
          list of measurements

    Returns
    -------
       dtypes
          sorted list of comment dtypes
    """
    # get intersection of all measurements with certain dtype
    dtypes = None
    if not mlist:
        raise ValueError('no measurements passed')
    for m in mlist:
        if not dtypes:
            dtypes = set(m.data.keys())
        else:
            dtypes = dtypes & set(m.data.keys())
    return sorted(list(dtypes))


@contextmanager
def ignored(*exceptions):
    try:
        yield
    except exceptions:
        pass


def get_full_argspec(func, args=None, kwargs=None):
    argspec = {}
    signature = inspect.signature(func)

    for i, v in enumerate(signature.parameters):
        try:
            # sets the value to the passed argument
            argspec.setdefault(v, args[i])
        except IndexError:  # no arg has been passed
            if not isinstance(signature.parameters[v].default, inspect._empty):
                argspec.setdefault(v, signature.parameters[v].default)
    return argspec


class plot(object):
    def __init__(self, single=False, mtypes='none', update_lims=True):  # todo define axis here? second_y?
        """
        If there are decorator arguments, the function
        to be decorated is not passed to the constructor!

        Parameter
        ---------
            mtypes: list(tuples)
                A list or list of tuples
            update_lims: bool
                default: True
                    if True updates both xlim and xlims to best fit the data
                    if False does not change the limits, might cause data to not be displayed

        Explanation:
        -----------
            the given mtype is used to filter all measurements to only use the ones where the feature is possible to
            produce.

            Example
            -------
                1. mtypes='hysteresis' will only use hysteresis measurements
                2. mtypes = ['armacq','trmacw' , 'irmacq'] can be calculated for ARM, TRM & IRM acquisition measurements
                    any of the three is available will be plotted
                3. mtypes = [('hysteresis', 'backfield')] needs a hysteresis and a backfield curve.
                    NOTE: The measurements need to have the same series!
                    #todo decide if we want to plot all possible or just the first combination.
                4. mtypes = [('mtype1', 'mtype2'),('mtype3','mtype4')] #todo cant think of an example but maybe possible

        """
        self.single = single
        self.update_lims = update_lims
        self.mtypes = tuple2list_of_tuples(mtypes)

    @staticmethod
    def short_feature_name(feature):
        return feature.__name__.replace('feature_', '')

    @staticmethod
    def plt_single_feature(feature, visual, *args, **kwargs):
        """
        plotting of a single feature
        """
        visual.log.debug('PLOTTING SINGLE FEATURE: {}'.format(feature.__name__))

        # get all lines in visual.ax object BEFORE the feature is plotted
        old_lines = set(visual.ax.lines)
        feature(visual, *args, **kwargs)

        # get all NEW lines in visual.ax object AFTER the feature is plotted
        new_lines = [i for i in visual.ax.lines if i not in old_lines]
        visual.linedict.setdefault(feature.__name__, []).extend(new_lines)

    def __call__(self, feature, idx=None):
        """
        If there are decorator arguments, __call__() is only called
        once, as part of the decoration process! You can only give
        it a single argument, which is the function object.
        """

        def wrapped_feature(*args, **kwargs):
            # format the argspec
            parameter = get_full_argspec(func=feature, args=args, kwargs=kwargs)
            visual = parameter['self']

            # update the plt_props of the feature
            kwargs.setdefault('plt_props', {})

            idx = kwargs.pop('idx', '')
            feature_name = feature.__name__.replace('feature_', '') + str(idx)
            visual_props = visual.plt_props[feature_name]
            kwargs['plt_props'].update(visual_props)

            if self.single:
                self.plt_single_feature(feature=feature, visual=visual, **kwargs)
            else:
                for mtype in self.mtypes:
                    for sample in visual._plt_input['sample']:
                        mlist = sample.get_measurement(mtype=mtype)
                        if len(mtype) > 1:
                            mobj = MlistToTupleList(mlist, mtype)
                        elif len(mtype) == 1:
                            mobj = mlist
                        else:
                            visual.log.error(
                                'FEATURE {} input doesnt match mtype requirements {}'.format(feature_name, mtype))
                        for mt_tuple in mobj:
                            try:
                                kwargs['plt_props'].update(mt_tuple[0].plt_props)
                            except TypeError:
                                kwargs['plt_props'].update(mt_tuple.plt_props)
                            kwargs['plt_props'].update(visual._plt_props_forced[feature_name])
                            feature(visual, mobj=mt_tuple, **kwargs)

                    if len(mtype) > 1:
                        mobj = MlistToTupleList(visual._plt_input['measurement'], mtype)
                    elif len(mtype) == 1:
                        mobj = visual._plt_input['measurement']
                    else:
                        visual.log.error(
                            'FEATURE {} input doesnt match mtype requirements {}'.format(feature.__name__, mtype))
                    for mt_tuple in mobj:
                        try:
                            kwargs['plt_props'].update(mt_tuple[0].plt_props)
                        except TypeError:
                            kwargs['plt_props'].update(mt_tuple.plt_props)
                        kwargs['plt_props'].update(visual._plt_props_forced[feature.__name__.replace('feature_', '')])
                        feature(visual, mobj=mt_tuple, **kwargs)

        return wrapped_feature


def MlistToTupleList(mlist, mtypes, ignore_series=False):
    """
    Transforma a list of measurements into a tuplelist, according to the mtypes specified.

    Parameter
    ---------
        mlist: list
            list of RockPy Measurements
        mtypes: tuple
            tuple for the mlist to be organized by.

    Example
    -------
        1. multiple mtypes (tuple)
            mlist = [Hys(S1), Hys(S2), Coe(S1), Coe(S2)] assuming that S1 means all series are the same
            mtypes = (hys, coe)

            1. the list is sorted into a dictionary with mtype:list(m)
            2. for each member of dict[hys] a corresponding coe measurement is searched which has
                a) the same parent sample
                b) exactly the same series


    """
    # create the dictionary
    mdict = {mtype: [m for m in mlist if m.mtype == mtype] for mtype in mtypes}

    out = []
    for m_mtype1 in mdict[mtypes[0]]:
        aux = [m_mtype1]
        for mtype in mtypes[1:]:
            for m_mtype_n in mdict[mtype]:
                if not m_mtype1.sample_obj == m_mtype_n.sample_obj:
                    break
                if compare_measurement_series(m_mtype1, m_mtype_n) or ignore_series:
                    aux.append(m_mtype_n)
                    if not ignore_series:
                        break
        out.append(tuple(aux))
    return out


def compare_measurement_series(m1, m2):
    """
    returns True if both series have exactly the same series.

    Parameter
    ---------
        m1: RockPy.Measeurement
        m2: RockPy.Measeurement

    Note
    ----
        ignores multiples of the same series
    """
    s1 = m1.series
    s2 = m2.series

    if all(s in s2 for s in s1) and all(s in s1 for s in s2):
        return True
    else:
        return False


def kwargs_to_calculation_parameter(rpobj=None, mtype_list=None, result=None, **kwargs):
    """
    looks though all provided kwargs and searches for possible calculation parameters. kwarg key naming see:
    separate_mtype_method_parameter

    Parameters
    ----------
        rpobj: RockPy Object (Study SampleGroup, Sample, Measurement
            default: None
        mtypes: list
            list of possible mtypes, parameters are filtered for these
        result: str
            parameters are filtered
    Hierarchy
    ---------
        4. passing a pure parameter causes all methods in all mtypes to be set to that value
           !!! this will be overwritten by !!!
        3. passing a mtype but no method
           !!! this will be overwritten by !!!
        2. passing a method
           !!! this will be overwritten by !!!
        1. passing a method and mtype
           !!! this will be overwritten by !!!

    Note
    ----
        passing a mtype:
            will add mtype: { all_methods_with_parameter: {parameter:parameter_value}} to calculation_parameter dict
        passing a method:
            will add all_mtypes_with_method: { method: {parameter:parameter_value}} to calculation_parameter dict
        passing mtype and method:
            will add all_mtypes_with_method: { all_methods_with_parameter: {parameter:parameter_value}} to calculation_parameter dict

    Returns
    -------
        calculation_parameter: dict
            dictionary with the parameters passed to the method, where a calculation_method could be found
        kwargs: dict

    computation notes:
        if only parameter is specified: look for all mtypes with the parameter and all methods with the parameter
        if mtype specified: look for all methods with the parameter
    """

    # the kwargs need to be sorted so that they follow the hierarchy
    # 1. parameter only
    # 2. mtype & parameter
    # 3. method & parameter
    # 4. mtype & method & parameter

    param_only = [i for i in kwargs if [i] == i.split('___') if [i] == i.split('__')]
    mtype_only = [i for i in kwargs if [i] == i.split('___') if [i] != i.split('__')]
    method_only = [i for i in kwargs if [i] == i.split('__') if [i] != i.split('___')]
    mixed = [i for i in kwargs if [i] != i.split('__') if [i] != i.split('___')]

    kwarg_list = param_only + mtype_only + method_only + mixed

    calc_params = {}

    for kwarg in kwarg_list:
        remove = False
        mtypes, methods, parameter = RockPy3.core.utils.separate_mtype_method_parameter(kwarg=kwarg)

        # get all mtypes, methods if not specified in kwarg
        # nothing specified
        if not mtypes and not methods:
            mtypes = [mtype for mtype, params in RockPy3.Measurement.mtype_calculation_parameter_list().items() if
                      parameter in params]

            # filter only given in mtype_list
            if mtype_list:
                mtypes = [mtype for mtype in mtypes if mtype in mtype_list]

        # no mtype specified
        elif not mtypes:
            # we need to add methods with recipes:
            # bc___recipe = 'simple' would otherwise not be added because there is no method calculate_bc
            for method in methods:
                for calc_method, method_params in RockPy3.Measurement.method_calculation_parameter_list().items():
                    if calc_method.split('_')[-1].isupper() and ''.join(calc_method.split('_')[:-1]) == method:
                        methods.append(calc_method)

            mtypes = [mtype for mtype, mtype_methods in RockPy3.Measurement.mtype_calculation_parameter().items()
                      if any(method in mtype_methods.keys() for method in methods)]
            # filter only given in mtype_list
            if mtype_list:
                mtypes = [mtype for mtype in mtypes if mtype in mtype_list]

        if not methods:
            methods = [method for method, params in RockPy3.Measurement.method_calculation_parameter_list().items()
                       if
                       parameter in params]
        # print(mtypes, methods, parameter, rpobj)

        # i an object is given, we can filter the possible mtypes, and methods further
        # 1. a measurement object
        if isinstance(rpobj, RockPy3.Measurement):
            mtypes = [mtype for mtype in mtypes if mtype == rpobj.mtype]
            methods = [method for method in methods if method in rpobj.possible_calculation_parameter()]
            # print(mtypes, methods, parameter)

        # 2. a sample object
        if isinstance(rpobj, RockPy3.Sample):
            mtypes = [mtype for mtype in mtypes if mtype == rpobj.mtypes]

        # 3. a samplegroup object
        # 4. a study object
        # 5. a visual object
        # 6. a result
        if result:
            methods = [method for method in methods if method in rpobj.possible_calculation_parameter()[result]]


        # if isinstance(rpobj, RockPy3.Visualize.base.Visual):
        #     mtypes = [mtype for mtype in mtypes if mtype in rpobj.__class__._required]

        # todo RockPy3.study, RockPy3.samplegroup

        ############################################################################################################
        # actual calculation
        for mtype in mtypes:
            # get the only  methods that are implemented in the mtype to be checked
            check_methods = set(RockPy3.Measurement.mtype_calculation_parameter()[mtype]) & set(methods)

            for method in check_methods:
                # with ignored(KeyError): # ignore keyerrors if mtype / method couple does not match
                try:
                    if parameter in RockPy3.Measurement.mtype_calculation_parameter()[mtype][method]:
                        RockPy3.logger.debug('PARAMETER found in << %s, %s >>' % (mtype, method))
                        remove = True
                        calc_params.setdefault(mtype, dict())
                        calc_params[mtype].setdefault(method, dict())
                        calc_params[mtype][method].update({parameter: kwargs[kwarg]})
                    else:
                        RockPy3.logger.error(
                            'PARAMETER << %s >> NOT found in << %s, %s >>' % (parameter, mtype, method))
                except KeyError:
                    RockPy3.logger.debug(
                        'PARAMETER << %s >> not found mtype, method pair probably wrong << %s, %s >>' % (
                            parameter, mtype, method))
        if remove:
            kwargs.pop(kwarg)

    return calc_params, kwargs


def compare_measurement_series(m1, m2):
    """
    returns True if both series have exactly the same series.

    Parameter
    ---------
        m1: RockPy3.Measeurement
        m2: RockPy3.Measeurement

    Note
    ----
        ignores multiples of the same series
    """
    s1 = m1.series
    s2 = m2.series

    if all(s in s2 for s in s1) and all(s in s1 for s in s2):
        return True
    else:
        return False


def MlistToTupleList(mlist, mtypes):
    """
    Transforma a list of measurements into a tuplelist, according to the mtypes specified.

    Parameter
    ---------
        mlist: list
            list of RockPy Measurements
        mtypes: tuple
            tuple for the mlist to be organized by.

    Example
    -------
        mlist = [Hys(S1), Hys(S2), Coe(S1), Coe(S2)] assuming that S1 means all series are the same
        mtypes = (hys, coe)

        1. the list is sorted into a dictionary with mtype:list(m)
        2. for each member of dict[hys] a corresponding coe measurement is searched which has
            a) the same parent sample
            b) exactly the same series
    """
    # create the dictionary
    mdict = {mtype: [m for m in mlist if m.mtype == mtype] for mtype in mtypes}
    # print mdict
    out = []

    for m_mtype1 in mdict[mtypes[0]]:
        # print m_mtype1
        aux = [m_mtype1]
        for mtype in mtypes[1:]:
            for m_mtype_n in mdict[mtype]:
                if not m_mtype1.sobj == m_mtype_n.sobj:
                    break
                if RockPy3.utils.general.compare_measurement_series(m_mtype1, m_mtype_n):
                    aux.append(m_mtype_n)
                    break
        out.append(tuple(aux))
    return out


def get_full_argspec_old(func, args, kwargs=None):
    """
    gets the full argspec from a function including the default values.

    Raises
    ------
        TypeError if the wrong number of args is gives for a number of arg_names
    """
    if not kwargs:
        kwargs = {}
    arg_names, varargs, varkw, defaults = inspect.getargspec(func=func)

    if defaults:
        args += (defaults)

    try:
        # parameters = {arg_names.pop(0): func}
        parameters = {arg: args[i] for i, arg in enumerate(arg_names)}
        parameters.update(kwargs)
    except IndexError:
        raise TypeError('{} takes exactly {} argument ({} given)'.format(func.__name__, len(arg_names), len(args)))
    return parameters


def separate_mtype_method_parameter(kwarg):
    """
    separetes the possible kwarg parameters for calculation_parameter lookup.
    the mtype has to be followed by 3x_ the methodname by 2x _ because calculation_parameter may be separated by 1x_

    Format
    ------
        mtype__method___parameter or multiples of mtyep, method
        mtype1__mtype2__method1___method2___parameter but always only one parameter LAST
        mtype1__method1___mtype2__method2___parameter mixed also possible


    Retuns
    ------
        mtype___method__parameter: [mtype], [method], parameter
        mtype___parameter: [mtype], [], parameter
        method__parameter: [], [method], parameter
        parameter: [], [], parameter
    """
    method = []
    mtype = []
    parameter = None

    # all possible methods, mtypes and the parameter
    possible = [j for i in kwarg.split('___') for j in i.split('__')]

    for i in possible:
        # remove the part
        kwarg = kwarg.replace(i, '')
        if kwarg.startswith('___'):
            method.append(i)
            kwarg = kwarg[3:]
        if kwarg.startswith('__'):
            mtype.append(i)
            kwarg = kwarg[2:]

    parameter = possible[-1]
    return mtype, method, parameter


def separate_calculation_parameter_from_kwargs(rpobj=None, mtype_list=None, **kwargs):
    """
    separates kwargs from calcuzlation arameters, without changing the signature
        e.g. hysteresis__no_points = n !-> hystersis:{no_points: n}

    """
    calculation_parameter, non_calculation_parameter = kwargs_to_calculation_parameter(rpobj=rpobj,
                                                                                       mtype_list=mtype_list, **kwargs)

    out = {}

    for key, value in kwargs.items():
        if not key in non_calculation_parameter:
            out.setdefault(key, value)

    return out, non_calculation_parameter


def tuple2list_of_tuples(item):
    """
    Takes a list of tuples or a tuple and returns a list of tuples

    Parameters
    ----------
       input: list, tuple

    Returns
    -------
       list
          Returns a list of tuples, if input is a tuple it converts it to a list of tuples
          if input == a list of tuples will just return input
    """
    if type(item) != tuple and type(item) != list:
        item = tuple([item])
    if type(item) == tuple:
        aux = list()
        aux.append(item)
        item = aux
    return item
