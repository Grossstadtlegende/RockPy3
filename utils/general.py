__author__ = 'mike'
import logging
import inspect
import time

import numpy as np
import matplotlib.pyplot as plt

from math import degrees, radians
from math import sin, cos, tan, asin, atan2
import RockPy3


def get_date_str():
    return time.strftime("%d.%m.%Y")


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


def get_full_argspec(func, args, kwargs=None):
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
        mtypes, methods, parameter = RockPy3.utils.general.separate_mtype_method_parameter(kwarg=kwarg)
        # print(mtypes, methods, parameter)
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
        print(mtypes, methods, parameter, rpobj)

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


def create_logger(name):
    log = logging.getLogger(name=name)
    log.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s: %(levelname)-10s %(name)-20s %(message)s')
    # fh = logging.FileHandler('RPV3.log')
    # fh.setFormatter(formatter)
    ch = logging.StreamHandler()
    # ch.setLevel(logging.WARNING)
    ch.setLevel(logging.NOTSET)
    ch.setFormatter(formatter)
    # log.addHandler(fh)
    log.addHandler(ch)

    return log  # ch#, fh


def create_dummy_measurement(mtype, fpath=None, ftype=None, idx=0, mdata=None, sample=None):
    s = RockPy3.Sample(name='dummy_sample')
    m = s.add_measurement(mtype=mtype, fpath=fpath, ftype=ftype,  # general
                          idx=idx, mdata=mdata,
                          )
    if sample:
        m.sobj = sample
    return m


def differentiate(data_list, diff=1, smoothing=1, norm=False, check=False):
    """
    caclulates a smoothed (if smoothing > 1) derivative of the data

    :param data_list: ndarray - 2D array with xy[:,0] = x_data, xy[:,1] = y_data
    :param diff: int - the derivative to be calculated f' -> diff=1; f'' -> diff=2
          default = 1
    :param smoothing: int - smoothing factor
    :param norm:
    :param check:
    :return:
    """
    log = logging.getLogger('RockPy3.FUNCTIONS.general.diff')
    log.info('DIFFERENTIATING\t data << %i derivative - smoothing: %i >>' % (diff, smoothing))
    data_list = np.array(data_list)
    # getting X, Y data
    X = data_list[:, 0]
    Y = data_list[:, 1]

    # ''' test with X^3'''
    # X = np.linspace(-10,10,1000)
    # Y = X**3
    # Y2 = 3*X**2
    # Y3 = 6*X

    # # derivative
    for i in range(diff):
        deri = [[X[i], (Y[i + smoothing] - Y[i - smoothing]) / (X[i + smoothing] - X[i - smoothing])] for i in
                range(smoothing, len(Y) - smoothing)]
        deri = np.array(deri)
        X = deri[:, 0]
        Y = deri[:, 1]
    MAX = max(abs(deri[:, 1]))

    if norm:
        deri[:, 1] /= MAX
    if check:
        if norm:
            plt.plot(data_list[:, 0], data_list[:, 1] / max(data_list[:, 1]))
        if not norm:
            plt.plot(data_list[:, 0], data_list[:, 1])
        plt.plot(deri[:, 0], deri[:, 1])
        plt.show()

    return deri


def rotate(xyz, axis='x', degree=0, *args):
    """


    :rtype : object
    :param x:
    :param y:
    :param z:
    :param axis:
    :param degree:
    :return:
    """
    a = radians(degree)

    RX = [[1, 0, 0],
          [0, cos(a), -sin(a)],
          [0, sin(a), cos(a)]]

    RY = [[cos(a), 0, sin(a)],
          [0, 1, 0],
          [-sin(a), 0, cos(a)]]

    RZ = [[cos(a), -sin(a), 0],
          [sin(a), cos(a), 0],
          [0, 0, 1]]

    if axis.lower() == 'x':
        out = np.dot(xyz, RX)
    if axis.lower() == 'y':
        out = np.dot(xyz, RY)
    if axis.lower() == 'z':
        out = np.dot(xyz, RZ)

    return out


def abs_min_max(list):
    list = [i for i in list if not i == np.nan or not i == inf]
    min_idx = np.argmin(np.fabs(list))
    # print min_idx
    max_idx = np.argmax(np.fabs(list))
    return list[min_idx], list[max_idx]


def XYZ2DIL(XYZ):
    """
    convert XYZ to dec, inc, length
    :param XYZ:
    :return:
    """
    DIL = []
    L = np.linalg.norm(XYZ)
    D = degrees(atan2(XYZ[1], XYZ[0]))  # calculate declination taking care of correct quadrants (atan2)
    if D < 0: D = D + 360.  # put declination between 0 and 360.
    if D > 360.: D = D - 360.
    DIL.append(D)  # append declination to Dir list
    I = degrees(asin(XYZ[2] / L))  # calculate inclination (converting to degrees)
    DIL.append(I)  # append inclination to Dir list
    DIL.append(L)  # append vector length to Dir list
    return DIL


def DIL2XYZ(DIL):
    """
    Convert a tuple of D,I,L components to a tuple of x,y,z.
    :param DIL:
    :return: (x, y, z)
    """
    (D, I, L) = DIL
    H = L * cos(radians(I))
    X = H * cos(radians(D))
    Y = H * sin(radians(D))
    Z = H * tan(radians(I))
    return (X, Y, Z)


def DI2XYZ(DI):
    """
    Convert a tuple of D,I to a tuple of x,y,z. Assuming unit length
    :param DI: declination, inclination
    :return: (x, y, z)
    """
    DI.append(1)
    return DIL2XYZ(DI)


def MirrorDirectionToNegativeInclination(dec, inc):
    if inc > 0:
        return (dec + 180) % 360, -inc
    else:
        return dec, inc


def MirrorDirectionToPositiveInclination(dec, inc):
    if inc < 0:
        return (dec + 180) % 360, -inc
    else:
        return dec, inc


def MirrorVectorToNegativeInclination(x, y, z):
    if z > 0:
        return -x, -y, -z
    else:
        return x, y, z


def MirrorVectorToPositiveInclination(x, y, z):
    if z < 0:
        return -x, -y, -z
    else:
        return x, y, z


def Proj_A_on_B_scalar(A, B):
    """
    project a vector A on a vector B and return the scalar result
    see http://en.wikipedia.org/wiki/Vector_projection
    :param A: vector which will be projected on vector B
    :param B: vector defining the direction
    :return: scalar value of the projection A on B
    """
    return np.dot(A, B) / np.linalg.norm(B)


# coordinate transformations
def core2geo(core_xyz, coreaz, coredip, invdrilldir=False):
    """convert core coordinates to geographic coordinates"""
    # if invdrilldir == True -> z-mark done or samples inserted in the wrong way during measurement -->> adapt coreaz and and coredip accordingly
    if invdrilldir:
        coreaz = (coreaz + 180) % 360
        coredip = 180 - coredip
    # rotate around y-axis to compensate core dip
    xyz = RotateVector(core_xyz[0:3], YRot(-coredip))
    # rotate around z-axis
    geo_xyz = RotateVector(xyz, ZRot(-coreaz))

    # do we have additional elements? Append the 4th (e.g. label)
    if len(core_xyz) > 3:
        geo_xyz = (geo_xyz[0], geo_xyz[1], geo_xyz[2], core_xyz[3])

    return geo_xyz


def geo2core(geo_xyz, coreaz, coredip, invdrilldir=False):
    """convert core coordinates to geographic coordinates"""
    # if invdrilldir == True -> z-mark done or samples inserted in the wrong way during measurement -->> adapt coreaz and and coredip accordingly
    if invdrilldir:
        coreaz = (coreaz + 180) % 360
        coredip = 180 - coredip
    # rotate around z-axis
    xyz = RotateVector(geo_xyz[0:3], ZRot(coreaz))
    # rotate around y-axis to compensate core dip
    core_xyz = RotateVector(xyz, YRot(coredip))

    # do we have additional elements? Append the 4th (e.g. label)
    if len(geo_xyz) > 3:
        core_xyz = (core_xyz[0], core_xyz[1], core_xyz[2], geo_xyz[3])

    return core_xyz


def core2geoTensor(core_T, coreaz, coredip, invdrilldir=False):
    """ convert core coordinates to geographic coordinates """
    # if invdrilldir == True -> z-mark done or samples inserted in the wrong way during measurement -->> adapt coreaz and and coredip accordingly
    if invdrilldir:
        coreaz = (coreaz + 180) % 360
        coredip = 180 - coredip
    # rotate around y-axis to compensate core dip
    T = RotateTensor(core_T, YRot(-coredip))
    # rotate around z-axis
    geo_T = RotateTensor(T, ZRot(-coreaz))
    return geo_T


def geo2bed(geo_xyz, bedaz, beddip):
    """ convert geographic coordinates to bedding corrected coordinates """
    # TEST: XYZ2DIL( geo2bed( (.3,.825,1.05), 40, 20)) -> D=62, I=32 (Butler S.72)
    # rotate around z-axis until strike is along y axis of fixed system
    xyz = RotateVector(geo_xyz[0:3], ZRot(bedaz))
    # now rotate around y-axis to compensate bed dip
    xyz = RotateVector(xyz, YRot(-beddip))
    # rotate back around z-axis
    bed_xyz = RotateVector(xyz, ZRot(-bedaz))

    # do we have additional elements? Append the 4th (e.g. label)
    if len(geo_xyz) > 3:
        bed_xyz = (bed_xyz[0], bed_xyz[1], bed_xyz[2], geo_xyz[3])

    return bed_xyz


def bed2geo(bed_xyz, bedaz, beddip):
    """ convert geographic coordinates to bedding corrected coordinates """
    # rotate back around z-axis
    xyz = RotateVector(bed_xyz[0:3], ZRot(bedaz))
    # now rotate around y-axis to compensate bed dip
    xyz = RotateVector(xyz, YRot(beddip))
    # rotate around z-axis until strike is along y axis of fixed system
    geo_xyz = RotateVector(xyz, ZRot(-bedaz))

    # do we have additional elements? Append the 4th (e.g. label)
    if len(bed_xyz) > 3:
        geo_xyz = (geo_xyz[0], geo_xyz[1], geo_xyz[2], bed_xyz[3])

    return geo_xyz


def geo2bedTensor(geo_T, bedaz, beddip):
    """ convert geographic coordinates to bedding corrected coordinates """
    # rotate around z-axis until strike is along y axis of fixed system
    T = RotateTensor(geo_T, ZRot(bedaz))
    # now rotate around y-axis to compensate bed dip
    T = RotateTensor(T, YRot(-beddip))
    # rotate back around z-axis
    bed_T = RotateTensor(T, ZRot(-bedaz))

    return bed_T


def coord_transform(xyz, initial, final, coreaz=None, coredip=None, bedaz=None, beddip=None):
    """
    transform vector xyz in initial coordinates to final coordinates

    Parameters
    ----------
        initial: string
            actual coordinate system
        final: string
            target coordinate system

        returns:
            vector xyz in final coordinates
    """

    if check_coordinate_system(initial) is None or check_coordinate_system(final) is None:
        return

    if initial == 'core':
        if final == 'geo' or final == 'bed':
            final_xyz = core2geo(xyz, coreaz, coredip)
        if final == 'bed':
            final_xyz = geo2bed(final_xyz, bedaz, beddip)
    elif initial == 'geo':
        if final == 'core':
            final_xyz = geo2core(xyz, coreaz, coredip)
        elif final == 'bed':
            final_xyz = geo2bed(xyz, bedaz, beddip)
    elif initial == 'bed':
        if final == 'geo' or final == 'core':
            final_xyz = bed2geo(xyz, bedaz, beddip)
        if final == 'core':
            final_xyz = geo2core(final_xyz, coreaz, coredip)
    return final_xyz


def CreateRotMat(angles):
    """Generate rotation matrix out of 3 Euler angles (a, b, c)."""

    A = ZRot(angles[0])
    B = XRot(angles[1])
    C = ZRot(angles[2])

    return np.dot(A, np.dot(B, C))


def XRot(xrot):
    """Generate rotation matrix for a rotation of xrot degrees around the x-axis"""
    a = radians(xrot)

    return np.array(((1, 0, 0), (0, cos(a), -sin(a)), (0, sin(a), cos(a))))


def YRot(yrot):
    """Generate rotation matrix for a rotation of yrot degrees around the y-axis"""
    a = radians(yrot)

    return np.array(((cos(a), 0, sin(a)), (0, 1, 0), (-sin(a), 0, cos(a))))


def ZRot(zrot):
    """Generate rotation matrix for a rotation of zrot degrees around the z-axis"""
    a = radians(zrot)

    return np.array(((cos(a), -sin(a), 0), (sin(a), cos(a), 0), (0, 0, 1)))


def RotateVector(v, rotmat):
    """Rotate a vector v = (x,y,z) about 3 Euler angles (x-convention) defined by matrix rotmat."""
    # return resulting vector
    return np.dot(v, rotmat)


def RotateTensor(T, rotmat):
    """ rotate 3x3 tensor about 3 Euler angles (x-convention) defined by matrix rotmat."""
    # return resulting tensor
    return np.dot(rotmat.T, dot(T, rotmat))


def add_unit(value, unit):
    out = str(value) + '[' + unit + ']'
    return out


def to_list(oneormoreitems):
    """
    convert argument to tuple of elements
    :param oneormoreitems: single number or string or list of numbers or strings
    :return: tuple of elements
    """
    return oneormoreitems if hasattr(oneormoreitems, '__iter__') else [oneormoreitems]


def plt_logo():
    x = np.linspace(-5, 5)
    y1 = np.tanh(x)
    y2 = np.tanh(x - 1)
    y3 = np.tanh(x + 1)

    y4, y5 = x * np.sin(np.pi) + np.cos(np.pi) * np.tanh(x), x * np.sin(np.pi) + np.cos(np.pi) * np.tanh(x - 1),
    y6, y7 = x * np.sin(np.pi) + np.cos(np.pi) * np.tanh(x), x * np.sin(np.pi) + np.cos(np.pi) * np.tanh(x + 1),

    plt.fill_between(x, y1, y2, color='#009440')
    plt.fill_between(x, y1, y3, color='#7F7F7F')

    plt.plot(x, y1, 'k')
    plt.plot(x, y2, 'k')
    plt.plot(x, y3, 'k')
    plt.plot(x, y4, 'k', zorder=100)
    plt.plot(x, y5, 'k', zorder=100)
    plt.plot(x, y6, 'k', zorder=100)
    plt.plot(x, y7, 'k', zorder=100)
    plt.fill_between(x, y4, y5, color='#009440', zorder=100)
    plt.fill_between(x, y6, y7, color='#7F7F7F', zorder=100)
    fig = plt.gcf()
    # circle1 = plt.Circle((0,0),4,color='w', alpha=0.1)
    # fig.gca().add_artist(circle1)

    c1 = np.sqrt(10 - x ** 2)
    c2 = -np.sqrt(10 - x ** 2)

    plt.plot(x, c1, 'k')
    plt.plot(x, c2, 'k')
    ax = fig.gca()
    ax.axis('equal')
    ax.set_xlim(-3.3, 3.3)
    ax.set_ylim(-3.3, 3.3)
    plt.show()


def compare_array(A, B, AinB=True):
    """
    compares two numpy arrays
    :param A:
    :param B:
    :param AinB: bool
       if true: returns true if A[i] in B
       if False returns False if A[i] in B
    :return:
    """

    out = []
    for i in A:
        if i in B:
            if AinB:
                out.append(True)
            else:
                out.append(False)
        else:
            if AinB:
                out.append(False)
            else:
                out.append(True)
    return out


def check_coordinate_system(coord):
    """
    checks if coord has a valid value
    """
    logger = logging.getLogger(__name__)
    if not coord in RockPy3.coordinate_systems:
        logger.error('invalid coordinate system << %s >> specified.' % str(coord))
        return None
    return coord


