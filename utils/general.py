__author__ = 'mike'
import logging
import time

import numpy as np
import matplotlib.pyplot as plt

from math import degrees, radians
from math import sin, cos, tan, asin, atan2
import RockPy3


def get_date_str():
    return time.strftime("%d.%m.%Y")


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


