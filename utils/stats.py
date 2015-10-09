__author__ = 'mike'
from lmfit import parameter
import numpy as np
from scipy.special import erfc


def normal(x, amp=1., center=300., sigma=10., dfunc='pdf', *args, **kwargs):
    '''
    parameters:
        amp: amplitude of function
        mean: mean of function - called 'center' in parameters
        standard deviation:
        variance:
        skewness:
        kurtosis:

    :option:
        output:
            pdf [standard] gives probability density function
            cdf

    :math:
       e^(-(x-mu)^2/(2 sigma^2))/(sqrt(2 pi) sigma)

    '''
    amp, center, sigma = float(amp), float(center), float(sigma)

    if dfunc == 'pdf':
        out_pdf = np.exp(-(x - center) ** 2 / (2 * sigma ** 2)) * (1 / np.sqrt(2 * np.pi) * sigma)
        out_pdf *= amp / max(out_pdf)
        return out_pdf

    if dfunc == 'cdf':
        # out_cdf = 0.5 * ( 1+ erfc((x-center)/()))
        out_cdf = amp * 0.5 * erfc((center - x) / (np.sqrt(2) * sigma))
        return out_cdf


def normal_skew(x, location=0, scale=1, shape=0):
    """
    Parameters
    ----------
       x: list
          x-values
       location: float
          location - similar to mean in unskewed
       scale: float
          scale
       shape: float
          shape
    """

    def pdf(x_input):
        return 1 / np.sqrt(2 * np.pi) * np.exp(-x_input ** 2 / 2)

    def cdf(x_input):
        return (1 + erfc(x_input / np.sqrt(2))) / 2

    location, scale = float(location), float(scale)
    t = (x - location) / scale
    return (2 / scale * pdf(t) * cdf(shape * t))
