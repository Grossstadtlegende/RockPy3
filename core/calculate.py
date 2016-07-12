import numpy as np
import scipy.optimize

def endmember_unmixing(data, end_members=2, maxiter=1000):

    """
    End-member unmixing adapted from:

    Heslop, D., and M. Dillon (2007), Unmixing magnetic remanence curves without a priori knowledge, Geophysical Journal International, 170(2), 556–566, doi:10.1111/j.1365-246X.2007.03432.x.
    Parameters
    ----------
    data: ndarray
        l-columns representing the field steps
        n-rows representing the measurement data
    end_members: int
        how many end-members are calculated
    maxiter:
        maximum iterations of the fitting routine

    Returns
    -------
    A: matrix of coefficients
        m columns representing the end members
        n rows for each measurement
    S: matrix of values
        l columns for each field step
        m rows one for each end member
    A.S: matrix with n rows for each measurement displaying the sum of the components e.g. the fit
    E: matrix with the errrors with n lines for each measurement

    Notes
    -----
    You can plot each member assuming 3 end members with i measurements, with plot( fields, S[0]*A[i][0])
    and the sum of them with plot( fields, S[0]*A[i][0]+S[1]*A[i][1]+S[2]*A[i][2])
    """
    X = data
    l = data.shape[1]
    m = end_members  # end members
    n = data.shape[0]
    eps = 1e-15 # no div/0

    # initialize the matrixes with random numbers
    A = np.random.random_sample((n, m))
    S = np.random.random_sample((m, l))

    #calculation
    for i in range(maxiter):
        S = S * np.dot(A.T, X) / (np.dot(A.T, np.dot(A, S)) + eps)
        A = A * np.dot(X, S.T) / (np.dot(A, np.dot(S, S.T)) + eps)

    return A, S, np.dot(A,S), X-np.dot(A,S)
