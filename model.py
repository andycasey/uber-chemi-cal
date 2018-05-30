import numpy as np
import matplotlib.pyplot as plt



def generate_data(N, M, D, scales=None, seed=None):
    """
    Generate some toy data to play with. Here we assume all :math:`N` stars have
    been observed by all :math:`M` surveys.

    :param N:
        The number of stars observed.

    :param M:
        The number of surveys.

    :param D:
        The dimensionality of the label space.

    :param scales: [optional]
        Optional values to provide for the relative scales on the latent factors.

    :param seed: [optional]
        An optional seed to provide to the random number generator.

    :returns:
        A two-length tuple containing the data :math:`y` and a dictionary with
        the true values.
    """

    if seed is not None:
        np.random.seed(seed)

    if scales is None:
        scales = np.abs(np.random.normal(0, 1, size=D))

    else:
        scales = np.array(scales)

    assert len(scales) == D

    X = np.random.normal(
        np.zeros(D),
        scales,
        size=(N, D))

    theta = np.random.normal(0, 1, size=(D, M))

    # TODO: Better way to randomly generate positive semi-definite covariance
    #       matrices that are *very* close to an identity matrix.

    # Use decomposition to ensure the resulting covariance matrix is positive
    # semi-definite.
    L = np.random.randn(M, D, D)
    L[:, np.arange(D), np.arange(D)] = np.exp(L[:, np.arange(D), np.arange(D)])
    i, j = np.triu_indices_from(L[0], 1)
    L[:, i, j] = 0.0

    # TODO: use matrix multiplication you idiot
    theta = np.array([np.dot(L[i], L[i].T) for i in range(M)])



    raise a

    # TODO: be smrt
    y = np.zeros((N, M, D), dtype=float)
    for i in range(N):
        y[i] = X[i] * theta.T

    # add noise.
    phi = np.abs(np.random.normal(0, 1, size=(M, D)))
    rank = scales * np.random.normal(0, 1, size=y.shape)

    noise = rank * phi 

    y += noise

    truths = dict(X=X, theta=theta.T, phi=phi.T, scales=scales)

    return (y, truths)


generate_data(1000, 3, 5)