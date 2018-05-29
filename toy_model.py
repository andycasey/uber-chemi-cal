
"""
Toy model for latent variable model in chemical abundance space.
"""

import numpy as np

import stan_utils as stan

np.random.seed(42)

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

    X = np.random.normal(
        np.zeros(D),
        scales**2,
        size=(N, D))

    theta = np.random.normal(0, 1, size=(D, M))

    # TODO: be smrt
    y = np.zeros((N, M, D), dtype=float)
    for i in range(N):
        y[i] = X[i] * theta.T

    # add noise.
    psi = np.random.normal(0, 1, size=D)**2
    phi = np.random.normal(0, 1, size=(M, D))**2

    rank = np.random.normal(0, 1, size=y.shape)

    noise = np.zeros_like(y)
    for m in range(M):
        noise[:, m] = rank[:, m] * (phi[m] + psi)

    y += noise

    truths = dict(X=X, theta=theta, psi=psi, phi=phi, scales=scales)

    return (y, truths)


y, truths = generate_data(N=256, M=3, D=5)

assert 0

model = stan.read_model("toy-model.stan")

