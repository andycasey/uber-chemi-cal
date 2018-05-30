import numpy as np
import matplotlib.pyplot as plt

import stan_utils as stan
from mpl_utils import (mpl_style, common_limits)


plt.style.use(mpl_style)


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

    y = np.dot(X, theta)

    # add noise.
    phi = np.abs(np.random.normal(0, 0.1, size=(M, D)))
    rank = np.random.normal(0, 1, size=y.shape)

    noise = scales * rank * phi 

    y += noise

    truths = dict(X=X, theta=theta, phi=phi, scales=scales, L=L, noise=noise)

    return (y, truths)


N, M, D = (250, 10, 5)
scales =  np.ones(D)

y, truths = generate_data(N=N, M=M, D=D, scales=scales)

model = stan.read_model("model.stan")


# Optimize the model
data = dict(y=y, N=N, M=M, D=D, scales=scales)

# TODO: initialize from true value.
op_kwds = dict(
    data=data, 
    iter=100000, 
    tol_obj=7./3 - 4./3 - 1, # machine precision
    tol_grad=7./3 - 4./3 - 1, # machine precision
    tol_rel_grad=1e3,
    tol_rel_obj=1e4
)


p_opt = model.optimizing(**op_kwds)



fig, axes = plt.subplots(1, D, figsize=(4 * D, 4))
for d, ax in enumerate(axes):

    x = truths["X"].T[d]
    y = p_opt["X"].T[d]

    ax.scatter(x, y)

    ax.set_xlabel(r"$X_{{{0},\textrm{{true}}}}$".format(d))
    ax.set_ylabel(r"$X_{{{0},\textrm{{opt}}}}$".format(d))

    common_limits(ax, plot_one_to_one=True)

fig.tight_layout()



fig, axes = plt.subplots(1, D, figsize=(4 * D, 4))
for d, ax in enumerate(axes):

    x = truths["phi"].T[d]
    y = p_opt["phi"][d]

    ax.scatter(x, y)

    ax.set_xlabel(r"$\phi_{{{0},\textrm{{true}}}}$".format(d))
    ax.set_ylabel(r"$\phi_{{{0},\textrm{{opt}}}}$".format(d))

    common_limits(ax, plot_one_to_one=True)

fig.tight_layout()




K = int(np.ceil(M**0.5))
L = int(np.ceil(M / K))

fig, axes = plt.subplots(K, L, figsize=(4 * L, 4 * K))
axes = np.array(axes).flatten()

for m, ax in enumerate(axes[:M]):

    x = truths["theta"][m].flatten()
    y = p_opt["theta"][m].flatten()

    ax.scatter(x, y)

    ax.set_xlabel(r"$\theta_{{{0},\textrm{{true}}}}$".format(m))
    ax.set_ylabel(r"$\theta_{{{0},\textrm{{opt}}}}$".format(m))
    
    common_limits(ax, plot_one_to_one=True)

for ax in axes[M:]:
    ax.set_visible(False)

fig.tight_layout()



