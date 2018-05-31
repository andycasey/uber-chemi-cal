from matplotlib import rc
rc("font", family="serif", size=12)
rc("text", usetex=True)

import daft

if __name__ == "__main__":
    pgm = daft.PGM([5, 4], origin=[-2.5, -2.10])
    pgm.add_node(daft.Node("god", r"", -2, 0, fixed=True))
    pgm.add_node(daft.Node("obs", r"$\mathcal{\ell}_{nm}$", 0, 0, observed=True))
    pgm.add_node(daft.Node("true", r"$\tilde{X}_n$", -1, 0))
    pgm.add_node(daft.Node("theta", r"$\theta_m$", 1, 0))
    pgm.add_node(daft.Node("sigma", r"$\sigma_m$", 1, -1))
    pgm.add_node(daft.Node("reg_theta", r"", 2, 0, fixed=True))
    pgm.add_node(daft.Node("reg_sigma", r"", 2, -1, fixed=True))
    pgm.add_plate(daft.Plate([-1.5, -1.00, 2, 2.25], label=r"stars $n$"))
    pgm.add_plate(daft.Plate([-0.5, -1.50, 2, 2.25], label=r"surveys $m$"))
    pgm.add_edge("theta", "obs")
    pgm.add_edge("true", "obs")
    pgm.add_edge("god", "true")
    pgm.add_edge("reg_theta", "theta")
    pgm.add_edge("reg_sigma", "sigma")
    pgm.add_edge("sigma", "obs")
    pgm.render()
    pgm.figure.savefig("pgm.pdf")
