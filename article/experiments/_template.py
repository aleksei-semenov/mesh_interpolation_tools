"""One-line description shown by `python -m article list`.

Copy this file to ``eNN_your_name.py`` (the ``eNN_`` prefix makes it discoverable).
"""

from article.common import (
    DOUBLE_COL, N_PROCESSES, field_on_nodes, mit, np, plt,
    random_delaunay_mesh, save_figure,
)

EXPERIMENT = __name__.rsplit(".", 1)[-1]


def main():
    params = dict(n_nodes=500, field="wave", seed=0)

    nodes, elem = random_delaunay_mesh(params["n_nodes"], seed=params["seed"])
    field = field_on_nodes(nodes, params["field"])

    # --- call the real package code here ---
    gx, gy, H = mit.tri2reg(nodes, elem, field, np.linspace(0, 1, 51),
                            np.linspace(0, 1, 51), n_processes=N_PROCESSES)

    fig, ax = plt.subplots(figsize=(DOUBLE_COL / 2, DOUBLE_COL / 2))
    ax.pcolormesh(gx, gy, H, shading="flat")
    save_figure(fig, EXPERIMENT, "figure_name", params)


if __name__ == "__main__":
    main()
