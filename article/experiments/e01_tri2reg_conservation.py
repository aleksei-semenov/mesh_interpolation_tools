"""tri2reg: source mesh vs regular grid, and integral conservation across grid resolutions."""

from article.common import (
    DOUBLE_COL, N_PROCESSES, field_on_nodes, mesh_integral, np, plt,
    random_delaunay_mesh, save_data, save_figure,
)
from mesh_interpolation_tools.tri2reg import (
    apply_interpolation_weights,
    calculate_interpolation_weights,
)

EXPERIMENT = __name__.rsplit(".", 1)[-1]


def interpolate(nodes, elem, field, n_cells):
    axis = np.linspace(0.0, 1.0, n_cells + 1)
    weights, gx, gy, area_map = calculate_interpolation_weights(
        nodes, elem, axis, axis, n_processes=N_PROCESSES)
    H = apply_interpolation_weights(weights, area_map, elem.T, field,
                                    n_processes=N_PROCESSES)
    return gx, gy, H, area_map


def main():
    params = dict(n_nodes=800, field="wave", seed=0,
                  resolutions=[8, 16, 32, 64, 128], example_resolution=32)

    nodes, elem = random_delaunay_mesh(params["n_nodes"], seed=params["seed"])
    field = field_on_nodes(nodes, params["field"])
    ref = mesh_integral(nodes, elem, field)

    # --- conservation vs. resolution
    rel_err = []
    for n in params["resolutions"]:
        _, _, H, area_map = interpolate(nodes, elem, field, n)
        grid_integral = float(np.sum(H * area_map))
        rel_err.append(abs(grid_integral - ref) / abs(ref))
        print(f"  {n:4d}x{n:<4d} relative integral error = {rel_err[-1]:.2e}")
    save_data(EXPERIMENT, "conservation", resolutions=params["resolutions"],
              rel_err=rel_err, reference=ref)

    # --- figure
    gx, gy, H, _ = interpolate(nodes, elem, field, params["example_resolution"])
    vmin, vmax = field.min(), field.max()

    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, DOUBLE_COL / 3.2),
                             constrained_layout=True)
    elem_vals = field[elem].mean(axis=0)
    axes[0].tripcolor(nodes[0], nodes[1], elem.T, facecolors=elem_vals,
                      vmin=vmin, vmax=vmax, edgecolors="k", linewidth=0.1)
    axes[0].set_title("(a) source mesh")
    m = axes[1].pcolormesh(gx, gy, H, vmin=vmin, vmax=vmax, shading="flat")
    axes[1].set_title(f"(b) tri2reg, {params['example_resolution']}² grid")
    for ax in axes[:2]:
        ax.set_aspect("equal")
        ax.set_axis_off()
    fig.colorbar(m, ax=axes[:2], shrink=0.8)

    axes[2].loglog(params["resolutions"], np.maximum(rel_err, 1e-17), "o-")
    axes[2].set_xlabel("grid cells per side")
    axes[2].set_ylabel("relative integral error")
    axes[2].set_title("(c) conservation")

    save_figure(fig, EXPERIMENT, "tri2reg_overview", params)


if __name__ == "__main__":
    main()
