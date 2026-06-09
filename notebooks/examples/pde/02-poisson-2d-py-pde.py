# %% [markdown]
# # Poisson 2D com `py-pde`
#
# Agora repetimos a ideia da aula anterior em duas dimensoes. O foco continua em
# usar os recursos da `py-pde`: grades cartesianas, `ScalarField`, equações em
# forma de expressão, armazenamento de trajetórias e visualização.
#
# O problema estacionário de referência é
#
# $$
# \Delta u = f(x,y),
# \qquad (x,y)\in(0,1)\times(0,1),
# $$
#
# com $u=0$ na fronteira. A solução exata escolhida é
#
# $$
# u_\mathrm{exata}(x,y)=\sin(\pi x)\sin(\pi y),
# $$
#
# logo
#
# $$
# f(x,y)=-2\pi^2\sin(\pi x)\sin(\pi y).
# $$

# %% [markdown]
# ## Importando as dependências

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pde

plt.rcParams.update(
    {
        "axes.grid": True,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "figure.figsize": (8, 4.5),
        "font.size": 11,
    }
)


# %% [markdown]
# ## Equação e campos

# %%
def coordinates_2d(grid):
    positions_x, positions_y = grid.axes_coords
    return np.meshgrid(positions_x, positions_y, indexing="ij")


def exact_stationary_2d(grid):
    coordinates_x, coordinates_y = coordinates_2d(grid)
    return np.sin(np.pi * coordinates_x) * np.sin(np.pi * coordinates_y)


def exact_transient_2d(grid, final_time):
    return (1.0 - np.exp(-2.0 * (np.pi**2) * final_time)) * exact_stationary_2d(grid)


def source_field_2d(grid):
    data = -2.0 * (np.pi**2) * exact_stationary_2d(grid)
    return pde.ScalarField(grid, data=data, label="f(x,y)")


def poisson_relaxation_2d(grid):
    source = source_field_2d(grid)
    equation = pde.PDE(
        {"u": "laplace(u) - source"},
        consts={"source": source},
        bc={"value": 0.0},
    )
    return equation, source


def plot_field_image(
    axis,
    grid,
    data,
    title,
    cmap="viridis",
    colorbar_label=None,
    vmin=None,
    vmax=None,
):
    positions_x, positions_y = grid.axes_coords
    image = axis.imshow(
        data.T,
        origin="lower",
        extent=[positions_x[0], positions_x[-1], positions_y[0], positions_y[-1]],
        aspect="equal",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_title(title)
    plt.colorbar(image, ax=axis, label=colorbar_label)


# %% [markdown]
# ## Estado estacionário por relaxação

# %%
grid_2d = pde.CartesianGrid([[0.0, 1.0], [0.0, 1.0]], [64, 64], periodic=False)
equation_2d, source_2d = poisson_relaxation_2d(grid_2d)
initial_2d = pde.ScalarField(grid_2d, data=np.zeros(grid_2d.shape), label="u")

stationary_time_2d = 0.5
solution_2d = equation_2d.solve(
    initial_2d,
    t_range=stationary_time_2d,
    dt=1e-3,
    solver="scipy",
    tracker=None,
    backend="numpy",
)
exact_2d = exact_stationary_2d(grid_2d)
error_2d = solution_2d.data - exact_2d

pd.DataFrame(
    {
        "quantidade": ["Nx", "Ny", "hx", "hy", "erro máximo"],
        "valor": [
            grid_2d.shape[0],
            grid_2d.shape[1],
            float(grid_2d.discretization[0]),
            float(grid_2d.discretization[1]),
            float(np.max(np.abs(error_2d))),
        ],
    }
)

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

plot_field_image(
    axes[0],
    grid_2d,
    exact_2d,
    "Solução exata",
    colorbar_label="u",
    vmin=0.0,
    vmax=1.0,
)
plot_field_image(
    axes[1],
    grid_2d,
    solution_2d.data,
    "Relaxação com py-pde",
    colorbar_label="u",
    vmin=0.0,
    vmax=1.0,
)
plot_field_image(
    axes[2],
    grid_2d,
    error_2d,
    "Erro",
    cmap="coolwarm",
    colorbar_label="erro",
)

fig.tight_layout()
plt.show()

# %% [markdown]
# ## Evolução transiente
#
# Nos mapas abaixo, todos os instantes usam a mesma barra de cor, de 0 a 1.
# Isso é essencial para comparar amplitudes ao longo do tempo.

# %%
transient_grid_2d = pde.CartesianGrid(
    [[0.0, 1.0], [0.0, 1.0]],
    [48, 48],
    periodic=False,
)
transient_equation_2d, _ = poisson_relaxation_2d(transient_grid_2d)
transient_initial_2d = pde.ScalarField(
    transient_grid_2d,
    data=np.zeros(transient_grid_2d.shape),
    label="u",
)

storage_2d = pde.MemoryStorage()
transient_final_time_2d = 0.15
transient_solution_2d = transient_equation_2d.solve(
    transient_initial_2d,
    t_range=transient_final_time_2d,
    dt=1e-3,
    solver="scipy",
    tracker=storage_2d.tracker(0.025),
    backend="numpy",
)

transient_times_2d = np.array(list(storage_2d.times))
transient_data_2d = np.array([field.data for _, field in storage_2d.items()])

np.max(
    np.abs(
        transient_solution_2d.data
        - exact_transient_2d(transient_grid_2d, transient_final_time_2d)
    )
)

# %%
snapshot_indices = [0, 1, 2, 4, -1]
fig, axes = plt.subplots(1, len(snapshot_indices), figsize=(17, 3.8))

for axis, snapshot_index in zip(axes, snapshot_indices):
    plot_field_image(
        axis,
        transient_grid_2d,
        transient_data_2d[snapshot_index],
        f"t = {transient_times_2d[snapshot_index]:.3f}",
        colorbar_label="u",
        vmin=0.0,
        vmax=1.0,
    )

fig.tight_layout()
plt.show()

# %% [markdown]
# ## Cortes e mapa espaço-tempo

# %%
positions_x_2d, positions_y_2d = transient_grid_2d.axes_coords
center_y_index = int(np.argmin(np.abs(positions_y_2d - 0.5)))
center_y_value = positions_y_2d[center_y_index]
centerline_data_2d = transient_data_2d[:, :, center_y_index]

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

for snapshot_index in snapshot_indices:
    axes[0].plot(
        positions_x_2d,
        centerline_data_2d[snapshot_index],
        label=f"t = {transient_times_2d[snapshot_index]:.3f}",
    )

axes[0].set_xlabel("x")
axes[0].set_ylabel(rf"$u(x,{center_y_value:.2f},t)$")
axes[0].set_title("Corte horizontal")
axes[0].legend(fontsize=8)

image = axes[1].imshow(
    centerline_data_2d,
    extent=[
        positions_x_2d[0],
        positions_x_2d[-1],
        transient_times_2d[-1],
        transient_times_2d[0],
    ],
    aspect="auto",
    cmap="viridis",
    vmin=0.0,
    vmax=1.0,
)
axes[1].set_xlabel("x")
axes[1].set_ylabel("t")
axes[1].set_title("Mapa espaço-tempo no corte central")
fig.colorbar(image, ax=axes[1], label="u")

fig.tight_layout()
plt.show()


# %% [markdown]
# ## Checagem de convergencia no transiente

# %%
def estimate_orders(step_sizes, errors):
    step_sizes = np.asarray(step_sizes, dtype=float)
    errors = np.asarray(errors, dtype=float)
    return np.log(errors[:-1] / errors[1:]) / np.log(step_sizes[:-1] / step_sizes[1:])


transient_rows_2d = []
final_time_2d = 0.05

for num_cells in [12, 16, 24, 32]:
    grid = pde.CartesianGrid(
        [[0.0, 1.0], [0.0, 1.0]],
        [num_cells, num_cells],
        periodic=False,
    )
    equation, _ = poisson_relaxation_2d(grid)
    initial_state = pde.ScalarField(grid, data=np.zeros(grid.shape))
    numerical = equation.solve(
        initial_state,
        t_range=final_time_2d,
        dt=1e-3,
        solver="scipy",
        tracker=None,
        backend="numpy",
    ).data
    exact = exact_transient_2d(grid, final_time_2d)
    error = numerical - exact
    step_size = float(grid.discretization[0])

    transient_rows_2d.append(
        {
            "N": num_cells,
            "h": step_size,
            "erro máximo": np.max(np.abs(error)),
            "erro L2 discreto": np.sqrt((step_size**2) * np.sum(error**2)),
        }
    )

transient_convergence_2d = pd.DataFrame(transient_rows_2d)
transient_convergence_2d

# %%
orders_max_2d = estimate_orders(
    transient_convergence_2d["h"],
    transient_convergence_2d["erro máximo"],
)
orders_l2_2d = estimate_orders(
    transient_convergence_2d["h"],
    transient_convergence_2d["erro L2 discreto"],
)

pd.DataFrame(
    {
        "N fino": transient_convergence_2d["N"].iloc[1:].to_numpy(),
        "ordem pelo erro máximo": orders_max_2d,
        "ordem pelo erro L2": orders_l2_2d,
    }
)
