# %% [markdown]
# # Introducao a PDEs com `py-pde`
#
# Nesta primeira aula, vamos usar a interface de alto nivel da `py-pde`, baseada
# em grades, campos e equacoes escritas como expressoes. A ideia e resolver uma
# relaxacao transiente cujo estado estacionario satisfaz o problema de Poisson
# em 1D.
#
# O problema estacionario de referencia e
#
# $$
# u_{xx} = f(x), \qquad x\in(0,1),
# $$
#
# com condicoes de Dirichlet homogeneas:
#
# $$
# u(0)=0, \qquad u(1)=0.
# $$
#
# Escolhemos
#
# $$
# u_\mathrm{exata}(x)=\sin(\pi x),
# \qquad
# f(x)=-\pi^2\sin(\pi x).
# $$
#
# Em vez de montar matrizes manualmente, escrevemos a relaxacao
#
# $$
# u_t = u_{xx} - f(x).
# $$
#
# Quando $t$ cresce, essa equacao converge para o estado estacionario de
# Poisson.

# %% [markdown]
# ## Importando as dependencias

# %%
from time import perf_counter

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
# ## Funcoes do problema
#
# A `py-pde` permite passar campos como constantes da expressao. Esse e o mesmo
# padrao usado em exemplos de PDEs heterogeneas: construimos um `ScalarField`
# para a fonte e passamos esse campo em `consts`.

# %%
def exact_stationary_1d(positions):
    return np.sin(np.pi * positions)


def exact_transient_1d(positions, final_time):
    return (1.0 - np.exp(-(np.pi**2) * final_time)) * exact_stationary_1d(positions)


def source_field_1d(grid):
    positions = grid.axes_coords[0]
    data = -(np.pi**2) * np.sin(np.pi * positions)
    return pde.ScalarField(grid, data=data, label="f(x)")


def poisson_relaxation_1d(grid):
    source = source_field_1d(grid)
    equation = pde.PDE(
        {"u": "laplace(u) - source"},
        consts={"source": source},
        bc={"value": 0.0},
    )
    return equation, source


# %% [markdown]
# ## Grade, campo e relaxacao ate o estado estacionario

# %%
grid_1d = pde.CartesianGrid([[0.0, 1.0]], 64, periodic=False)
positions_1d = grid_1d.axes_coords[0]
equation_1d, source_1d = poisson_relaxation_1d(grid_1d)

initial_1d = pde.ScalarField(
    grid_1d,
    data=np.zeros(grid_1d.shape),
    label="u(x,t)",
)
stationary_time_1d = 1.5
solution_1d = equation_1d.solve(
    initial_1d,
    t_range=stationary_time_1d,
    dt=1e-3,
    solver="scipy",
    tracker=None,
    backend="numpy",
)
exact_1d = exact_stationary_1d(positions_1d)

pd.DataFrame(
    {
        "quantidade": ["numero de celulas", "h", "tempo de relaxacao", "erro maximo"],
        "valor": [
            grid_1d.shape[0],
            float(grid_1d.discretization[0]),
            stationary_time_1d,
            float(np.max(np.abs(solution_1d.data - exact_1d))),
        ],
    }
)

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

axes[0].plot(positions_1d, exact_1d, color="black", linewidth=2.0, label="exata")
axes[0].plot(positions_1d, solution_1d.data, "--", label="py-pde")
axes[0].set_xlabel("x")
axes[0].set_ylabel("u(x)")
axes[0].set_title("Estado estacionario por relaxacao")
axes[0].legend()

axes[1].plot(positions_1d, source_1d.data, color="tab:orange")
axes[1].set_xlabel("x")
axes[1].set_ylabel("f(x)")
axes[1].set_title("Fonte como ScalarField")

fig.tight_layout()
plt.show()

# %% [markdown]
# ## Evolucao transiente
#
# Agora guardamos varios instantes da relaxacao. Em 1D, a comparacao temporal
# pode ser feita por curvas e por um mapa espaco-tempo.

# %%
transient_grid_1d = pde.CartesianGrid([[0.0, 1.0]], 64, periodic=False)
transient_positions_1d = transient_grid_1d.axes_coords[0]
transient_equation_1d, _ = poisson_relaxation_1d(transient_grid_1d)
transient_initial_1d = pde.ScalarField(
    transient_grid_1d,
    data=np.zeros(transient_grid_1d.shape),
    label="u(x,t)",
)

storage_1d = pde.MemoryStorage()
transient_final_time_1d = 0.5
transient_solution_1d = transient_equation_1d.solve(
    transient_initial_1d,
    t_range=transient_final_time_1d,
    dt=1e-3,
    solver="scipy",
    tracker=storage_1d.tracker(0.05),
    backend="numpy",
)

transient_times_1d = np.array(list(storage_1d.times))
transient_data_1d = np.array([field.data for _, field in storage_1d.items()])

np.max(
    np.abs(
        transient_solution_1d.data
        - exact_transient_1d(transient_positions_1d, transient_final_time_1d)
    )
)

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

for storage_index in [0, 1, 2, 4, 6, -1]:
    axes[0].plot(
        transient_positions_1d,
        transient_data_1d[storage_index],
        label=f"t = {transient_times_1d[storage_index]:.2f}",
    )

axes[0].plot(
    transient_positions_1d,
    exact_stationary_1d(transient_positions_1d),
    color="black",
    linewidth=2.0,
    label="estacionaria",
)
axes[0].set_xlabel("x")
axes[0].set_ylabel("u(x,t)")
axes[0].set_title("Perfis ao longo do tempo")
axes[0].legend(fontsize=8)

image = axes[1].imshow(
    transient_data_1d,
    extent=[
        transient_positions_1d[0],
        transient_positions_1d[-1],
        transient_times_1d[-1],
        transient_times_1d[0],
    ],
    aspect="auto",
    cmap="viridis",
    vmin=0.0,
    vmax=1.0,
)
axes[1].set_xlabel("x")
axes[1].set_ylabel("t")
axes[1].set_title("Mapa espaco-tempo")
fig.colorbar(image, ax=axes[1], label="u")

fig.tight_layout()
plt.show()


# %% [markdown]
# ## Convergencia espacial
#
# Repetimos a mesma equacao em varias malhas e comparamos a solucao em um tempo
# final fixo. O erro temporal e controlado pelo solver adaptativo do SciPy, e o
# erro observado deve ser dominado pela discretizacao espacial.

# %%
def estimate_orders(step_sizes, errors):
    step_sizes = np.asarray(step_sizes, dtype=float)
    errors = np.asarray(errors, dtype=float)
    return np.log(errors[:-1] / errors[1:]) / np.log(step_sizes[:-1] / step_sizes[1:])


def convergence_triangle(axis, base_x, base_y, order, width_factor=1.8, color="0.25"):
    left_x = base_x / width_factor
    right_x = base_x
    lower_y = base_y
    upper_y = base_y * width_factor**order

    axis.plot([left_x, right_x], [lower_y, lower_y], color=color, linewidth=1.2)
    axis.plot([right_x, right_x], [lower_y, upper_y], color=color, linewidth=1.2)
    axis.plot([left_x, right_x], [lower_y, upper_y], color=color, linewidth=1.2)
    axis.text(
        np.sqrt(left_x * right_x),
        upper_y * 1.08,
        rf"$O(h^{order:g})$",
        ha="center",
        va="bottom",
        color=color,
    )


final_time_1d = 0.2
convergence_rows_1d = []

for num_cells in [16, 32, 64, 128]:
    grid = pde.CartesianGrid([[0.0, 1.0]], num_cells, periodic=False)
    positions = grid.axes_coords[0]
    equation, _ = poisson_relaxation_1d(grid)
    initial_state = pde.ScalarField(grid, data=np.zeros(grid.shape))
    numerical = equation.solve(
        initial_state,
        t_range=final_time_1d,
        dt=1e-3,
        solver="scipy",
        tracker=None,
        backend="numpy",
    ).data
    exact = exact_transient_1d(positions, final_time_1d)
    error = numerical - exact
    step_size = float(grid.discretization[0])

    convergence_rows_1d.append(
        {
            "N": num_cells,
            "h": step_size,
            "erro maximo": np.max(np.abs(error)),
            "erro L2 discreto": np.sqrt(step_size * np.sum(error**2)),
        }
    )

convergence_1d = pd.DataFrame(convergence_rows_1d)
convergence_1d

# %%
fig, axis = plt.subplots()

axis.loglog(convergence_1d["h"], convergence_1d["erro maximo"], "o-", label="max")
axis.loglog(convergence_1d["h"], convergence_1d["erro L2 discreto"], "s-", label="L2")
convergence_triangle(axis, base_x=0.04, base_y=5e-5, order=2)

axis.set_xlabel("h")
axis.set_ylabel("erro")
axis.set_title("Convergencia espacial no tempo final")
axis.legend()
plt.show()


# %% [markdown]
# ## Comparando solvers temporais
#
# A comparacao abaixo usa a mesma equacao declarada com `PDE`, mudando apenas o
# solver temporal.

# %%
def run_transient_solver_1d(
    label, solver_name, time_step, num_cells=32, final_time=0.2
):
    grid = pde.CartesianGrid([[0.0, 1.0]], num_cells, periodic=False)
    positions = grid.axes_coords[0]
    equation, _ = poisson_relaxation_1d(grid)
    initial_state = pde.ScalarField(grid, data=np.zeros(grid.shape))

    tic = perf_counter()
    solution = equation.solve(
        initial_state,
        t_range=final_time,
        dt=time_step,
        solver=solver_name,
        tracker=None,
        backend="numpy",
    )
    elapsed_ms = 1e3 * (perf_counter() - tic)

    exact = exact_transient_1d(positions, final_time)
    error = np.max(np.abs(solution.data - exact))

    return {
        "metodo": label,
        "dt usado": time_step,
        "erro maximo": error,
        "tempo (ms)": elapsed_ms,
    }


solver_comparison_1d = pd.DataFrame(
    [
        run_transient_solver_1d("Euler explicito", "explicit", 1e-4),
        run_transient_solver_1d("Euler implicito", "implicit", 2e-4),
        run_transient_solver_1d("SciPy", "scipy", 1e-3),
    ]
)
solver_comparison_1d

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
positions = np.arange(len(solver_comparison_1d))

axes[0].bar(positions, solver_comparison_1d["erro maximo"], color="tab:blue")
axes[0].set_yscale("log")
axes[0].set_ylabel("erro maximo")
axes[0].set_title("Erro no tempo final")

axes[1].bar(positions, solver_comparison_1d["tempo (ms)"], color="tab:purple")
axes[1].set_ylabel("tempo (ms)")
axes[1].set_title("Tempo de execucao")

for axis in axes:
    axis.set_xticks(positions)
    axis.set_xticklabels(solver_comparison_1d["metodo"], rotation=25, ha="right")

fig.tight_layout()
plt.show()
