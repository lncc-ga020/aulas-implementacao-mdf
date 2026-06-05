# %% [markdown]
# # Poisson 2D com `py-pde`
#
# Neste notebook, vamos repetir a estrutura da aula anterior, mas agora em duas dimensões.
#
# O objetivo é mostrar que a passagem de 1D para 2D muda principalmente a geometria da grade e a forma de visualização. A lógica continua a mesma:
#
# * definir uma classe customizada para o modelo;
# * criar a grade;
# * construir o termo fonte;
# * resolver o problema estacionário;
# * comparar com uma solução analítica;
# * resolver uma versão transiente;
# * visualizar o campo no domínio e em cortes unidimensionais.

# %% [markdown]
# ## Importando as dependências

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pde
from pde.solvers import ScipySolver

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
# ## Problema-modelo
#
# Vamos resolver
#
# $$
# \Delta u = f(x,y),
# \qquad (x,y)\in(0,1)\times(0,1),
# $$
#
# com condição de Dirichlet homogênea em toda a fronteira:
#
# $$
# u=0\quad\text{em }\partial\Omega.
# $$
#
# Escolhemos a solução exata
#
# $$
# u_{\mathrm{exata}}(x,y)=\sin(\pi x)\sin(\pi y).
# $$
#
# Como
#
# $$
# \Delta u_{\mathrm{exata}}
# =
# -2\pi^2\sin(\pi x)\sin(\pi y),
# $$
#
# temos
#
# $$
# f(x,y)=-2\pi^2\sin(\pi x)\sin(\pi y).
# $$

# %%
class Poisson2D(pde.PDEBase):
    def __init__(self, boundary_condition=None):
        super().__init__()
        self.boundary_condition = (
            {"value": 0} if boundary_condition is None else boundary_condition
        )

    def coordinates(self, grid):
        positions_x, positions_y = grid.axes_coords
        return np.meshgrid(positions_x, positions_y, indexing="ij")

    def exact_stationary(self, grid):
        coordinates_x, coordinates_y = self.coordinates(grid)
        return np.sin(np.pi * coordinates_x) * np.sin(np.pi * coordinates_y)

    def exact_transient(self, grid, final_time):
        return (1.0 - np.exp(-2.0 * (np.pi**2) * final_time)) * self.exact_stationary(
            grid
        )

    def source_data(self, grid):
        return -2.0 * (np.pi**2) * self.exact_stationary(grid)

    def source_field(self, grid):
        return pde.ScalarField(grid, data=self.source_data(grid), label="f(x,y)")

    def solve_stationary(self, grid):
        return pde.solve_poisson_equation(
            self.source_field(grid),
            bc=self.boundary_condition,
            label="u numérico",
        )

    def evolution_rate(self, state, time=0):
        return state.laplace(self.boundary_condition) - self.source_field(state.grid)


poisson_2d = Poisson2D()

# %% [markdown]
# A versão transiente associada é
#
# $$
# u_t=\Delta u-f(x,y),
# \qquad u(x,y,0)=0.
# $$
#
# Para esse caso, a solução analítica é
#
# $$
# u(x,y,t)=
# \left(1-e^{-2\pi^2t}\right)\sin(\pi x)\sin(\pi y).
# $$

# %% [markdown]
# ## Solução estacionária em uma malha 2D
#
# Agora criamos uma grade cartesiana uniforme no quadrado unitário.

# %%
grid_2d = pde.CartesianGrid([[0.0, 1.0], [0.0, 1.0]], [64, 64], periodic=False)
solution_2d = poisson_2d.solve_stationary(grid_2d)
exact_2d = poisson_2d.exact_stationary(grid_2d)
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
def plot_field_image(axis, grid, data, title, cmap="viridis", colorbar_label=None):
    positions_x, positions_y = grid.axes_coords
    image = axis.imshow(
        data.T,
        origin="lower",
        extent=[positions_x[0], positions_x[-1], positions_y[0], positions_y[-1]],
        aspect="equal",
        cmap=cmap,
    )
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_title(title)
    plt.colorbar(image, ax=axis, label=colorbar_label)


fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

plot_field_image(axes[0], grid_2d, exact_2d, "Solução exata", colorbar_label="u")
plot_field_image(
    axes[1], grid_2d, solution_2d.data, "Solução numérica", colorbar_label="u"
)
plot_field_image(
    axes[2], grid_2d, error_2d, "Erro", cmap="coolwarm", colorbar_label="erro"
)

fig.tight_layout()
plt.show()


# %% [markdown]
# Em duas dimensões, a solução passa a ser um campo no domínio. Por isso, mapas de cor geralmente são mais informativos do que apenas curvas.

# %% [markdown]
# ## Convergência espacial do problema estacionário
#
# Repetimos a solução estacionária para várias malhas. Como o operador de Laplace discreto é de segunda ordem, esperamos observar erro proporcional a $h^2$.

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


stationary_rows_2d = []

for num_cells in [16, 24, 32, 48, 64]:
    grid = pde.CartesianGrid(
        [[0.0, 1.0], [0.0, 1.0]],
        [num_cells, num_cells],
        periodic=False,
    )
    numerical = poisson_2d.solve_stationary(grid).data
    exact = poisson_2d.exact_stationary(grid)
    error = numerical - exact
    step_size = float(grid.discretization[0])

    stationary_rows_2d.append(
        {
            "N": num_cells,
            "h": step_size,
            "erro máximo": np.max(np.abs(error)),
            "erro L2 discreto": np.sqrt((step_size**2) * np.sum(error**2)),
        }
    )

stationary_convergence_2d = pd.DataFrame(stationary_rows_2d)
stationary_convergence_2d

# %%
orders_max_2d = estimate_orders(
    stationary_convergence_2d["h"],
    stationary_convergence_2d["erro máximo"],
)
orders_l2_2d = estimate_orders(
    stationary_convergence_2d["h"],
    stationary_convergence_2d["erro L2 discreto"],
)

pd.DataFrame(
    {
        "N fino": stationary_convergence_2d["N"].iloc[1:].to_numpy(),
        "ordem pelo erro máximo": orders_max_2d,
        "ordem pelo erro L2": orders_l2_2d,
    }
)

# %%
fig, axis = plt.subplots()

axis.loglog(
    stationary_convergence_2d["h"],
    stationary_convergence_2d["erro máximo"],
    "o-",
    label=r"$\|e_h\|_\infty$",
)
axis.loglog(
    stationary_convergence_2d["h"],
    stationary_convergence_2d["erro L2 discreto"],
    "s-",
    label=r"$\|e_h\|_{2,h}$",
)
convergence_triangle(axis, base_x=0.04, base_y=2.5e-5, order=2)

axis.set_xlabel("h")
axis.set_ylabel("erro")
axis.set_title("Convergência espacial: Poisson 2D estacionário")
axis.legend()
plt.show()

# %% [markdown]
# A taxa observada novamente fica próxima de 2. Isso é a extensão direta do que já apareceu em 1D.

# %% [markdown]
# ## Problema transiente
#
# Vamos agora resolver a relaxação
#
# $$
# u_t=\Delta u-f(x,y),
# \qquad u(x,y,0)=0.
# $$
#
# A solução começa nula e se aproxima do estado estacionário.

# %%
transient_grid_2d = pde.CartesianGrid(
    [[0.0, 1.0], [0.0, 1.0]],
    [48, 48],
    periodic=False,
)
initial_2d = pde.ScalarField(
    transient_grid_2d,
    data=np.zeros(transient_grid_2d.shape),
    label="u(x,y,t)",
)

transient_storage_2d = pde.MemoryStorage()
transient_solver_2d = ScipySolver(
    poisson_2d,
    backend="numpy",
    method="BDF",
    rtol=1e-6,
    atol=1e-8,
)
transient_controller_2d = pde.Controller(
    transient_solver_2d,
    t_range=0.15,
    tracker=transient_storage_2d.tracker(0.025),
)
transient_controller_2d.run(initial_2d, dt=1e-3)

transient_times_2d = np.array(list(transient_storage_2d.times))
transient_data_2d = np.array([field.data for _, field in transient_storage_2d.items()])

transient_controller_2d.diagnostics["solver"]

# %%
snapshot_indices = [0, 1, 2, 4, -1]
fig, axes = plt.subplots(1, len(snapshot_indices), figsize=(17, 3.8))

for axis, snapshot_index in zip(axes, snapshot_indices):
    current_time = transient_times_2d[snapshot_index]
    plot_field_image(
        axis,
        transient_grid_2d,
        transient_data_2d[snapshot_index],
        f"t = {current_time:.3f}",
        colorbar_label="u",
    )

fig.tight_layout()
plt.show()

# %% [markdown]
# Os mapas mostram o campo inteiro em alguns instantes. A forma espacial é a mesma da solução estacionária, mas a amplitude cresce com o tempo.

# %% [markdown]
# ## Cortes e mapa espaço-tempo
#
# Em 2D, uma forma útil de reduzir a visualização é olhar um corte. Aqui tomamos a reta horizontal que passa pelo meio do domínio, isto é, $y\approx 0.5$.

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
)
axes[1].set_xlabel("x")
axes[1].set_ylabel("t")
axes[1].set_title("Mapa espaço-tempo no corte central")
fig.colorbar(image, ax=axes[1], label="u")

fig.tight_layout()
plt.show()

# %% [markdown]
# O corte central transforma o problema 2D em uma curva 1D para cada instante. O mapa espaço-tempo empilha esses cortes, permitindo acompanhar como a solução evolui ao longo dessa reta.

# %% [markdown]
# ## Checagem de convergência no transiente
#
# Também podemos medir o erro no tempo final. Aqui usamos uma janela curta e um solver adaptativo. A taxa observada deve ficar próxima de segunda ordem quando o erro espacial domina.

# %%
transient_rows_2d = []
final_time_2d = 0.05

for num_cells in [12, 16, 24, 32]:
    grid = pde.CartesianGrid(
        [[0.0, 1.0], [0.0, 1.0]],
        [num_cells, num_cells],
        periodic=False,
    )
    initial_state = pde.ScalarField(grid, data=np.zeros(grid.shape))
    solver = ScipySolver(
        poisson_2d,
        backend="numpy",
        method="BDF",
        rtol=1e-6,
        atol=1e-8,
    )
    controller = pde.Controller(solver, t_range=final_time_2d, tracker=None)
    numerical = controller.run(initial_state, dt=1e-3).data
    exact = poisson_2d.exact_transient(grid, final_time_2d)
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
fig, axis = plt.subplots()

axis.loglog(
    transient_convergence_2d["h"],
    transient_convergence_2d["erro máximo"],
    "o-",
    label=r"$\|e_h\|_\infty$",
)
axis.loglog(
    transient_convergence_2d["h"],
    transient_convergence_2d["erro L2 discreto"],
    "s-",
    label=r"$\|e_h\|_{2,h}$",
)
convergence_triangle(axis, base_x=0.065, base_y=1.05e-4, order=2, width_factor=1.25)

axis.set_xlabel("h")
axis.set_ylabel("erro")
axis.set_title("Convergência no tempo final")
axis.legend()
plt.show()

# %% [markdown]
# Esta checagem mistura discretização espacial e temporal, mas as tolerâncias do solver foram escolhidas para deixar o erro espacial dominante. A taxa fica compatível com o comportamento de segunda ordem observado no caso estacionário.

# %% [markdown]
# ## Conclusão
#
# A estrutura da aula 1D se transporta quase diretamente para 2D:
#
# * a grade passa a ter duas direções;
# * o campo passa a ser uma matriz;
# * a visualização passa a depender de mapas de cor, cortes e mapas espaço-tempo;
# * a análise de convergência continua baseada na comparação com uma solução exata.
#
# Essa organização será útil para problemas com coeficientes variáveis, como o escoamento de Darcy heterogêneo.
