# %% [markdown]
# # Darcy 2D com obstáculo usando recursos da `py-pde`
#
# Este notebook é uma alternativa à aula 03. A geometria e os parâmetros físicos
# seguem a mesma configuração:
#
# * domínio $50\,\mathrm{m}\times 50\,\mathrm{m}$;
# * obstáculo quadrado central com 15% da área do domínio;
# * $k_m=10^{-13}\,\mathrm{m^2}$ na matriz;
# * $k_o=10^{-18}\,\mathrm{m^2}$ no obstáculo;
# * $\Delta P=100\,\mathrm{MPa}$ entre esquerda e direita.
#
# Aqui a pressão é obtida por uma relaxação transiente escrita diretamente como
# uma `pde.PDE`. Trabalhamos com a pressão adimensional
#
# $$
# P=\frac{p-p_R}{p_L-p_R},
# $$
#
# e resolvemos
#
# $$
# P_t =
# k_\mathrm{rel}\Delta P
# +
# \nabla k_\mathrm{rel}\cdot\nabla P.
# $$
#
# Essa é a forma expandida de $\nabla\cdot(k_\mathrm{rel}\nabla P)$.

# %% [markdown]
# ## Importando as dependências

# %%
from matplotlib.patches import Rectangle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pde
import pde.fields.base as pde_field_base
import pde.fields.collection as pde_field_collection
import pde.tools.expressions as pde_expressions
from pde.tools.misc import get_common_dtype

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
# ## Compatibilidade local

# %%
def patch_number_array_for_numpy2(module):
    original_number_array = module.number_array

    def number_array_numpy2_compatible(data, dtype=None, copy=True):
        try:
            return original_number_array(data, dtype=dtype, copy=copy)
        except ValueError as error:
            if copy is False and "Unable to avoid copy" in str(error):
                target_dtype = (
                    get_common_dtype(data) if dtype is None else np.dtype(dtype)
                )
                return np.asarray(data, dtype=target_dtype)
            raise

    module.number_array = number_array_numpy2_compatible


for module in [pde_field_base, pde_field_collection, pde_expressions]:
    patch_number_array_for_numpy2(module)


# %% [markdown]
# ## Problema de Darcy

# %%
class DarcyPDEProblem:
    def __init__(
        self,
        length=50.0,
        shape=(80, 80),
        obstacle_area_fraction=0.15,
        interface_width=None,
        permeability_matrix=1e-13,
        permeability_obstacle=1e-18,
        viscosity=1e-3,
        pressure_left=100e6,
        pressure_right=0.0,
    ):
        self.length = float(length)
        self.shape = tuple(shape)
        self.obstacle_area_fraction = float(obstacle_area_fraction)
        self.obstacle_side = self.length * np.sqrt(self.obstacle_area_fraction)
        self.permeability_matrix = float(permeability_matrix)
        self.permeability_obstacle = float(permeability_obstacle)
        self.viscosity = float(viscosity)
        self.pressure_left = float(pressure_left)
        self.pressure_right = float(pressure_right)
        self.grid = pde.CartesianGrid(
            [[0.0, self.length], [0.0, self.length]],
            self.shape,
            periodic=False,
        )
        self.interface_width = (
            1.5 * float(np.min(self.grid.discretization))
            if interface_width is None
            else float(interface_width)
        )

    def coordinates(self):
        positions_x, positions_y = self.grid.axes_coords
        return np.meshgrid(positions_x, positions_y, indexing="ij")

    def obstacle_bounds(self):
        start = 0.5 * (self.length - self.obstacle_side)
        end = 0.5 * (self.length + self.obstacle_side)
        return start, end

    def obstacle_indicator(self):
        coordinates_x, coordinates_y = self.coordinates()
        distance = (
            np.maximum(
                np.abs(coordinates_x - 0.5 * self.length),
                np.abs(coordinates_y - 0.5 * self.length),
            )
            - 0.5 * self.obstacle_side
        )
        return 0.5 * (1.0 - np.tanh(distance / self.interface_width))

    def permeability_relative_field(self):
        ratio = self.permeability_obstacle / self.permeability_matrix
        data = np.exp(self.obstacle_indicator() * np.log(ratio))
        return pde.ScalarField(self.grid, data=data, label=r"$k/k_m$")

    def permeability_field(self):
        data = self.permeability_matrix * self.permeability_relative_field().data
        return pde.ScalarField(self.grid, data=data, label="k")

    def pressure_boundary_condition(self):
        return [
            {"low": {"value": 1.0}, "high": {"value": 0.0}},
            {"derivative": 0.0},
        ]

    def pressure_equation(self):
        permeability_relative = self.permeability_relative_field()
        return pde.PDE(
            {"P": ("k_rel * laplace(P) " "+ dot(gradient(k_rel), gradient(P))")},
            consts={"k_rel": permeability_relative},
            bc=self.pressure_boundary_condition(),
        )

    def initial_pressure(self):
        positions_x = self.grid.axes_coords[0]
        data = (1.0 - positions_x[:, None] / self.length) * np.ones(self.shape)
        return pde.ScalarField(self.grid, data=data, label="P")

    def solve_pressure(self, t_range=250.0, dt=0.04):
        return self.pressure_equation().solve(
            self.initial_pressure(),
            t_range=t_range,
            dt=dt,
            solver="explicit",
            tracker=None,
            backend="numpy",
        )

    def dimensional_pressure(self, pressure_dimensionless):
        delta_pressure = self.pressure_left - self.pressure_right
        data = self.pressure_right + delta_pressure * pressure_dimensionless.data
        return pde.ScalarField(self.grid, data=data, label="p [Pa]")

    def darcy_velocity(self, pressure_dimensionless):
        pressure_gradient = pressure_dimensionless.gradient(
            bc=self.pressure_boundary_condition()
        )
        scale = (
            self.permeability_matrix
            * (self.pressure_left - self.pressure_right)
            / self.viscosity
        )
        velocity = -scale * self.permeability_relative_field() * pressure_gradient
        velocity.label = "q"
        return velocity


darcy = DarcyPDEProblem()
permeability = darcy.permeability_field()
pressure_dimensionless = darcy.solve_pressure()
pressure = darcy.dimensional_pressure(pressure_dimensionless)
velocity = darcy.darcy_velocity(pressure_dimensionless)
speed = np.sqrt(velocity.data[0] ** 2 + velocity.data[1] ** 2)

# %% [markdown]
# ## Campos resultantes

# %%
pd.DataFrame(
    {
        "quantidade": [
            "Lx = Ly",
            "Nx",
            "Ny",
            "Delta P",
            "k matriz",
            "k obstáculo",
            "lado do obstáculo",
            "fração de área do obstáculo",
            "largura de interface",
            "max |q|",
        ],
        "valor": [
            f"{darcy.length:g} m",
            darcy.shape[0],
            darcy.shape[1],
            f"{(darcy.pressure_left - darcy.pressure_right) / 1e6:g} MPa",
            f"{darcy.permeability_matrix:.1e} m^2",
            f"{darcy.permeability_obstacle:.1e} m^2",
            f"{darcy.obstacle_side:.2f} m",
            f"{100 * darcy.obstacle_area_fraction:.1f}%",
            f"{darcy.interface_width:.2f} m",
            f"{np.max(speed):.3e} m/s",
        ],
    }
)


# %%
def plot_field_image(axis, grid, data, title, cmap="viridis", colorbar_label=None):
    image = axis.imshow(
        data.T,
        origin="lower",
        extent=[0.0, grid.axes_bounds[0][1], 0.0, grid.axes_bounds[1][1]],
        aspect="equal",
        cmap=cmap,
    )
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.set_title(title)
    plt.colorbar(image, ax=axis, label=colorbar_label)


def draw_obstacle(axis, model, color="white"):
    start, end = model.obstacle_bounds()
    axis.add_patch(
        Rectangle(
            (start, start),
            end - start,
            end - start,
            fill=False,
            edgecolor=color,
            linewidth=1.6,
        )
    )


fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

plot_field_image(
    axes[0],
    darcy.grid,
    np.log10(permeability.data),
    r"$\log_{10}(k)$",
    cmap="magma",
    colorbar_label=r"$\log_{10}(k/\mathrm{m^2})$",
)
plot_field_image(
    axes[1],
    darcy.grid,
    pressure.data / 1e6,
    "Pressão",
    cmap="viridis",
    colorbar_label="MPa",
)
plot_field_image(
    axes[2],
    darcy.grid,
    speed,
    "Modulo da velocidade de Darcy",
    cmap="cividis",
    colorbar_label="m/s",
)

for axis in axes:
    draw_obstacle(axis, darcy)

fig.tight_layout()
plt.show()

# %% [markdown]
# ## Linhas de corrente e cortes

# %%
positions_x, positions_y = darcy.grid.axes_coords
center_y_index = int(np.argmin(np.abs(positions_y - 0.5 * darcy.length)))
center_x_index = int(np.argmin(np.abs(positions_x - 0.5 * darcy.length)))

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

plot_field_image(
    axes[0],
    darcy.grid,
    pressure.data / 1e6,
    "Pressão e linhas de corrente",
    cmap="viridis",
    colorbar_label="MPa",
)
axes[0].streamplot(
    positions_x,
    positions_y,
    velocity.data[0].T,
    velocity.data[1].T,
    color="white",
    density=1.2,
    linewidth=0.8,
)
draw_obstacle(axes[0], darcy)

axes[1].plot(
    positions_x,
    pressure.data[:, center_y_index] / 1e6,
    label=rf"$p(x,{positions_y[center_y_index]:.1f})$",
)
axes[1].plot(
    positions_y,
    pressure.data[center_x_index, :] / 1e6,
    label=rf"$p({positions_x[center_x_index]:.1f},y)$",
)
axes[1].set_xlabel("coordenada [m]")
axes[1].set_ylabel("pressão [MPa]")
axes[1].set_title("Cortes pelo centro do domínio")
axes[1].legend()

fig.tight_layout()
plt.show()
