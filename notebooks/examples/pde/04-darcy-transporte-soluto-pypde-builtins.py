# %% [markdown]
# # Darcy + transporte usando `PDE` e `FieldCollection`
#
# Este notebook e uma alternativa a aula 04, usando mais diretamente a interface
# de alto nivel da `py-pde`.
#
# Mantemos a configuracao fisica:
#
# * dominio $50\,\mathrm{m}\times 50\,\mathrm{m}$;
# * obstaculo quadrado central com 15% da area;
# * $k_m=10^{-13}\,\mathrm{m^2}$;
# * $k_o=10^{-18}\,\mathrm{m^2}$;
# * $\Delta P=100\,\mathrm{MPa}$;
# * $\phi=0.2$;
# * $D=2\times 10^{-5}\,\mathrm{m^2/s}$;
# * tempo final $60000\,\mathrm{s}$.
#
# A pressao e resolvida por uma relaxacao declarada com `pde.PDE`. O transporte
# usa um estado com duas variaveis, `P` e `c`, em uma `FieldCollection`:
#
# $$
# P_t = 0,
# \qquad
# c_t = -\mathbf v\cdot\nabla c + D_\mathrm{efetivo}\Delta c,
# $$
#
# com $\mathbf v=\mathbf q/\phi$. O termo $D_\mathrm{efetivo}$ preserva o
# coeficiente fisico e acrescenta uma difusao numerica moderada para estabilizar
# a adveccao centrada usada pelos operadores built-in.

# %% [markdown]
# ## Importando as dependencias

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
                target_dtype = get_common_dtype(data) if dtype is None else np.dtype(dtype)
                return np.asarray(data, dtype=target_dtype)
            raise

    module.number_array = number_array_numpy2_compatible


for module in [pde_field_base, pde_field_collection, pde_expressions]:
    patch_number_array_for_numpy2(module)


# %% [markdown]
# ## Darcy com campos da `py-pde`

# %%
class DarcyPDEProblem:
    def __init__(
        self,
        length=50.0,
        shape=(60, 60),
        obstacle_area_fraction=0.15,
        interface_width=None,
        permeability_matrix=1e-13,
        permeability_obstacle=1e-18,
        viscosity=1e-3,
        porosity=0.2,
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
        self.porosity = float(porosity)
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

    def concentration_boundary_condition(self):
        return [
            {"low": {"value": 1.0}, "high": {"derivative": 0.0}},
            {"derivative": 0.0},
        ]

    def pressure_equation(self):
        permeability_relative = self.permeability_relative_field()
        return pde.PDE(
            {
                "P": (
                    "k_rel * laplace(P) "
                    "+ dot(gradient(k_rel), gradient(P))"
                )
            },
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
        darcy_velocity = -scale * self.permeability_relative_field() * pressure_gradient
        darcy_velocity.label = "q"
        return darcy_velocity

    def pore_velocity(self, pressure_dimensionless):
        velocity = self.darcy_velocity(pressure_dimensionless)
        velocity.data /= self.porosity
        velocity.label = "v"
        return velocity


darcy = DarcyPDEProblem()
pressure_dimensionless = darcy.solve_pressure()
pressure = darcy.dimensional_pressure(pressure_dimensionless)
pore_velocity = darcy.pore_velocity(pressure_dimensionless)
speed = np.sqrt(pore_velocity.data[0] ** 2 + pore_velocity.data[1] ** 2)

# %% [markdown]
# ## Sistema acoplado

# %%
physical_diffusion = 2e-5
step_x, step_y = map(float, darcy.grid.discretization)
numerical_diffusion = 0.25 * np.max(speed) * min(step_x, step_y)
effective_diffusion = physical_diffusion + numerical_diffusion

transport_equation = pde.PDE(
    {
        "P": "0",
        "c": "-dot(v, gradient(c)) + D_eff * laplace(c)",
    },
    consts={"v": pore_velocity, "D_eff": effective_diffusion},
    bc=darcy.concentration_boundary_condition(),
)

initial_concentration = pde.ScalarField(
    darcy.grid,
    data=np.zeros(darcy.shape),
    label="c",
)
initial_state = pde.FieldCollection(
    [pressure_dimensionless, initial_concentration],
    labels=["P", "c"],
)

cfl_time_step = 0.35 * min(step_x, step_y) / np.max(speed)
time_step = min(20.0, cfl_time_step)
final_time = 60_000.0

storage = pde.MemoryStorage()
solution = transport_equation.solve(
    initial_state,
    t_range=final_time,
    dt=time_step,
    solver="explicit",
    tracker=storage.tracker(final_time / 8.0),
    backend="numpy",
)

times = np.array(list(storage.times))
concentration_history = np.array([state[1].data for _, state in storage.items()])

pd.DataFrame(
    {
        "quantidade": [
            "lado do obstaculo",
            "fracao de area do obstaculo",
            "porosidade",
            "D fisico",
            "D numerico",
            "D usado",
            "max |v|",
            "dt CFL",
            "dt usado",
            "tempo final",
            "min c final",
            "max c final",
        ],
        "valor": [
            f"{darcy.obstacle_side:.2f} m",
            f"{100 * darcy.obstacle_area_fraction:.1f}%",
            darcy.porosity,
            f"{physical_diffusion:.2e} m^2/s",
            f"{numerical_diffusion:.2e} m^2/s",
            f"{effective_diffusion:.2e} m^2/s",
            f"{np.max(speed):.3e} m/s",
            f"{cfl_time_step:.1f} s",
            f"{time_step:.1f} s",
            f"{final_time / 3600:.1f} h",
            float(np.min(solution[1].data)),
            float(np.max(solution[1].data)),
        ],
    }
)

# %% [markdown]
# ## Visualizacao

# %%
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
    image = axis.imshow(
        data.T,
        origin="lower",
        extent=[0.0, grid.axes_bounds[0][1], 0.0, grid.axes_bounds[1][1]],
        aspect="equal",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
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


snapshot_indices = [0, 1, 2, 4, -1]
fig, axes = plt.subplots(1, len(snapshot_indices), figsize=(17, 3.7))

for axis, snapshot_index in zip(axes, snapshot_indices):
    plot_field_image(
        axis,
        darcy.grid,
        concentration_history[snapshot_index],
        f"t = {times[snapshot_index] / 3600:.1f} h",
        cmap="viridis",
        colorbar_label="c",
        vmin=0.0,
        vmax=1.0,
    )
    draw_obstacle(axis, darcy)

fig.tight_layout()
plt.show()

# %%
positions_x, positions_y = darcy.grid.axes_coords
center_y_index = int(np.argmin(np.abs(positions_y - 0.5 * darcy.length)))

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

plot_field_image(
    axes[0],
    darcy.grid,
    pressure.data / 1e6,
    "Pressao e linhas de corrente",
    cmap="cividis",
    colorbar_label="MPa",
)
axes[0].streamplot(
    positions_x,
    positions_y,
    pore_velocity.data[0].T,
    pore_velocity.data[1].T,
    color="white",
    density=1.1,
    linewidth=0.8,
)
draw_obstacle(axes[0], darcy)

for snapshot_index in snapshot_indices:
    axes[1].plot(
        positions_x,
        concentration_history[snapshot_index, :, center_y_index],
        label=f"t = {times[snapshot_index] / 3600:.1f} h",
    )

axes[1].set_xlabel("x [m]")
axes[1].set_ylabel(rf"$c(x,{positions_y[center_y_index]:.1f},t)$")
axes[1].set_ylim(-0.05, 1.05)
axes[1].set_title("Corte central da concentracao")
axes[1].legend(fontsize=8)

fig.tight_layout()
plt.show()
