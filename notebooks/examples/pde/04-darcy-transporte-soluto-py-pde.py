# %% [markdown]
# # Darcy acoplado com transporte de soluto
#
# Nesta aula, usamos o campo de velocidade obtido pelo problema de Darcy para
# transportar um soluto diluido em meio poroso.
#
# A pressao satisfaz
#
# $$
# \nabla\cdot\left(\frac{k}{\mu}\nabla p\right)=0,
# \qquad
# \mathbf q=-\frac{k}{\mu}\nabla p.
# $$
#
# Para o soluto, usamos a forma conservativa
#
# $$
# \phi\frac{\partial c}{\partial t}
# +
# \nabla\cdot(\mathbf q c-\phi D\nabla c)=0.
# $$
#
# O soluto e injetado na borda de maior pressao. O estado tem duas variaveis,
# pressao e concentracao, representadas por `FieldCollection`.

# %% [markdown]
# ## Importando as dependencias

# %%
from matplotlib.patches import Rectangle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pde
import pde.fields.collection as pde_field_collection
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from pde.solvers import ExplicitSolver
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
_original_number_array = pde_field_collection.number_array


def number_array_numpy2_compatible(data, dtype=None, copy=True):
    try:
        return _original_number_array(data, dtype=dtype, copy=copy)
    except ValueError as error:
        if copy is False and "Unable to avoid copy" in str(error):
            target_dtype = get_common_dtype(data) if dtype is None else np.dtype(dtype)
            return np.asarray(data, dtype=target_dtype)
        raise


pde_field_collection.number_array = number_array_numpy2_compatible


# %% [markdown]
# ## Problema de Darcy

# %%
class DarcyObstacle2D:
    def __init__(
        self,
        length=50.0,
        shape=(60, 60),
        obstacle_area_fraction=0.15,
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

    def coordinates(self):
        positions_x, positions_y = self.grid.axes_coords
        return np.meshgrid(positions_x, positions_y, indexing="ij")

    def obstacle_bounds(self):
        start = 0.5 * (self.length - self.obstacle_side)
        end = 0.5 * (self.length + self.obstacle_side)
        return start, end

    def obstacle_mask(self):
        coordinates_x, coordinates_y = self.coordinates()
        start, end = self.obstacle_bounds()
        return (
            (coordinates_x >= start)
            & (coordinates_x <= end)
            & (coordinates_y >= start)
            & (coordinates_y <= end)
        )

    def permeability_data(self):
        permeability = self.permeability_matrix * np.ones(self.shape)
        permeability[self.obstacle_mask()] = self.permeability_obstacle
        return permeability

    def mobility_data(self):
        return self.permeability_data() / self.viscosity

    @staticmethod
    def harmonic_mean(left_value, right_value):
        return 2.0 * left_value * right_value / (left_value + right_value)

    def cell_index(self, cell_x, cell_y):
        return cell_x * self.shape[1] + cell_y

    def assemble_pressure_system(self):
        num_cells_x, num_cells_y = self.shape
        step_x, step_y = map(float, self.grid.discretization)
        mobility = self.mobility_data()

        rows = []
        columns = []
        values = []
        right_hand_side = np.zeros(num_cells_x * num_cells_y)

        for cell_x in range(num_cells_x):
            for cell_y in range(num_cells_y):
                row = self.cell_index(cell_x, cell_y)
                diagonal = 0.0

                if cell_x == 0:
                    transmissibility = 2.0 * mobility[cell_x, cell_y] * step_y / step_x
                    diagonal += transmissibility
                    right_hand_side[row] += transmissibility * self.pressure_left
                else:
                    face_mobility = self.harmonic_mean(
                        mobility[cell_x - 1, cell_y],
                        mobility[cell_x, cell_y],
                    )
                    transmissibility = face_mobility * step_y / step_x
                    diagonal += transmissibility
                    rows.append(row)
                    columns.append(self.cell_index(cell_x - 1, cell_y))
                    values.append(-transmissibility)

                if cell_x == num_cells_x - 1:
                    transmissibility = 2.0 * mobility[cell_x, cell_y] * step_y / step_x
                    diagonal += transmissibility
                    right_hand_side[row] += transmissibility * self.pressure_right
                else:
                    face_mobility = self.harmonic_mean(
                        mobility[cell_x, cell_y],
                        mobility[cell_x + 1, cell_y],
                    )
                    transmissibility = face_mobility * step_y / step_x
                    diagonal += transmissibility
                    rows.append(row)
                    columns.append(self.cell_index(cell_x + 1, cell_y))
                    values.append(-transmissibility)

                if cell_y > 0:
                    face_mobility = self.harmonic_mean(
                        mobility[cell_x, cell_y - 1],
                        mobility[cell_x, cell_y],
                    )
                    transmissibility = face_mobility * step_x / step_y
                    diagonal += transmissibility
                    rows.append(row)
                    columns.append(self.cell_index(cell_x, cell_y - 1))
                    values.append(-transmissibility)

                if cell_y < num_cells_y - 1:
                    face_mobility = self.harmonic_mean(
                        mobility[cell_x, cell_y],
                        mobility[cell_x, cell_y + 1],
                    )
                    transmissibility = face_mobility * step_x / step_y
                    diagonal += transmissibility
                    rows.append(row)
                    columns.append(self.cell_index(cell_x, cell_y + 1))
                    values.append(-transmissibility)

                rows.append(row)
                columns.append(row)
                values.append(diagonal)

        matrix = sp.coo_matrix(
            (values, (rows, columns)),
            shape=(num_cells_x * num_cells_y, num_cells_x * num_cells_y),
        ).tocsr()
        return matrix, right_hand_side

    def solve_pressure(self):
        matrix, right_hand_side = self.assemble_pressure_system()
        pressure = spla.spsolve(matrix, right_hand_side).reshape(self.shape)
        return pde.ScalarField(self.grid, data=pressure, label="p [Pa]")

    def face_fluxes(self, pressure):
        pressure_data = np.asarray(pressure.data)
        num_cells_x, num_cells_y = self.shape
        step_x, step_y = map(float, self.grid.discretization)
        mobility = self.mobility_data()

        flux_x = np.zeros((num_cells_x + 1, num_cells_y))
        flux_y = np.zeros((num_cells_x, num_cells_y + 1))

        flux_x[0, :] = -mobility[0, :] * (
            pressure_data[0, :] - self.pressure_left
        ) / (0.5 * step_x)
        flux_x[-1, :] = -mobility[-1, :] * (
            self.pressure_right - pressure_data[-1, :]
        ) / (0.5 * step_x)

        for face_x in range(1, num_cells_x):
            face_mobility = self.harmonic_mean(
                mobility[face_x - 1, :],
                mobility[face_x, :],
            )
            flux_x[face_x, :] = -face_mobility * (
                pressure_data[face_x, :] - pressure_data[face_x - 1, :]
            ) / step_x

        for face_y in range(1, num_cells_y):
            face_mobility = self.harmonic_mean(
                mobility[:, face_y - 1],
                mobility[:, face_y],
            )
            flux_y[:, face_y] = -face_mobility * (
                pressure_data[:, face_y] - pressure_data[:, face_y - 1]
            ) / step_y

        return flux_x, flux_y

    def cell_center_velocity(self, pressure):
        flux_x, flux_y = self.face_fluxes(pressure)
        velocity_x = 0.5 * (flux_x[:-1, :] + flux_x[1:, :])
        velocity_y = 0.5 * (flux_y[:, :-1] + flux_y[:, 1:])
        speed = np.sqrt(velocity_x**2 + velocity_y**2)
        return velocity_x, velocity_y, speed


darcy = DarcyObstacle2D()
pressure = darcy.solve_pressure()
flux_x, flux_y = darcy.face_fluxes(pressure)
velocity_x, velocity_y, speed = darcy.cell_center_velocity(pressure)

# %% [markdown]
# ## Classe acoplada

# %%
class DarcyTransportPDE(pde.PDEBase):
    def __init__(
        self,
        flux_x,
        flux_y,
        porosity=0.2,
        diffusion=2e-5,
        inlet_concentration=1.0,
        external_concentration=0.0,
    ):
        super().__init__()
        self.flux_x = np.asarray(flux_x)
        self.flux_y = np.asarray(flux_y)
        self.porosity = float(porosity)
        self.diffusion = float(diffusion)
        self.inlet_concentration = float(inlet_concentration)
        self.external_concentration = float(external_concentration)

    def concentration_rate(self, concentration):
        concentration_data = np.asarray(concentration.data)
        step_x, step_y = map(float, concentration.grid.discretization)
        num_cells_x, num_cells_y = concentration_data.shape

        total_flux_x = np.zeros((num_cells_x + 1, num_cells_y))
        total_flux_y = np.zeros((num_cells_x, num_cells_y + 1))

        for face_x in range(1, num_cells_x):
            face_flux = self.flux_x[face_x, :]
            upwind_concentration = np.where(
                face_flux >= 0.0,
                concentration_data[face_x - 1, :],
                concentration_data[face_x, :],
            )
            total_flux_x[face_x, :] = face_flux * upwind_concentration

        left_flux = self.flux_x[0, :]
        total_flux_x[0, :] = np.where(
            left_flux >= 0.0,
            left_flux * self.inlet_concentration,
            left_flux * concentration_data[0, :],
        )

        right_flux = self.flux_x[-1, :]
        total_flux_x[-1, :] = np.where(
            right_flux >= 0.0,
            right_flux * concentration_data[-1, :],
            right_flux * self.external_concentration,
        )

        for face_y in range(1, num_cells_y):
            face_flux = self.flux_y[:, face_y]
            upwind_concentration = np.where(
                face_flux >= 0.0,
                concentration_data[:, face_y - 1],
                concentration_data[:, face_y],
            )
            total_flux_y[:, face_y] = face_flux * upwind_concentration

        total_flux_x[1:-1, :] += (
            -self.porosity
            * self.diffusion
            * (concentration_data[1:, :] - concentration_data[:-1, :])
            / step_x
        )
        total_flux_y[:, 1:-1] += (
            -self.porosity
            * self.diffusion
            * (concentration_data[:, 1:] - concentration_data[:, :-1])
            / step_y
        )

        divergence = (total_flux_x[1:, :] - total_flux_x[:-1, :]) / step_x
        divergence += (total_flux_y[:, 1:] - total_flux_y[:, :-1]) / step_y
        rate = -divergence / self.porosity
        return pde.ScalarField(concentration.grid, data=rate, label="dc/dt")

    def evolution_rate(self, state, t=0):
        pressure_field, concentration = state
        pressure_rate = pde.ScalarField(
            pressure_field.grid,
            data=np.zeros_like(pressure_field.data),
            label="dp/dt",
        )
        concentration_rate = self.concentration_rate(concentration)
        return pde.FieldCollection([pressure_rate, concentration_rate])


transport_model = DarcyTransportPDE(flux_x, flux_y)

# %% [markdown]
# ## Condicao inicial e escala de tempo

# %%
initial_pressure = pde.ScalarField(
    darcy.grid,
    data=pressure.data.copy(),
    label="p [Pa]",
)
initial_concentration = pde.ScalarField(
    darcy.grid,
    data=np.zeros(darcy.shape),
    label="c [-]",
)
initial_state = pde.FieldCollection([initial_pressure, initial_concentration])

step_x, step_y = map(float, darcy.grid.discretization)
max_pore_velocity = np.max(speed) / transport_model.porosity
cfl_time_step = 0.35 * min(step_x, step_y) / max_pore_velocity
time_step = min(100.0, cfl_time_step)
final_time = 60_000.0

pd.DataFrame(
    {
        "quantidade": [
            "lado do obstaculo",
            "fracao de area do obstaculo",
            "porosidade",
            "D efetivo",
            "max |q|",
            "max |q|/phi",
            "dt CFL",
            "dt usado",
            "tempo final",
        ],
        "valor": [
            f"{darcy.obstacle_side:.2f} m",
            f"{100 * darcy.obstacle_area_fraction:.1f}%",
            transport_model.porosity,
            f"{transport_model.diffusion:.2e} m^2/s",
            f"{np.max(speed):.3e} m/s",
            f"{max_pore_velocity:.3e} m/s",
            f"{cfl_time_step:.1f} s",
            f"{time_step:.1f} s",
            f"{final_time / 3600:.1f} h",
        ],
    }
)

# %% [markdown]
# ## Simulacao do transporte

# %%
storage = pde.MemoryStorage()
solver = ExplicitSolver(transport_model, scheme="runge-kutta", backend="numpy")
controller = pde.Controller(
    solver,
    t_range=final_time,
    tracker=storage.tracker(final_time / 8.0),
)
final_state = controller.run(initial_state, dt=time_step)

times = np.array(list(storage.times))
concentration_history = np.array([state[1].data for _, state in storage.items()])

controller.diagnostics["solver"]

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
    velocity_x.T,
    velocity_y.T,
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

# %% [markdown]
# Esta configuracao usa a mesma escala fisica da primeira versao da aula:
# velocidade de Darcy dimensional, CFL baseado na velocidade de poro e tempo
# final de aproximadamente 16,7 horas.
