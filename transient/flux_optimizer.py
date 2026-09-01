"""Multi-timestep heliostat aimpoint optimization for a cylindrical receiver."""

from pathlib import Path
import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from mmaWrapper import runMMA

jax.config.update("jax_enable_x64", False)


class HeliostatFluxOptimizerTimeSeries:
    """Optimize receiver aimpoint offsets jointly across a series of timesteps."""

    def __init__(
        self,
        target_flux_path: Path,
        heliostat_csv_files: list[Path],
        receiver_radius_m: float,
        receiver_height_m: float,
        move_penalty_weight: float = 0.05,
        move_cap_m: float | None = None,
        cap_sharpness: float = 50.0,
    ):
        self.receiver_radius_m = receiver_radius_m
        self.receiver_height_m = receiver_height_m
        self.circumference_m = 2 * np.pi * receiver_radius_m
        self.move_penalty_weight = move_penalty_weight
        self.move_cap_m = move_cap_m
        self.cap_sharpness = cap_sharpness
        self.T = len(heliostat_csv_files)
        if self.T < 2:
            raise ValueError("At least two timesteps are required.")

        self.heliostat_csv_files = [Path(p) for p in heliostat_csv_files]
        self.F_target, heliostat_data = self._load_data(
            Path(target_flux_path), self.heliostat_csv_files
        )
        self.Ny, self.Nx = self.F_target.shape
        self.N_total = heliostat_data[0].shape[0]
        self.x = jnp.linspace(0.0, self.circumference_m, self.Nx)
        self.y = jnp.linspace(0.0, self.receiver_height_m, self.Ny)
        self.X, self.Y = jnp.meshgrid(self.x, self.y)
        self._setup_optical_params(heliostat_data)

        self.mse_and_grad = jax.jit(jax.value_and_grad(self.loss))
        self.con_and_grad = jax.jit(jax.value_and_grad(self.movement_cap_constraint))
        self.F_target_max = jnp.maximum(jnp.max(self.F_target), 1e-8)
        self.iter_hist, self.obj_hist, self.con_hist = [], [], []
        self.x_best = self.cluster_abr_opt = self.final_loss = self.last_result = None

    @staticmethod
    def _load_csv(path: Path, skiprows: int = 0) -> np.ndarray:
        if not path.is_file():
            raise FileNotFoundError(f"Required input file not found: {path}")
        try:
            values = np.loadtxt(path, delimiter=",", skiprows=skiprows)
        except ValueError as error:
            raise ValueError(f"Could not parse numeric data in {path}: {error}") from error
        if values.size == 0:
            raise ValueError(f"Input file is empty: {path}")
        return np.atleast_2d(values)

    def _load_data(self, target_flux_path: Path, csv_files: list[Path]):
        target = jnp.asarray(self._load_csv(target_flux_path) / 1e3)
        heliostat_data = [self._load_csv(path, skiprows=1) for path in csv_files]
        if any(data.shape[1] != 8 for data in heliostat_data):
            raise ValueError("Heliostat CSV files must contain exactly 8 columns.")
        count = heliostat_data[0].shape[0]
        if any(data.shape[0] != count for data in heliostat_data):
            raise ValueError("All timestep CSV files must contain the same heliostats.")
        return target, heliostat_data

    def _setup_optical_params(self, heliostat_data: list[np.ndarray]) -> None:
        stacked = np.stack(heliostat_data)
        self.sigma_x = jnp.asarray(stacked[:, :, 1])
        self.sigma_y = jnp.asarray(stacked[:, :, 2])
        self.amplitude = jnp.asarray(stacked[:, :, 3])
        self.x_normal = jnp.asarray(stacked[0, :, 6])
        self.y_normal = jnp.asarray(stacked[0, :, 7])

    def _periodic_distance(self, x1, x2):
        return jnp.mod(x1 - x2 + self.circumference_m / 2, self.circumference_m) - self.circumference_m / 2

    def _physical_positions(self, normalized_offsets):
        a_max = self.receiver_radius_m * jnp.pi / 2
        b_max = self.receiver_height_m / 2
        x_pos = jnp.mod(self.x_normal + (normalized_offsets[:, :, 0] - 0.5) * 2 * a_max, self.circumference_m)
        y_pos = self.y_normal + (normalized_offsets[:, :, 1] - 0.5) * 2 * b_max
        return x_pos, y_pos

    def _flux_single_timestep(self, normalized_offsets, sigma_x, sigma_y, amplitude):
        a_max = self.receiver_radius_m * jnp.pi / 2
        b_max = self.receiver_height_m / 2
        a_offsets = (normalized_offsets[:, 0] - 0.5) * 2 * a_max
        b_offsets = (normalized_offsets[:, 1] - 0.5) * 2 * b_max
        theta_diff = self._periodic_distance(self.X[None, :, :], self.x_normal[:, None, None]) / self.receiver_radius_m
        projected_x = self.receiver_radius_m * jnp.sin(theta_diff)
        gamma_factor = self.receiver_radius_m / jnp.sqrt(
            jnp.maximum(self.receiver_radius_m**2 - projected_x**2, 1e-6)
        )
        gaussian = amplitude[:, None, None] * jnp.exp(-0.5 * (
            ((projected_x - a_offsets[:, None, None]) / sigma_x[:, None, None]) ** 2
            + ((self.Y[None, :, :] - self.y_normal[:, None, None] - b_offsets[:, None, None])
               / sigma_y[:, None, None]) ** 2
        )) / gamma_factor
        return jnp.sum(gaussian * (jnp.abs(theta_diff) <= jnp.pi / 2), axis=0)

    def total_flux_all_timesteps(self, normalized_offsets):
        return jax.vmap(self._flux_single_timestep)(normalized_offsets, self.sigma_x, self.sigma_y, self.amplitude)

    def loss(self, flattened_offsets):
        offsets = flattened_offsets.reshape(self.T, self.N_total, 2)
        flux = self.total_flux_all_timesteps(offsets)
        tracking = jnp.mean(((flux - self.F_target) / self.F_target_max) ** 2)
        x_pos, y_pos = self._physical_positions(offsets)
        dx = self._periodic_distance(x_pos[1:], x_pos[:-1])
        movement = jnp.mean(dx**2 + (y_pos[1:] - y_pos[:-1])**2)
        return tracking + self.move_penalty_weight * movement

    def movement_cap_constraint(self, flattened_offsets):
        if self.move_cap_m is None:
            return jnp.array(-1e-3)
        offsets = flattened_offsets.reshape(self.T, self.N_total, 2)
        x_pos, y_pos = self._physical_positions(offsets)
        distances_squared = self._periodic_distance(x_pos[1:], x_pos[:-1])**2 + (y_pos[1:] - y_pos[:-1])**2
        smooth_max = jax.scipy.special.logsumexp(self.cap_sharpness * distances_squared) / self.cap_sharpness
        return smooth_max - self.move_cap_m**2

    def _loss_and_gradient(self, offsets):
        loss, gradient = self.mse_and_grad(jnp.asarray(offsets, dtype=jnp.float32))
        return float(loss), np.asarray(gradient, dtype=np.float32).reshape(-1, 1)

    def _constraint_and_gradient(self, offsets):
        constraint, gradient = self.con_and_grad(jnp.asarray(offsets, dtype=jnp.float32))
        return np.asarray([constraint]).reshape(-1, 1), np.asarray(gradient, dtype=np.float32).reshape(1, -1)

    def _mma_input(self, offsets):
        loss, loss_gradient = self._loss_and_gradient(offsets)
        constraint, constraint_gradient = self._constraint_and_gradient(offsets)
        return loss, loss_gradient, constraint, constraint_gradient

    def _record_iteration(self, iteration, _offsets, objective, constraint):
        self.iter_hist.append(iteration)
        self.obj_hist.append(objective)
        self.con_hist.append(constraint.item())

    def _initial_offsets(self, seed: int, init_range: float = 0.2) -> np.ndarray:
        rng = np.random.default_rng(seed)
        single_timestep = rng.uniform(0.5 - init_range, 0.5 + init_range, size=(self.N_total, 2))
        return np.broadcast_to(single_timestep, (self.T, self.N_total, 2)).copy().ravel()

    def optimize_mma(self, seed: int = 42, restarts: int = 1, max_iterations: int = 150,
                     min_iterations: int = 10, time_limit_seconds: int = 3600,
                     move_limit: float = 0.2, verbose: bool = False):
        """Run MMA and return optimized normalized offsets and the final loss."""
        offsets = self._initial_offsets(seed)
        for _ in range(5):  # Compile JAX once before timing/optimizing.
            self._mma_input(offsets)
        start = time.perf_counter()
        self._mma_input(offsets)
        print(f"One objective evaluation: {time.perf_counter() - start:.3f}s (T={self.T}, N={self.N_total})")

        best_loss = np.inf
        rng = np.random.default_rng(seed + 1)
        for restart in range(restarts):
            self.iter_hist, self.obj_hist, self.con_hist = [], [], []
            result = runMMA(
                self._mma_input, offsets.reshape(-1, 1), np.zeros_like(offsets).reshape(-1, 1),
                np.ones_like(offsets).reshape(-1, 1), fTolerance=1e-4, gTolerance=1e-2,
                maxIterations=max_iterations, minIterations=min_iterations,
                timeLimitSecs=time_limit_seconds, move_limit=move_limit, kktTol=1e-6,
                verbose=verbose, progress_callback=None, callback=self._record_iteration,
            )
            if result[1] < best_loss:
                self.x_best, best_loss = np.asarray(result[0]).ravel(), result[1]
                offsets = self.x_best.copy()
            else:
                offsets = rng.uniform(0, 1, size=offsets.size)
            self.last_result = result
            print(f"Restart {restart + 1}/{restarts}: best loss {best_loss:.6e}")

        self.cluster_abr_opt = jnp.asarray(self.x_best.reshape(self.T, self.N_total, 2))
        self.final_loss = self._loss_and_gradient(self.x_best)[0]
        return self.cluster_abr_opt, self.final_loss

    def flux_at(self, timestep: int):
        if self.cluster_abr_opt is None:
            raise RuntimeError("Run optimize_mma before requesting optimized flux.")
        return self._flux_single_timestep(self.cluster_abr_opt[timestep], self.sigma_x[timestep], self.sigma_y[timestep], self.amplitude[timestep])

    def plot_target(self):
        plt.figure(figsize=(10, 6))
        plt.pcolormesh(self.X, self.Y, self.F_target, shading="auto", cmap="viridis")
        plt.colorbar(label="Flux [kW/m²]")
        plt.xlabel("Circumference [m]")
        plt.ylabel("Height [m]")
        plt.title("Target receiver flux map")
        plt.tight_layout()
        plt.show()

    def plot_history(self):
        if not self.iter_hist:
            raise RuntimeError("No optimization history is available.")
        figure, objective_axis = plt.subplots()
        constraint_axis = objective_axis.twinx()
        objective_axis.plot(self.iter_hist, self.obj_hist, "b-o", label="Objective")
        constraint_axis.plot(self.iter_hist, self.con_hist, "r-o", label="Constraint")
        objective_axis.set(xlabel="Iteration", ylabel="Objective")
        constraint_axis.set_ylabel("Movement-cap constraint")
        figure.tight_layout()
        plt.show()

    def plot_timestep(self, timestep: int, show_aimpoints: bool = False):
        target, optimized = self.F_target, self.flux_at(timestep)
        figure, axes = plt.subplots(1, 2, figsize=(16, 6))
        for axis, values, title in zip(axes, (target, optimized), ("Target flux map", f"Optimized flux map — timestep {timestep}")):
            image = axis.pcolormesh(self.X, self.Y, values, shading="auto", cmap="viridis")
            figure.colorbar(image, ax=axis, label="Flux [kW/m²]")
            axis.set(xlabel="Circumference [m]", ylabel="Height [m]", title=title)
        if show_aimpoints:
            x_pos, y_pos = self._physical_positions(self.cluster_abr_opt)
            axes[1].plot(x_pos[timestep], y_pos[timestep], "rx", markersize=4, alpha=0.7)
        figure.tight_layout()
        plt.show()


    def compute_total_travel(self) -> np.ndarray:
        """Total aimpoint travel distance per heliostat, summed across all timesteps."""
        if self.cluster_abr_opt is None:
            raise RuntimeError("Run optimize_mma before computing travel.")
        x_pos, y_pos = map(np.asarray, self._physical_positions(self.cluster_abr_opt))
        dx = np.asarray(self._periodic_distance(x_pos[1:], x_pos[:-1]))
        return np.sum(np.hypot(dx, np.diff(y_pos, axis=0)), axis=0)

    def plot_movement_summary(self, n_trajectories: int = 15):
        x_pos, y_pos = map(np.asarray, self._physical_positions(self.cluster_abr_opt))
        total_travel = self.compute_total_travel()
        figure, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].hist(total_travel, bins=30, color="steelblue")
        axes[0].set(xlabel="Total aimpoint travel [m]", ylabel="Heliostats", title="Aimpoint movement distribution")
        for index in np.argsort(-total_travel)[:n_trajectories]:
            x_values, y_values = x_pos[:, index].copy(), y_pos[:, index].copy()
            gaps = np.abs(np.diff(x_values)) > self.circumference_m / 2
            x_values = np.insert(x_values, np.where(gaps)[0] + 1, np.nan)
            y_values = np.insert(y_values, np.where(gaps)[0] + 1, np.nan)
            axes[1].plot(x_values, y_values, "-o", markersize=3, alpha=0.7)
        axes[1].set(xlabel="Circumference [m]", ylabel="Height [m]", title="Highest-movement aimpoint trajectories")
        figure.tight_layout()
        plt.show()
        print(f"Mean total travel: {total_travel.mean():.3f} m; max: {total_travel.max():.3f} m")

    def export_travel_to_csv(
        self,
        output_dir: Path | None = None,
        suffix: str = "_with_travel",
        timestep_index: int = 0,
    ) -> Path:
        """Write a copy of a heliostat CSV with an added `total_travel_m` column."""
        total_travel = self.compute_total_travel()
        source_path = self.heliostat_csv_files[timestep_index]
        output_dir = Path(output_dir) if output_dir is not None else source_path.parent
        output_path = output_dir / f"{source_path.stem}{suffix}{source_path.suffix}"

        with open(source_path, "r") as f:
            header = f.readline().rstrip("\n")

        data = np.atleast_2d(np.loadtxt(source_path, delimiter=",", skiprows=1))

        if data.shape[0] != total_travel.shape[0]:
            raise ValueError(
                f"Row count mismatch: CSV has {data.shape[0]} heliostats, "
                f"travel array has {total_travel.shape[0]}."
            )

        combined = np.column_stack([data, total_travel])
        np.savetxt(
            output_path, combined, delimiter=",",
            header=f"{header},total_travel_m", comments="", fmt="%.6f",
        )
        print(f"Wrote {output_path}")
        return output_path