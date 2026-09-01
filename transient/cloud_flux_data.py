"""Input configuration and SolarPILOT data generation for cloud-flux studies."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from copylot import CoPylot


@dataclass(frozen=True)
class FluxExperimentConfig:
    """All inputs needed to generate a time series of heliostat optical data."""

    weather_file: Path
    output_dir: Path = Path("outputs/cloud_flux")
    x_res: int = 50
    y_res: int = 50
    rated_power_mw: float = 550
    tower_height_m: float = 190
    receiver_diameter_m: float = 17.65
    receiver_height_m: float = 21.8
    start_hour: float = 9.0
    end_hour: float = 15.0
    n_timesteps: int = 5

    @property
    def receiver_radius_m(self) -> float:
        return self.receiver_diameter_m / 2

    @property
    def receiver_circumference_m(self) -> float:
        return 2 * np.pi * self.receiver_radius_m

    @property
    def time_steps(self) -> np.ndarray:
        if self.n_timesteps < 1:
            raise ValueError("n_timesteps must be at least 1.")
        return np.linspace(self.start_hour, self.end_hour, self.n_timesteps)


def _extract_heliostat_data(cp: CoPylot, data_handle: int, results: dict, config: FluxExperimentConfig):
    """Extract per-heliostat optical data for one simulated timestep."""
    x_locations = np.asarray(results["x_location"])
    y_locations = np.asarray(results["y_location"])
    distances = np.hypot(x_locations, y_locations)
    if np.any(distances == 0):
        raise ValueError("A heliostat is located at the tower origin; image scale is undefined.")

    y_scale = np.hypot(distances, config.tower_height_m) / distances
    image_sizes = np.asarray([
        cp.get_heliostat_image_sizes(data_handle, heliostat_id=i)
        for i in range(len(x_locations))
    ])
    amplitudes = np.asarray([
        np.asarray(cp.get_heliostat_fluxmap(data_handle, heliostat_id=i)).max() * 1.1
        for i in range(len(x_locations))
    ])
    image_sizes = np.column_stack((
        config.tower_height_m * image_sizes[:, 0],
        config.tower_height_m * image_sizes[:, 1] * y_scale,
    ))
    x_normal = np.mod(
        (np.arctan2(y_locations, x_locations) - np.pi / 2) * config.receiver_radius_m,
        config.receiver_circumference_m,
    )
    y_normal = np.full(len(x_locations), config.receiver_height_m / 2)
    return x_locations, y_locations, image_sizes, amplitudes, x_normal, y_normal


def _save_timestep_data(data: tuple, output_path: Path, config: FluxExperimentConfig) -> None:
    x_locations, y_locations, image_sizes, amplitudes, x_normal, y_normal = data
    # Shift the nominal aimpoint to the unwrapped receiver coordinate convention.
    half_circumference = np.pi * config.receiver_diameter_m / 2
    x_unwrapped = np.where(
        (x_normal > 0) & (x_normal < half_circumference),
        x_normal + half_circumference,
        x_normal - half_circumference,
    )
    pd.DataFrame({
        "heliostat_id": np.arange(len(image_sizes)),
        "sigma_x": image_sizes[:, 0],
        "sigma_y": image_sizes[:, 1],
        "amplitude": amplitudes,
        "x_locations": x_locations,
        "y_locations": y_locations,
        "x_normal": x_unwrapped,
        "y_normal": y_normal,
    }).to_csv(output_path, index=False)


def generate_timestep_data(config: FluxExperimentConfig) -> list[Path]:
    """Run SolarPILOT at each configured hour and save one CSV per timestep."""
    weather_file = Path(config.weather_file)
    if not weather_file.is_file():
        raise FileNotFoundError(f"Weather file not found: {weather_file}")
    config.output_dir.mkdir(parents=True, exist_ok=True)

    cp = CoPylot(debug=False)
    data_handle = cp.data_create()
    try:
        cp.data_set_string(data_handle, "ambient.0.weather_file", str(weather_file))
        cp.data_set_number(data_handle, "fluxsim.0.x_res", config.x_res)
        cp.data_set_number(data_handle, "fluxsim.0.y_res", config.y_res)
        cp.data_set_number(data_handle, "solarfield.0.q_des", config.rated_power_mw)
        cp.data_set_number(data_handle, "solarfield.0.tht", config.tower_height_m)
        cp.data_set_number(data_handle, "receiver.0.rec_diameter", config.receiver_diameter_m)
        cp.data_set_number(data_handle, "receiver.0.rec_height", config.receiver_height_m)
        cp.generate_layout(data_handle)

        csv_files = []
        for index, hour in enumerate(config.time_steps):
            cp.data_set_number(data_handle, "fluxsim.0.flux_hour", float(hour))
            cp.update_geometry(data_handle)
            cp.simulate(data_handle)
            output_path = config.output_dir / f"heliostat_t{index:02d}.csv"
            _save_timestep_data(
                _extract_heliostat_data(cp, data_handle, cp.detail_results(data_handle), config),
                output_path,
                config,
            )
            csv_files.append(output_path)
            print(f"Saved timestep {index} (flux_hour={hour:.2f}) -> {output_path}")
        return csv_files
    finally:
        cp.data_free(data_handle)
