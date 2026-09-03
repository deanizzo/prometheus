import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import Bounds
import time
import imageio.v2 as imageio # type: ignore
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from optimization.mmaWrapper import runMMA

jax.config.update("jax_enable_x64", False)

# ============================================================
# 1. MODEL, OPTIMIZATION, AND PLOTTING CLASS
# ============================================================
class HeliostatFluxOptimizer:
    """
    Heliostat flux optimizer for cylindrical receiver.
    
    Design variables: (a, b) for each heliostat
    - a: X-offset from x_normal (arc length displacement on cylinder)
    - b: Y-offset from y_normal (vertical displacement)
    
    Physics model (from heliostat_cylinder_flux.ipynb):
    - theta = dx / R (angular position)
    - projected_x = R * sin(theta)
    - Gaussian: f(projected_x - a, dy - b)
    - Gamma attenuation: γ = R / sqrt(R² - projected_x²)
    - Attenuation = 1/γ = cos(theta)  [matches full_2D_approx.py]
    - Visibility: ±90° around x_normal, 0 ≤ y ≤ H
    """
    def __init__(self, problem_name: str, receiver_radius: int = 8.825, receiver_height: int = 21.6,
                 var_column=2, gamma=4.0, sample_heliostats=None):
        self.problem_name = problem_name
        self.receiver_radius = receiver_radius
        self.receiver_height = receiver_height
        self.circumference = 2 * np.pi * receiver_radius
        self.var_column = var_column
        self.gamma = gamma
        self.sample_heliostats = sample_heliostats
        
        # Load data
        self.F_target, self.heliostat_data, self.solution_data = self._load_flux_map_heliostat_data(problem_name)
        
        # Setup grid (grid size derived from F_target shape)
        self._setup_grid()
        
        # Setup optical parameters
        self._setup_optical_params()
        
        # JIT compile loss and gradient functions
        self.mse_and_grad = jax.jit(jax.value_and_grad(self.mse_loss))
        self.con_and_grad = jax.jit(jax.value_and_grad(self.binarization_constraint))
        
        # History tracking
        self.iter_hist = []
        self.obj_hist = []
        self.con_hist = []
        self.x_hist = []
        self.x_best = None
        self.final_mse = None
        self.cluster_abr_opt = None
        self.last_result = None
        self.F_target_max = jnp.maximum(jnp.max(self.F_target), 1e-8)
        
    def _load_flux_map_heliostat_data(self, problem_name):
        if problem_name == "ideal_cylinder":
            csv_path = "./data_dynamic/ideal_flux.csv"
            heliostat_csv_path = "./data_dynamic/heliostat_dyn.csv"
            solution_csv_path = None
            scale_factor = 1.6e3
        else:
            raise ValueError(f"Unknown problem: {problem_name}")
            
        # Load target flux
        try:
            np_data = np.loadtxt(csv_path, delimiter=",")
            F_target = jnp.array(np_data)
        except:
            F_target = jnp.ones((61, 48))
        F_target = F_target / scale_factor
        
        # Load heliostat data
        # CSV columns: heliostat_id, sigma_x, sigma_y, amplitude, x_normal, y_normal
        try:
            np_data = np.loadtxt(heliostat_csv_path, delimiter=",", skiprows=1)
            heliostat_data = np_data  # Keep all 6 columns
            
            # Apply sampling if specified
            if self.sample_heliostats is not None and heliostat_data.shape[0] > self.sample_heliostats:
                idx = np.random.choice(heliostat_data.shape[0], self.sample_heliostats, replace=False)
                heliostat_data = heliostat_data[idx]
        except:
            heliostat_data = np.zeros((10, 6))
        
        # Load solution data if available
        solution_data = None
        if solution_csv_path is not None:
            try:
                np_data = np.loadtxt(solution_csv_path, delimiter=",", skiprows=1)
                solution_data = jnp.array(np_data)
            except:
                solution_data = None
            
        return F_target, heliostat_data, solution_data
    
    def _setup_grid(self):
        """Setup receiver grid (unrolled cylinder surface)"""
        self.Ny, self.Nx = self.F_target.shape[0], self.F_target.shape[1]
        self.x = jnp.linspace(0.0, self.circumference, self.Nx)
        self.y = jnp.linspace(0.0, self.receiver_height, self.Ny)
        self.X, self.Y = jnp.meshgrid(self.x, self.y)
        self.N_total = self.heliostat_data.shape[0]
    
    def _setup_optical_params(self):
        """Extract heliostat optical parameters from CSV data"""
        self.sigma_x = jnp.array(self.heliostat_data[:, 1])
        self.sigma_y = jnp.array(self.heliostat_data[:, 2])
        self.amplitude = jnp.array(self.heliostat_data[:, 3])
        print(f"Loaded {self.N_total} heliostats with optical parameters:")
        print(f"  sigma_x (first 5): {self.sigma_x[:5]}")
        self.x_normal = jnp.array(self.heliostat_data[:, 6])  # Fixed receiver normal X
        self.y_normal = jnp.array(self.heliostat_data[:, 7])  # Fixed receiver normal Y
    
    # ============================================================
    # CYLINDRICAL FLUX CALCULATION (from heliostat_cylinder_flux.ipynb)
    # ============================================================
    
    def _periodic_distance(self, x, x_normal):
        """
        Calculate periodic signed distance on cylinder.
        Wraps to [-circumference/2, circumference/2].
        """
        dx_signed = x - x_normal
        dx_signed = jnp.where(dx_signed > self.circumference / 2,
                              dx_signed - self.circumference,
                              jnp.where(dx_signed < -self.circumference / 2,
                                        dx_signed + self.circumference,
                                        dx_signed))
        return dx_signed
    
    def total_flux_vectorized(self, ab_norml):
        """
        Vectorized computation of total flux on cylindrical receiver.
        All heliostats are always active (rho removed as design variable).
        A heliostat contributes only within its ±90° visibility arc.
        
        Attenuation: 1/gamma only (= cos theta), matching full_2D_approx.py.
        No extra cosine incidence factor.
        
        Args:
            ab_norml: (n_heliostats, 2) array of [a_norm, b_norm]
                - a_norm: normalized X-offset [0, 1]  (0.5 = no offset)
                - b_norm: normalized Y-offset [0, 1]  (0.5 = no offset)
        
        Returns:
            flux: (Ny, Nx) total flux on receiver
        """
        n_heliostats = ab_norml.shape[0]
        
        # Denormalize offsets to physical units
        # a: X-offset range is ±90° = ±(R*π/2) for maximum physical range
        # b: Y-offset range is full height for maximum range
        a_max = self.receiver_radius * jnp.pi / 2  # ±90° arc length
        b_max = self.receiver_height / 2  # ±half height
        
        # Map [0, 1] to [-a_max, +a_max] and [-b_max, +b_max]
        a_offsets = (ab_norml[:, 0] - 0.5) * 2 * a_max  # X-offset in meters
        b_offsets = (ab_norml[:, 1] - 0.5) * 2 * b_max  # Y-offset in meters
        
        # Broadcast heliostat parameters
        x_norm = self.x_normal[:, None, None]  # (n, 1, 1)
        y_norm = self.y_normal[:, None, None]  # (n, 1, 1)
        sigma_x = self.sigma_x[:, None, None]  # (n, 1, 1)
        sigma_y = self.sigma_y[:, None, None]  # (n, 1, 1)
        amp = self.amplitude[:, None, None]  # (n, 1, 1)
        a = a_offsets[:, None, None]  # (n, 1, 1)
        b = b_offsets[:, None, None]  # (n, 1, 1)
        
        # Broadcast grid: (n, Ny, Nx)
        X_broadcast = jnp.broadcast_to(self.X[None, :, :], (n_heliostats, self.Ny, self.Nx))
        Y_broadcast = jnp.broadcast_to(self.Y[None, :, :], (n_heliostats, self.Ny, self.Nx))
        
        # Cylindrical distances and angle
        dx = self._periodic_distance(X_broadcast, x_norm)
        dy = Y_broadcast - y_norm
        theta_diff = dx / self.receiver_radius
        
        # Reference projection mapping: projected_x = R * sin(theta)
        projected_x = self.receiver_radius * jnp.sin(theta_diff)
        
        # Gaussian arguments with offsets (a, b)
        # Key: offsets are applied in projected space
        gx = projected_x - a
        gy = dy - b
        
        # Gamma attenuation: 1/gamma = cos(theta)  [matches full_2D_approx.py g() function]
        # No extra cosine incidence factor — attenuation is purely the projection Jacobian
        gamma_factor = self.receiver_radius / jnp.sqrt(
            jnp.maximum(self.receiver_radius ** 2 - projected_x ** 2, 1e-6)
        )
        
        # Gaussian flux with 1/gamma attenuation only
        gaussian = amp * jnp.exp(
            -0.5 * ((gx / sigma_x) ** 2 + (gy / sigma_y) ** 2)
        ) / gamma_factor
        
        # Visibility masks: ±90° around x_normal; outside this range contribution is zero
        mask_x = jnp.abs(theta_diff) <= jnp.pi / 2  # ±90° visibility
        mask_y = (Y_broadcast >= 0) & (Y_broadcast <= self.receiver_height)  # Receiver bounds
        mask = mask_x & mask_y
        
        # Apply visibility mask (no rho; all heliostats contribute within their visible arc)
        flux_all = gaussian * mask
        
        # Sum over all heliostats
        flux_total = jnp.sum(flux_all, axis=0)
        
        return flux_total
    
    # ============================================================
    # LOSS AND CONSTRAINT FUNCTIONS
    # ============================================================
    
    def mse_loss(self, ab_norml_flat):
        """Mean squared error loss between predicted and target flux"""
        ab_norml = ab_norml_flat.reshape((-1, self.var_column))
        F = self.total_flux_vectorized(ab_norml)
        phi = jnp.mean(((F - self.F_target) / self.F_target_max) ** 2)
        return phi
    
    def binarization_constraint(self, ab_norml_flat):
        """Binarization constraint - DISABLED (rho removed as design variable).
        Returns a trivially satisfied value so MMA constraint interface remains intact."""
        # rho removed; constraint always satisfied
        # ab_norml = ab_norml_flat.reshape((-1, self.var_column))
        # rho = ab_norml[:, 2]
        # rho_1_rho = rho * (1.0 - rho)
        # con = jnp.mean((self.gamma * rho_1_rho) / (1 + (self.gamma - 4) * rho_1_rho)) - 1e-3
        # return con
        return jnp.array(-1e-3)  # Always satisfied (g ≤ 0)
    
    def scipy_loss(self, ab_norml_flat_np):
        """Scipy-compatible loss function"""
        ab_norml_flat_np = np.asarray(ab_norml_flat_np, dtype=np.float32)
        ab_norml_flat = jnp.array(ab_norml_flat_np, dtype=jnp.float32)
        loss, grad = self.mse_and_grad(ab_norml_flat)
        return float(loss), np.asarray(grad, dtype=np.float32).reshape((-1, 1))
    
    def scipy_con(self, ab_norml_flat_np):
        """Scipy-compatible constraint function"""
        ab_norml_flat_np = np.asarray(ab_norml_flat_np, dtype=np.float32)
        ab_norml_flat = jnp.array(ab_norml_flat_np, dtype=jnp.float32)
        con, grad = self.con_and_grad(ab_norml_flat)
        return np.array([con]).reshape((-1, 1)), np.asarray(grad, dtype=np.float32).reshape((1, -1))
    
    # ============================================================
    # INITIALIZATION AND OPTIMIZATION
    # ============================================================
    
    def initialize_clusters(self, seed=42, init_rho=0.75, init_a_range=0.2, init_b_range=0.2, NN=None):
        """
        Initialize optimization variables (a, b).
        
        Args:
            seed: random seed
            init_rho: unused (kept for API compatibility)
            init_a_range: initial range for a_norm (±init_a_range from 0.5)
            init_b_range: initial range for b_norm (±init_b_range from 0.5)
            NN: number of heliostats (default: all)
        """
        NN = self.N_total if NN is None else NN
        rng = np.random.default_rng(seed)
        
        # Note: normalized a, b are in [0, 1], where 0.5 represents no offset
        ab_np = np.zeros((NN, self.var_column))
        ab_np[:, 0] = rng.uniform(0.5 - init_a_range, 0.5 + init_a_range, NN)  # a_norm
        ab_np[:, 1] = rng.uniform(0.5 - init_b_range, 0.5 + init_b_range, NN)  # b_norm
        
        cluster_ab = jnp.array(ab_np)
        x0 = np.asarray(cluster_ab, dtype=np.float64).reshape(-1)
        
        # Bounds: a, b ∈ [0, 1]
        bounds = Bounds([0.0] * (NN * 2), [1.0] * (NN * 2))
        
        return cluster_ab, x0, bounds, rng
    
    def mma_in_fun(self, x0):
        """MMA input function (loss and constraint with gradients)"""
        f0val, df0dx = self.scipy_loss(x0)
        fval, dfdx = self.scipy_con(x0)
        return f0val, df0dx, fval, dfdx
    
    def mma_callback(self, iteration, xk, fk, gk):
        """Callback to track optimization history"""
        self.iter_hist.append(iteration)
        self.obj_hist.append(fk)
        self.con_hist.append(gk.item())
        self.x_hist.append(xk.reshape(-1).copy())
    
    def optimize_mma(self, seed=42, init_rho=0.75, n_st=2, fTolerance=1e-4, gTolerance=1e-2, 
                    maxIterations=100, minIterations=10, timeLimitSecs=3600, move_limit=0.2, 
                    kktTol=1e-6, verbose=False, getGIF=False):
        """
        Run MMA optimization to find optimal (a, b) for each heliostat.
        
        Args:
            seed: random seed
            init_rho: unused (kept for API compatibility; rho removed as design variable)
            n_st: number of random restarts
            Other args: MMA optimizer parameters
        """
        cluster_ab, x0, bounds, rng = self.initialize_clusters(seed=seed, init_rho=init_rho)
        
        # Benchmark function evaluation time
        nn = 5
        st = time.perf_counter()
        for _ in range(nn):
            self.mma_in_fun(x0)
        et = time.perf_counter()
        print(f"Avg time taken for one fun eval: {(et - st)/nn:.6f} seconds")
        
        fbest = np.inf
        for i in range(n_st):
            self.iter_hist = []
            self.obj_hist = []
            self.con_hist = []
            self.x_hist = []
            
            result = runMMA(
                self.mma_in_fun,
                x0.reshape((-1, 1)),
                np.full_like(x0, 0.0).reshape((-1, 1)),  # a, b lower bound = 0
                np.full_like(x0, 1.0).reshape((-1, 1)),  # a, b upper bound = 1
                fTolerance=fTolerance,
                gTolerance=gTolerance,
                maxIterations=maxIterations,
                minIterations=minIterations,
                timeLimitSecs=timeLimitSecs,
                move_limit=move_limit,
                kktTol=kktTol,
                verbose=verbose,
                progress_callback=None,
                callback=self.mma_callback
            )
            
            if result[1] < fbest:
                self.x_best = np.asarray(result[0]).reshape(-1)
                fbest = result[1]
                x0 = self.x_best.copy()
            else:
                # Random restart: uniformly sample a, b in [0, 1]
                x0 = rng.uniform(0, 1, x0.size)
            
            print(f"Restart {i+1}/{n_st}, Best Objective so far: {float(fbest):.6e}, " + 
                  f"solved by MMA in {float(result[-1]):.2f} seconds, took {int(result[-2])} iterations")
            self.last_result = result
        
        self.cluster_abr_opt = jnp.array(self.x_best.reshape((-1, self.var_column)))
        self.final_mse, _ = self.scipy_loss(self.x_best)
        
        if getGIF:
            self.create_gif(gif_name="mma_flux_optimization_evolution.gif", fps=5, hold_last=20)
        
        return self.x_best, self.final_mse, self.cluster_abr_opt
    
    # ============================================================
    # PLOTTING AND VISUALIZATION
    # ============================================================
    
    def plot_target(self):
        """Plot target flux distribution"""
        plt.figure(figsize=(10, 8))
        plt.pcolormesh(self.X, self.Y, self.F_target, shading='auto', cmap='viridis')
        plt.colorbar(label='Flux [kW/m²]')
        plt.xlabel('Circumference [m]')
        plt.ylabel('Height [m]')
        plt.title(f'Target Receiver Flux Map\nType: {self.problem_name}')
        plt.axis('equal')
        plt.tight_layout()
        plt.show()
    
    def plot_flux_map(self, F, title='Flux Map'):
        """Plot a flux distribution"""
        plt.figure(figsize=(10, 8))
        plt.pcolormesh(self.X, self.Y, F, shading='auto', cmap='viridis')
        plt.colorbar(label='Flux [kW/m²]')
        plt.xlabel('Circumference [m]')
        plt.ylabel('Height [m]')
        plt.title(title)
        plt.axis('equal')
        plt.tight_layout()
        plt.show()
    
    def plot_history(self):
        """Plot optimization history"""
        if not self.iter_hist:
            print("No history to plot.")
            return
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.set_xlabel("Iteration", fontweight='bold')
        ax1.set_ylabel("Normalized objective function", color="b", fontweight='bold')
        ax2.set_ylabel("Constraint value", color="r", fontweight='bold')
        ax2.yaxis.set_label_position("right")
        ax2.yaxis.tick_right()
        ax1.plot(self.iter_hist, self.obj_hist, 'b-o', label="Objective")
        ax2.plot(self.iter_hist, self.con_hist, 'r-o', label="Constraint")
        ax1.tick_params(axis='y', labelcolor='b')
        ax2.tick_params(axis='y', labelcolor='r')
        plt.pause(0.01)
    
    def create_gif(self, gif_name="flux_opt_evolution.gif", fps=5, hold_last=20, save_flux_maps=False):
        """Create animated GIF of optimization evolution"""
        if not self.x_hist:
            print("No history to create GIF.")
            return
        frames = []
        vmin = float(jnp.min(self.F_target))
        vmax = float(jnp.max(self.F_target))
        for idx, xk in enumerate(self.x_hist):
            cluster_ab_iter = jnp.array(xk.reshape((-1, self.var_column)))
            F_iter = self.total_flux_vectorized(cluster_ab_iter)
            if save_flux_maps:
                np.savetxt(f"outputs/mma/data/flux_map_iter_{idx+1}.csv", np.asarray(F_iter), delimiter=",")
            # fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
        #     ax.pcolormesh(self.X, self.Y, F_iter, shading='auto', cmap='viridis', vmin=vmin, vmax=vmax)
        #     ax.set_xlabel('Circumference [m]')
        #     ax.set_ylabel('Height [m]')
        #     ax.set_title(f'Flux Optimization Evolution, Iteration {idx+1}')
        #     ax.set_aspect('equal')
        #     fig.canvas.draw()
        #     frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        #     frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        #     frames.append(frame)
        #     plt.close(fig)
        # if hold_last > 0:
        #     for _ in range(hold_last):
        #         frames.append(frames[-1])
        # imageio.mimsave(gif_name, frames, fps=fps, loop=0)
        print(f"GIF saved as {gif_name}")
    
    def plot_optimized_vs_target(self, show_heliostat_positions=True):
        """Plot optimized flux vs target flux side by side"""
        if self.cluster_abr_opt is None:
            print("Run optimize_mma first.")
            return
        
        #print(f"Final Objective: {self.final_mse[0]:.4e}")
        F_opt = self.total_flux_vectorized(self.cluster_abr_opt)
        
        # Denormalize offsets
        a_max = self.receiver_radius * jnp.pi / 2
        b_max = self.receiver_height / 2
        a_offsets = (self.cluster_abr_opt[:, 0] - 0.5) * 2 * a_max
        b_offsets = (self.cluster_abr_opt[:, 1] - 0.5) * 2 * b_max
        
        # Active heliostats: those whose aimed point lies within ±90° of their x_normal.
        # Outside this arc the ±90° visibility mask zeroes out their contribution.
        active_mask = jnp.abs(a_offsets) <= a_max
        active_heliostats = jnp.sum(active_mask)
        print(f"Total number of active heliostats: {int(active_heliostats)} out of total {self.N_total} heliostats")
        
        # Calculate actual heliostat aim positions (x_normal + a, y_normal + b)
        x_positions = (self.x_normal + a_offsets) % self.circumference
        y_positions = self.y_normal + b_offsets
        
        x_active = x_positions[active_mask]
        y_active = y_positions[active_mask]
        
        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Target flux
        im0 = axes[0].pcolormesh(self.X, self.Y, self.F_target, shading='auto', cmap='viridis')
        axes[0].set_xlabel('Circumference [m]')
        axes[0].set_ylabel('Height [m]')
        axes[0].set_title('Target Flux Map (F_target)')
        # axes[0].axis('equal')
        plt.colorbar(im0, ax=axes[0], label='Flux [KW/m^2]')
        
        # Optimized flux
        im1 = axes[1].pcolormesh(self.X, self.Y, F_opt, shading='auto', cmap='viridis')
        if show_heliostat_positions:
            axes[1].plot(x_active, y_active, 'rx', markersize=4, label='Optimized Centers', alpha=0.7)
        axes[1].set_xlabel('Circumference [m]')
        axes[1].set_ylabel('Height [m]')
        axes[1].set_title('Optimized Flux Map (F)')
        # axes[1].axis('equal')
        plt.colorbar(im1, ax=axes[1], label='Flux [KW/m^2]')
        axes[1].legend()
        
        # Add informative suptitle with MSE
        # fig.suptitle(
        #     f'Heliostat Flux Optimization with {self.N_total} heliostats with \n'
        #     f'active heliostats = {int(active_heliostats)} \n'
        #     r'$\mathrm{mean}\left(\!\left(\frac{F - F_{target}}{\max(F_{target})}\right)^{2}\right)$'
        #     f' = {np.round(float(self.final_mse[0]), decimals=4)}',
        #     fontsize=16,
        #     fontweight='bold'
        # )
        plt.tight_layout()
        plt.show()

    def plot_3d_surface(self):
        """Plot target and optimized flux as 3D surfaces."""
        if self.cluster_abr_opt is None:
            print("Run optimize_mma first.")
            return

        F_opt = self.total_flux_vectorized(self.cluster_abr_opt)

        fig = plt.figure(figsize=(16, 8))
        ax = fig.add_subplot(111, projection='3d')
        surf1 = ax.plot_surface(self.X, self.Y, F_opt, cmap='viridis', alpha=0.5, linewidth=0, antialiased=True)
        surf2 = ax.plot_surface(self.X, self.Y, self.F_target, cmap='seismic', alpha=0.5, linewidth=0, antialiased=True)
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('Flux [kW/m²]')
        ax.set_title(f'Heliostat Flux Optimization with {self.N_total} heliostats', fontsize=14, fontweight='bold')
        fig.colorbar(surf2, ax=ax, shrink=0.5, aspect=15, label='Target Flux')
        fig.colorbar(surf1, ax=ax, shrink=0.5, aspect=15, label='Optimized Flux')
        plt.tight_layout()
        plt.show()


    def plot_y_slices(self, n_slices=6):
        """Plot Y-direction slices comparing target and optimized flux."""
        if self.cluster_abr_opt is None:
            print("Run optimize_mma first.")
            return

        F_opt = self.total_flux_vectorized(self.cluster_abr_opt)
        y_indices = np.linspace(0, self.Y.shape[0] - 1, n_slices, dtype=int)
        fig, axes = plt.subplots(n_slices, 1, figsize=(7, 2 * n_slices), sharex=True)
        if n_slices == 1:
            axes = [axes]
        for ax, i in zip(axes, y_indices):
            ax.plot(self.X[i, :], F_opt[i, :], label='Optimized Flux', linewidth=2)
            ax.plot(self.X[i, :], self.F_target[i, :], '--', label='Target Flux', linewidth=2, color='red')
            ax.set_ylabel('Flux [kW/m²]')
            ax.set_title(f'Y = {self.Y[i, 0]:.2f} m')
            ax.grid(True)
        axes[-1].set_xlabel('X [m]')
        axes[0].legend(loc='best')
        plt.tight_layout()
        plt.show()

