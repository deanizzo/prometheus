# Initialize cylindrical heliostat flux optimizer
# Design variables: (a, b) for each heliostat (all normalized to [0, 1])
# - a: X-offset from x_normal, [0,1] maps to [-90°, +90°] arc length
# - b: Y-offset from y_normal, [0,1] maps to [-half_height, +half_height]
from optimization.opt_model import HeliostatFluxOptimizer
import numpy as np
import jax.numpy as jnp
import pandas as pd

receiver_radius = 8.825  # meters
receiver_height = 21.6  # meters


problem_name = "ideal_cylinder"
model = HeliostatFluxOptimizer(
    problem_name=problem_name,
    receiver_radius=receiver_radius,
    receiver_height=receiver_height,
    var_column=2,  # (a, b)
    gamma=4.0,
    sample_heliostats=None  # Use all heliostats
)

print(f"Cylindrical Receiver Configuration:")
print(f"  Radius: {receiver_radius:.3f} m (Diameter: {receiver_radius*2:.3f} m)")
print(f"  Height: {receiver_height:.3f} m")
print(f"  Circumference: {model.circumference:.3f} m")
print(f"  Grid size: {model.Ny} x {model.Nx} (derived from target flux)")
print(f"  Total heliostats: {model.N_total}")
print(f"\nDesign variables per heliostat: (a, b) \u2208 [0, 1]")
print(f"  a: [0,1] maps to \u00b1{receiver_radius * np.pi / 2:.2f} m (\u00b190\u00b0)")
print(f"  b: [0,1] maps to \u00b1{receiver_height / 2:.2f} m")



# Initial guess check before MMA optimization
# a=0.5, b=0.5 → no offset for all heliostats (aimed at x_normal, y_normal)

ab_guess = np.zeros((model.N_total, 2), dtype=np.float32)
ab_guess[:, 0] = 0.5  # a = 0.5 → zero x-offset (at x_normal)
ab_guess[:, 1] = 0.5  # b = 0.5 → zero y-offset (at y_normal)

F_guess = model.total_flux_vectorized(jnp.array(ab_guess))
mse_guess = jnp.mean(((F_guess - model.F_target) / model.F_target_max) ** 2)

print("Initial guess (before MMA): a={:.1f}, b={:.1f} for all heliostats".format(ab_guess[0, 0], ab_guess[0, 1]))
print(f"Initial normalized MSE: {float(mse_guess):.6e}")

model.plot_flux_map(F_guess, title='Initial Guess Flux Map (a=0.5, b=0.5)')

# ============================================================
# 3. OPTIMIZATION (MMA)
# ============================================================
x_best, final_mse, cluster_abr_opt = model.optimize_mma(
    seed=42,
    init_rho=0.75,
    n_st=1,
    fTolerance=1e-4,
    gTolerance=1e-2,
    maxIterations=15,
    minIterations=5,
    timeLimitSecs=3600,
    move_limit=0.2,
    kktTol=1e-6,
    verbose=False,
    getGIF=False
)
model.plot_history()



print(cluster_abr_opt)
df = pd.read_csv('./data_dynamic/heliostat_dyn.csv')
df['a_opt'] = [inner[0] for inner in cluster_abr_opt]
df['b_opt'] = [inner[1] for inner in cluster_abr_opt]

df.to_csv('./data_dynamic/heliostat_dyn.csv', index=False)