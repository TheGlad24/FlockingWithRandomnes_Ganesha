import os
os.environ["NUMBA_NUM_THREADS"] = "16"
import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange
from scipy.stats import entropy, gaussian_kde
from matplotlib.animation import FuncAnimation, PillowWriter
from PIL import Image

# ------------ PARAMETERS ------------ #
N = 100
arena_size = 800
dt = 0.02
steps = 10000
flocking_phase = 1000

r_alpha = 50.0
d_alpha = 45.0
r_beta = 100.0
d_beta = 60.0
epsilon = 0.1
h_alpha = 0.2
c1_alpha, c2_alpha = 1.2, 1.0
c1_beta, c2_beta = 10.0, 0.0
c1_gamma, c2_gamma = 0.6, 0.0
v_max = 5.0

goal = np.array([750.0, 300.0])
initial_obstacles = np.array([[400.0, 250.0, 30.0], [400.0, 350.0, 30.0]])
obstacle_speeds = np.random.uniform(-1, 1, (len(initial_obstacles), 2)) * 2.0
models = ['deterministic', 'fokker', 'kramers']
results = {}

@njit
def sigma_1(z):
    return z / np.sqrt(1 + z**2)

@njit
def bump(z_norm, h):
    if z_norm < h:
        return 1.0
    elif z_norm < 1.0:
        return 0.5 * (1 + np.cos(np.pi * (z_norm - h) / (1 - h)))
    return 0.0

# Add this change in your compute_control function:

@njit(parallel=True)
def compute_control(pos, vel, t, goal, obs, model_type):
    new_vel = np.zeros_like(vel)
    for i in prange(pos.shape[0]):
        f_alpha = np.zeros(2)
        align = np.zeros(2)
        for j in range(pos.shape[0]):
            if i == j:
                continue
            diff = pos[j] - pos[i]
            norm = np.sqrt(np.sum(diff**2) + epsilon * np.sum(diff**2)**2)
            if norm < r_alpha:
                phi_val = bump(norm / r_alpha, h_alpha) * (sigma_1(norm - d_alpha) - sigma_1(r_alpha - d_alpha))
                f_alpha += c1_alpha * phi_val * (diff / (np.linalg.norm(diff) + 1e-5))
                align += c2_alpha * bump(norm / r_alpha, h_alpha) * (vel[j] - vel[i])

        f_beta = np.zeros(2)
        for k in range(obs.shape[0]):
            to_obs = pos[i] - obs[k, :2]
            dist = np.linalg.norm(to_obs)
            buffer = obs[k, 2] + 5
            if dist < r_beta:
                f_beta += np.exp(-(dist - buffer)) * (to_obs / (dist + 1e-5))

        # Adaptive goal attraction strength
        if model_type == 'deterministic':
            c1_gamma_effective = 0.6
        else:  # Stronger goal attraction for stochastic models
            c1_gamma_effective = 1.2

        f_gamma = np.zeros(2)
        if t > flocking_phase:
            to_goal = goal - pos[i]
            f_gamma = c1_gamma_effective * to_goal / (np.linalg.norm(to_goal) + 1e-5)

        total = f_alpha + align + c1_beta * f_beta + f_gamma
        new_v = vel[i] + dt * total

        # Reduce noise intensity for better convergence
        if model_type == 'fokker':
            B_xt = 1.0  # Reduced from 1.5
            dW = np.random.normal(0, np.sqrt(dt), size=2)
            new_v += B_xt * dW
        elif model_type == 'kramers':
            B_xt = 1.0  # Reduced from 1.5
            dW = np.random.normal(0, np.sqrt(dt), size=2)
            higher_order_noise = dW + 0.5 * (dW ** 3)
            new_v += B_xt * higher_order_noise

        speed = np.linalg.norm(new_v)
        new_vel[i] = new_v if speed < v_max else new_v / speed * v_max
    return new_vel




def update_obstacles(obs_positions, obs_speeds):
    obs_positions += obs_speeds * dt
    for i in range(len(obs_positions)):
        for d in range(2):
            if obs_positions[i, d] < 0 or obs_positions[i, d] > arena_size:
                obs_speeds[i, d] *= -1
                obs_positions[i, d] = np.clip(obs_positions[i, d], 0, arena_size)
    return obs_positions, obs_speeds
def run_simulation(model_type):
    np.random.seed(42)
    positions = np.random.uniform([50, 100], [150, 200], (N, 2))
    velocities = np.zeros((N, 2))
    obs_positions = initial_obstacles[:, :2].copy()
    obs_speeds = obstacle_speeds.copy()
    trajectory, com_history, dist_to_goal, avg_speeds, entropies, frames = [], [], [], [], [], []

    for t in range(steps):
        obs_positions, obs_speeds = update_obstacles(obs_positions, obs_speeds)
        obstacles_dynamic = np.hstack((obs_positions, initial_obstacles[:, 2:]))
        velocities = compute_control(positions, velocities, t, goal, obstacles_dynamic, model_type)
        positions += velocities * dt
        positions = np.clip(positions, 0, arena_size)
        trajectory.append(positions.copy())

        if t % 50 == 0:
            frames.append((positions.copy(), obstacles_dynamic.copy()))

        com = np.mean(positions, axis=0)
        com_history.append(com)
        dists = np.linalg.norm(positions - goal, axis=1)
        dist_to_goal.append(np.mean(dists))
        avg_speeds.append(np.linalg.norm(velocities, axis=1).mean())

        hist2d, _, _ = np.histogram2d(positions[:, 0], positions[:, 1], bins=20, range=[[0, arena_size], [0, 600]])
        p = hist2d.flatten() / np.sum(hist2d)
        entropies.append(entropy(p + 1e-12))

    return np.array(trajectory), np.array(com_history), np.array(dist_to_goal), np.array(avg_speeds), np.array(entropies), frames

def create_gif(frames, model_name):
    fig, ax = plt.subplots(figsize=(6, 5))
    def update(frame):
        pos, obs = frame
        ax.clear()
        ax.scatter(pos[:, 0], pos[:, 1], c='blue', s=20)
        ax.plot(goal[0], goal[1], 'r*', markersize=15)
        for o in obs:
            circle = plt.Circle(o[:2], o[2], color='red', alpha=0.5)
            ax.add_patch(circle)
        ax.set_xlim(0, arena_size)
        ax.set_ylim(0, 600)
        ax.set_title(f"{model_name} Simulation")
        ax.grid(True)
    ani = FuncAnimation(fig, update, frames=frames, interval=50)
    gif_name = f"{model_name}_simulation.gif"
    ani.save(gif_name, writer=PillowWriter(fps=10))
    plt.close()
    return gif_name

gif_paths = []
for model in models:
    print(f"Running {model} model...")
    traj, com_hist, dist_goal, avg_speed, entropies, frames = run_simulation(model)
    results[model] = {'traj': traj, 'com': com_hist, 'dist_goal': dist_goal,
                      'avg_speed': avg_speed, 'entropy': entropies}
    gif_path = create_gif(frames, model)
    gif_paths.append(gif_path)
time = np.arange(steps) * dt

def plot_and_save(filename, xlabel, ylabel, title, plot_func):
    plt.figure(figsize=(8, 5))
    for model in models:
        plot_func(model)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    plt.savefig(filename)



# 1. Velocity Paths Over Time (for Average Velocity Magnitude)
plt.figure(figsize=(8, 5))
for model in models:
    traj = results[model]['traj']
    velocities = np.diff(traj, axis=0) / dt
    avg_speeds = np.linalg.norm(velocities, axis=2).mean(axis=1)
    plt.plot(time[:-1], avg_speeds, label=model)
plt.title("Average Velocity Magnitude Over Time")
plt.xlabel("Time (s)")
plt.ylabel("Velocity Magnitude")
plt.grid(True)
plt.legend()
plt.savefig("velocity_paths_over_time.png")

# 2. PDF of Final Velocities for Each Model
plt.figure(figsize=(8, 5))
for model in models:
    traj = results[model]['traj']
    final_velocities = (traj[-1] - traj[-2]) / dt
    speeds = np.linalg.norm(final_velocities, axis=1)
    kde = gaussian_kde(speeds)
    x_vals = np.linspace(0, np.max(speeds), 200)
    plt.plot(x_vals, kde(x_vals), label=model)
plt.title("PDF of Final Velocity Magnitudes")
plt.xlabel("Velocity Magnitude")
plt.ylabel("Density")
plt.grid(True)
plt.legend()
plt.savefig("velocity_pdf_final.png")

# 3. Entropy Over Time (Exploration Measure)
plt.figure(figsize=(8, 5))
for model in models:
    plt.plot(time, results[model]['entropy'], label=model)
plt.title("Entropy of Agent Distribution Over Time")
plt.xlabel("Time (s)")
plt.ylabel("Entropy")
plt.grid(True)
plt.legend()
plt.savefig("entropy_over_time.png")

# 4. PDF of Final Distances to Goal
plt.figure(figsize=(8, 5))
for model in models:
    final_pos = results[model]['traj'][-1]
    dists = np.linalg.norm(final_pos - goal, axis=1)
    kde = gaussian_kde(dists)
    x_vals = np.linspace(0, np.max(dists), 200)
    plt.plot(x_vals, kde(x_vals), label=model)
plt.title("PDF of Final Distances to Goal")
plt.xlabel("Distance to Goal")
plt.ylabel("Density")
plt.grid(True)
plt.legend()
plt.savefig("pdf_final_distances.png")

# 5. Success Rate Over Time
plt.figure(figsize=(8, 5))
for model in models:
    traj = results[model]['traj']
    success_series = []
    for pos in traj:
        reached = np.linalg.norm(pos - goal, axis=1) < 40
        success_series.append(np.sum(reached) / N)
    plt.plot(time, success_series, label=model)
plt.title("Success Rate Over Time")
plt.xlabel("Time (s)")
plt.ylabel("Success Rate")
plt.grid(True)
plt.legend()
plt.savefig("success_rate_over_time.png")

# Update Success Rate Calculation to Average Over Last 1000 Steps
print("\n✅ Final Success Rates (Averaged over Last 1000 Steps):")
for model in models:
    traj = results[model]['traj']
    success_counts = []
    for pos in traj[-1000:]:
        reached = np.linalg.norm(pos - goal, axis=1) < 40
        success_counts.append(np.sum(reached) / N)
    avg_success_rate = np.mean(success_counts) * 100
    print(f"{model.capitalize()} Model Success Rate: {avg_success_rate:.2f}%")
plt.figure(figsize=(8, 5))
for model in models:
    final_pos = results[model]['traj'][-1]
    dists = np.linalg.norm(final_pos - goal, axis=1)
    kde = gaussian_kde(dists)
    x_vals = np.linspace(0, np.max(dists), 200)
    plt.plot(x_vals, kde(x_vals), label=model)
plt.title("PDF of Final Distances to Goal")
plt.xlabel("Distance to Goal")
plt.ylabel("Density")
plt.grid(True)
plt.legend()
plt.savefig("pdf_final_distances.png")
plt.figure(figsize=(8, 5))
for model in models:
    traj = results[model]['traj']
    final_velocities = (traj[-1] - traj[-2]) / dt
    speeds = np.linalg.norm(final_velocities, axis=1)
    kde = gaussian_kde(speeds)
    x_vals = np.linspace(0, np.max(speeds), 200)
    plt.plot(x_vals, kde(x_vals), label=model)
plt.title("PDF of Final Velocity Magnitudes")
plt.xlabel("Velocity Magnitude")
plt.ylabel("Density")
plt.grid(True)
plt.legend()
plt.savefig("pdf_final_velocities.png")
plt.figure(figsize=(8, 5))
for model in models:
    entropy_vals = results[model]['entropy']
    kde = gaussian_kde(entropy_vals)
    x_vals = np.linspace(0, np.max(entropy_vals), 200)
    plt.plot(x_vals, kde(x_vals), label=model)
plt.title("PDF of Entropy Across Time")
plt.xlabel("Entropy")
plt.ylabel("Density")
plt.grid(True)
plt.legend()
plt.savefig("pdf_entropy.png")
plt.figure(figsize=(8, 5))
for model in models:
    traj = results[model]['traj']
    success_series = []
    for pos in traj:
        reached = np.linalg.norm(pos - goal, axis=1) < 40
        success_series.append(np.sum(reached) / N)
    kde = gaussian_kde(success_series)
    x_vals = np.linspace(0, 1, 200)
    plt.plot(x_vals, kde(x_vals), label=model)
plt.title("PDF of Success Rate Over Time")
plt.xlabel("Success Rate")
plt.ylabel("Density")
plt.grid(True)
plt.legend()
plt.savefig("pdf_success_rate.png")
plt.figure(figsize=(8, 5))
for model in models:
    traj = results[model]['traj']
    neighbor_counts = []
    for pos in traj:
        count = np.sum(np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1) < r_alpha, axis=1) - 1
        neighbor_counts.append(np.mean(count))
    kde = gaussian_kde(neighbor_counts)
    x_vals = np.linspace(0, N, 200)
    plt.plot(x_vals, kde(x_vals), label=model)
plt.title("PDF of Average Number of Neighbors")
plt.xlabel("Average Neighbors")
plt.ylabel("Density")
plt.grid(True)
plt.legend()
plt.savefig("pdf_agent_density.png")
from scipy.spatial.distance import pdist

plt.figure(figsize=(8, 5))
for model in models:
    final_pos = results[model]['traj'][-1]
    pairwise_dists = pdist(final_pos)
    kde = gaussian_kde(pairwise_dists)
    x_vals = np.linspace(0, np.max(pairwise_dists), 200)
    plt.plot(x_vals, kde(x_vals), label=model)
plt.title("PDF of Final Inter-Agent Distances")
plt.xlabel("Inter-Agent Distance")
plt.ylabel("Density")
plt.grid(True)
plt.legend()
plt.savefig("pdf_inter_agent_distances.png")
# ------------ Save Final Data for Analysis (Without Plotting) ------------ #

# 1. Save Final Velocity Magnitudes Data
final_velocity_data = {}
for model in models:
    traj = results[model]['traj']
    final_velocities = (traj[-1] - traj[-2]) / dt
    speeds = np.linalg.norm(final_velocities, axis=1)
    final_velocity_data[model] = speeds

np.savez("final_velocity_magnitudes.npz", **final_velocity_data)
print("✅ Final Velocity Magnitudes Data Saved: final_velocity_magnitudes.npz")

# 2. Save Entropy Over Time Data
entropy_data = {}
for model in models:
    entropy_vals = results[model]['entropy']
    entropy_data[model] = entropy_vals

np.savez("entropy_over_time.npz", **entropy_data)
print("✅ Entropy Over Time Data Saved: entropy_over_time.npz")
