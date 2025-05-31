# Stochastic Multi-Agent Flocking Simulation

This project implements and analyzes a multi-agent flocking system based on **Olfati-Saber's Flocking Algorithm 3**, enhanced with **stochastic models** to simulate more realistic, real-world agent behavior. The agents navigate toward a fixed goal while avoiding static and dynamic obstacles using bio-inspired local interaction rules.

## 🧠 Key Features

- **Three models implemented**:
  - **Deterministic**
  - **Fokker–Planck-based** stochastic noise
  - **Kramers–Moyal higher-order stochastic noise**

- **Simulation Metrics**:
  - Success rate over time
  - Entropy of agent distribution
  - Velocity evolution
  - Distance to goal
  - Inter-agent spacing
  - PDF plots for all major metrics

- **GIF Animations**:
  - Simulations for each model are rendered and saved as `.gif` files for visual comparison.

## 📁 Project Structure

.
├── FlockingRandom_Ganesha.py # Main simulation script
├── Outputs/
│ ├── sample_gif.gif # Generated simulation GIFs
│ ├── velocity_paths_over_time.png # Average velocity over time
│ ├── pdf_final_distances.png # Distance to goal PDF
│ ├── pdf_entropy.png # Entropy PDF
│ ├── success_rate_over_time.png # Success rate over time
│ └── ... # Other saved plots and diagrams
├── final_velocity_magnitudes.npz # Saved data arrays
├── entropy_over_time.npz

markdown
Copy
Edit

## 📊 Graphs Generated

- **Velocity vs. Time**
- **PDF of Final Velocities**
- **PDF of Distance to Goal**
- **PDF of Agent Entropy**
- **PDF of Agent Success Rate**
- **PDF of Agent Density**
- **PDF of Inter-Agent Distances**

## 🚀 How to Run

1. Install the dependencies:
   ```bash
   pip install numpy matplotlib scipy numba pillow
Run the main script:

bash
Copy
Edit
python FlockingRandom_Ganesha.py
Check the Outputs/ folder for generated plots and .gif animations.

📌 Simulation Highlights
Each model was evaluated over 10,000 time steps with 100 agents in a bounded arena.

Obstacles are dynamic and randomized for robust testing.

Success rates are calculated over the final 1,000 steps.

PDF-based metrics allow probabilistic comparison across models.

🔒 Real-World Application
This simulation mimics real-world robotic swarms operating in noisy environments — making it ideal for applications in:

Disaster search and rescue

Decentralized drone navigation

Bio-inspired AI and collective robotics

