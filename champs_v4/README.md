# Champs Magnétiques - 3D FDTD Electromagnetic Simulator

This project implements a 3D Finite Difference Time Domain (FDTD) solver using Yee's algorithm for simulating electromagnetic fields. It focuses on macroscopic EM simulations, including solenoid (coil) examples with visualization and animation capabilities.

## Features

- **3D Yee FDTD Solver**: Leap-frog time-stepping for Maxwell's equations.
- **Material Support**: Relative permittivity and conductivity maps.
- **Current Sources**: Macroscopic Jx, Jy, Jz sources (e.g., for coils).
- **Boundary Conditions**: Perfectly Matched Layer (PML) for absorption.
- **Visualization**: 2D slice animations and 3D field plots.
- **Animation Module**: Reusable module for creating MP4 animations from simulation frames.
- **Examples**: Ready-to-run simulations of solenoids and plane waves.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/tt-thoma/champs-magnetiques.git
   cd champs-magnetiques/champs_v4
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   Optional: Install Numba for acceleration:
   ```bash
   pip install numba
   ```

## Quick Start

Run a basic solenoid simulation:
```bash
python examples/run_coil.py
```
This will simulate a solenoid and save a magnetic field snapshot to `results/coil_B_mid_slice.png`.

Run an animated 2D slice:
```bash
python examples/run_coil_slice_anim.py
```

For longer simulations with logging:
```bash
python examples/2D/run_coil_10min.py
```

## Project Structure

- `fdtd_yee_3d.py`: Core Yee3D class implementing the FDTD solver.
- `examples/`: Example scripts for various simulations.
  - `2D/`: 2D slice simulations.
  - `3D/`: 3D simulations.
- `visualization/`: Plotting and animation utilities.
- `results/`: Output directory for images, logs, and animations.
- `tests/`: Unit tests (if available).

## Usage

### Basic Simulation

```python
from fdtd_yee_3d import Yee3D
import numpy as np

# Create a 3D grid
nx, ny, nz = 100, 100, 50
dx = 1e-3  # 1 mm
dt = dx / (3e8 * np.sqrt(3))  # CFL condition
sim = Yee3D(nx, ny, nz, dx, dt)

# Set materials (optional)
epsilon_r = np.ones((nx, ny, nz))
sigma = np.zeros((nx, ny, nz))
sim.set_materials(epsilon_r, sigma)

# Run simulation
for _ in range(1000):
    sim.step()
```

### Adding a Coil

```python
# Approximate a solenoid
center = (nx//2, ny//2)
radius = 20
z0, z1 = 10, 40
turns = 20
current = 1.0
sim.add_coil(center, radius, 'z', turns, current)
```

### Visualization

Use the visualization module:
```python
from visualization.field_slice_anim import animate_slice

# Animate B field on z-slice
animate_slice(sim, field='B', axis='z', out_dir='results/', nframes=100)
```

### Creating Animations from Frames

If you have pre-generated frames, use the animation module:
```python
from visualization.animation_module import create_animation

# Create MP4 from 2D slice frames
create_animation(
    frames_dir='results/simulation/frames',
    output_file='results/simulation/animation.mp4',
    framerate=10,
    mode='slice',
    crf=23
)

# Generate and create 3D animation
create_animation(
    frames_dir='results/3d_frames',
    output_file='results/3d_animation.mp4',
    framerate=10,
    mode='3d',
    sim=my_yee3d_instance,
    field='B',
    nsteps=100,
    frame_interval=5
)
```

## Documentation

See `PROJECT_DOCUMENTATION.md` for detailed API documentation and theory.

## Testing

Run the unit tests:
```bash
python tests/test_yee3d.py
```

Or with pytest (if installed):
```bash
pytest tests/
```

