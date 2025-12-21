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

### 1. Clone the repository
   ```bash
   git clone https://github.com/tt-thoma/champs-magnetiques.git
   cd champs-magnetiques/champs_v4
   ```

### 2. Create a virtual environment (Recommended)
   ```bash
   python -m venv .venv
   ```
   Activate it in windows:
   * Command Prompt (cmd) -- `.venv/Scripts/activate.bat`
   * Powershell -- `.\.venv\Scripts\Activate.ps1`
   
   Activate it in linux:
   * Shell (sh) or Bash -- `source ./.venv/bin/activate`

### 3. Install dependencies
  ```bash
  pip install -r requirements.txt
  ```
  You can also install optional dependencies with:
  ```bash
  pip install -r requirements_opt.txt
  ```

## Quick Start

### Run a basic solenoid simulation
```bash
python examples/run_coil.py
```
This will simulate a solenoid and save a magnetic field snapshot to `results/coil_B_mid_slice.png`.

### Run an animated 2D slice
```bash
python examples/run_coil_slice_anim.py
```

### For longer simulations (with logging)
```bash
python examples/2D/run_coil_10min.py
```

## Project Structure

```
champs-magnetiques/
├── champs_v4/
│   ├── fdtd_yee_3d.py           # Core Yee3D FDTD solver
│   ├── visualization/
│   │   ├── vector_field_viz.py  # Vector field visualization (4 modes)
│   │   ├── field_slice_anim.py  # 2D slice animation utilities
│   │   └── animation_module.py  # MP4 video generation
│   └── results/                 # Output directory (animations, frames, logs)
├── examples/
│   ├── anim_01-05_*.py          # 5 scalar field animations (Ez)
│   ├── anim_01-05_vector_*.py   # 5 vector field animations (H)
│   ├── demo_*.py                # 3 pedagogical demonstrations
│   └── generate_all_10_animations.py  # Main launcher
├── tests/                       # Unit tests
├── archive/
│   └── examples_old/            # Archived test scripts and documentation
├── README.md                    # This file
├── PROJECT_DOCUMENTATION.md     # Detailed API and theory
├── TODO.md                      # Development roadmap
└── requirements.txt             # Python dependencies
```

**Key directories:**
- `champs_v4/`: Core FDTD solver and visualization modules
- `examples/`: Ready-to-run animations and demos (13 total)
- `champs_v4/results/`: Generated animations and frames
- `archive/`: Old/test files (for reference)

## Usage

### Basic Simulation

```python
from champs_v4.fdtd_yee_3d import Yee3D
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
create_animation(frames_dir='results/simulation/frames', output_file='results/simulation/animation.mp4', frame_rate=10,
                 mode='slice', crf=23)

# Generate and create 3D animation
create_animation(frames_dir='results/3d_frames', output_file='results/3d_animation.mp4', frame_rate=10, mode='3d',
                 sim=my_yee3d_instance, field='B', nsteps=100, frame_interval=5)
```

## Electromagnetic Wave Animations

The project includes **13 ready-to-run animations** demonstrating electromagnetic wave phenomena:

### Scalar Field Animations (Ez magnitude)

5 animations showing the electric field magnitude:

1. **Dielectric Refraction** (`examples/anim_01_dielectric_refraction.py`)
   - Air → Glass interface (n=1.0 → 1.5)
   - Demonstrates Snell's law and partial reflection

2. **Metal Reflection** (`examples/anim_02_metal_reflection.py`)
   - Spherical wave hitting copper plate (σ = 5.8×10⁷ S/m)
   - Total reflection, standing wave formation

3. **Lossy Medium Attenuation** (`examples/anim_03_lossy_medium.py`)
   - Air → Absorbing material (σ = 5.0 S/m, εᵣ = 2.5)
   - Exponential amplitude decay

4. **Resonant Cavity** (`examples/anim_04_dielectric_cavity.py`)
   - Rectangular cavity with conducting walls
   - Resonant mode formation (εᵣ = 4.0)

5. **Multilayer Structure** (`examples/anim_05_layered_materials.py`)
   - 4 alternating layers: Air → Glass → Plastic → Air
   - Multiple reflections and interference patterns

### Vector Field Animations (H field vectors)

5 animations showing the magnetic field vectors with **normalized arrows** (uniform length, colored by magnitude):

1. **Vector Refraction** (`examples/anim_01_vector_refraction.py`)
2. **Vector Metal Reflection** (`examples/anim_02_vector_metal.py`)
3. **Vector Lossy Medium** (`examples/anim_03_vector_lossy.py`)
4. **Vector Cavity** (`examples/anim_04_vector_cavity.py`)
5. **Vector Multilayer** (`examples/anim_05_vector_multilayer.py`)

Each vector animation automatically detects TM/TE mode and visualizes the appropriate field (H for TM, E for TE).

### Demo Scripts

3 pedagogical demonstrations:

- **Source Comparison** (`examples/demo_source_comparison.py`): Gaussian pulse vs continuous wave
- **Normalized Vectors** (`examples/demo_normalized_vectors.py`): Shows normalized vector visualization
- **Simple Propagation** (`examples/demo_simple_propagation.py`): Basic wave propagation explanation

### Running Animations

#### Single animation
```bash
python examples/anim_01_dielectric_refraction.py
```

#### All 10 animations (5 scalar + 5 vector)
```bash
python examples/generate_all_10_animations.py
```

Each animation generates:
- **PNG frames** in `champs_v4/results/anim_XX/frames/`
- **MP4 video** (requires ffmpeg) in `champs_v4/results/anim_XX/`

### Vector Visualization Modes

The `VectorFieldVisualizer` class (`champs_v4/visualization/vector_field_viz.py`) supports 4 visualization modes:

1. **Streamlines**: Tangent lines showing field circulation
2. **Quiver (standard)**: Arrows with length proportional to magnitude
3. **Quiver (normalized)**: All arrows same length, colored by magnitude
4. **Hybrid**: Combines streamlines and quiver plots

**Key parameters:**
- `density`: Streamline density (0.5 = sparse, 3.0 = dense)
- `arrow_scale`: Arrow size multiplier (default 3.0 for normalized)
- `field`: 'E', 'H', or 'auto' (auto-detects TM/TE mode)

### Source Types

All animations use **Gaussian pulse sources** for clear wave packet visualization:

```python
t0 = 80 * dt  # Pulse center
spread = 15 * dt  # Pulse width
pulse = np.exp(-((t - t0) / spread) ** 2) * np.sin(omega * t)
```

**Advantages over continuous waves:**
- ✅ Compact wave packet
- ✅ Clear reflection/transmission visualization
- ✅ Minimal interference artifacts
- ✅ Easier to track propagation

### Technical Notes

- **Auto-detection**: Vector animations automatically detect dominant field mode
  - |Ez| > |Exy| → TM mode → visualize H field
  - |Ez| < |Exy| → TE mode → visualize E field
- **2D TM mode**: Ez perpendicular, Hx/Hy in-plane
- **PML boundaries**: 10-cell Perfectly Matched Layers for absorption
- **CFL stability**: dt = dx/(c√2) for 2D simulations
- **Grid resolution**: 0.5-1.0 mm typical (λ/20 minimum for accuracy)

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

<video src="https://raw.githubusercontent.com/tt-thoma/champs-magnetiques/ec6ede8a46403ad3e2bf4dd91e4d598b3ea0f5d3/examples/results/anim_03_lossy/lossy_medium.mp4" />
