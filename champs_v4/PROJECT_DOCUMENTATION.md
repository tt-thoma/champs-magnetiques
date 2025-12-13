# Champs Magnétiques - Project Documentation

## Overview

This project provides a Python implementation of the 3D Finite Difference Time Domain (FDTD) method using Yee's algorithm. FDTD is a numerical technique for solving Maxwell's equations in time and space, commonly used for electromagnetic simulations.

## Theory

### Yee's Algorithm

Yee's scheme staggers electric and magnetic field components in space and time:

- Electric fields (E) are updated at integer time steps.
- Magnetic fields (H) are updated at half-integer time steps.

This leap-frog scheme ensures numerical stability and accuracy.

### Maxwell's Equations

The FDTD method discretizes:

∇ × E = -∂B/∂t  
∇ × H = ∂D/∂t + J  

Where D = ε E, B = μ H.

### CFL Condition

For stability, the time step must satisfy:

dt < dx / (c √3)

Where c is the speed of light.

## Limitations

### General FDTD Yee Limitations

1. **Numerical Dispersion**: Errors in wave propagation speed depending on frequency and direction. Relative error ~ (k dx)^2, where k = 2π/λ.

2. **CFL Condition**: Time step limited by spatial resolution, requiring many iterations for slow phenomena.

3. **Memory and Computation Time**: Scales as O(nx × ny × nz), limiting grid size.

### Low Frequency Specific Issues

1. **Wavelength Resolution**: λ = c / f. For f = 1 Hz, λ ≈ 300,000 km – impossible to simulate directly. The grid must cover only local regions of interest.

2. **PML Absorption**: Standard PML may reflect low-frequency waves. Convolutional PML (CPML) or wider layers can help.

3. **Conductivity Handling**: Semi-implicit scheme for σ is stable but may have errors for very low σ or high frequencies.

4. **Stability and Convergence**: For very low frequencies, reaching steady state requires enormous step counts (e.g., 2e11 steps for 50 Hz).

**Recommendations**: Suitable for macroscopic EM from DC to MHz with dx ~ λ/20. For lower frequencies, simulate local transients or use quasi-static approximations.

## API Reference

### Yee3D Class

#### Constructor

```python
Yee3D(nx, ny, nz, dx, dt, use_numba=False, pml_width=10, pml_sigma_max=1.0)
```

- `nx, ny, nz`: Number of cells in each dimension.
- `dx`: Spatial step (m).
- `dt`: Time step (s).
- `use_numba`: Enable JIT compilation (requires numba).
- `pml_width`: PML boundary thickness.
- `pml_sigma_max`: Maximum PML conductivity.

#### Methods

##### set_materials(epsilon_r, sigma=None)

Set material properties.

- `epsilon_r`: Relative permittivity array (nx, ny, nz).
- `sigma`: Conductivity array (nx, ny, nz).

##### add_coil(center, radius_cells, axis='z', turns=1, current=1.0)

Add a solenoid current source.

- `center`: (ix, iy) center indices.
- `radius_cells`: Radius in cells.
- `axis`: Axis ('x', 'y', 'z').
- `turns`: Number of turns.
- `current`: Current amplitude.

##### step()

Advance simulation by one time step.

## Examples

### Basic Plane Wave

```python
from fdtd_yee_3d import Yee3D
import numpy as np

nx, ny, nz = 100, 100, 1  # 2D slice
dx = 1e-3
dt = dx / (3e8 * np.sqrt(2))
sim = Yee3D(nx, ny, nz, dx, dt)

# Inject pulse
sim.Ez[50, 50, 0] = 1.0

for _ in range(200):
    sim.step()
    print(f"Ez at center: {sim.Ez[50, 50, 0]}")
```

### Solenoid Simulation

See `examples/run_coil.py` for a complete solenoid example.

## Visualization

### 2D Slice Animation

```python
from visualization.field_slice_anim import animate_slice

animate_slice(sim, field='B', axis='z', out_dir='results/', nframes=100)
```

This creates PNG frames and can assemble them into an MP4 using ffmpeg.

### 3D Quiver Plot

Use matplotlib's 3D plotting for full 3D visualization.

## Performance

- For large grids, enable Numba: `use_numba=True`.
- Use appropriate grid sizes; memory scales as O(nx*ny*nz).
- I/O (saving frames) can be slow; consider subsampling.

## Troubleshooting

### Import Errors

Ensure the repository root is in `sys.path` when running examples:

```python
import sys
import os
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _repo_root)
```

### Stability Issues

- Check CFL condition.
- Reduce dt if fields blow up.
- Ensure PML is wide enough.

### Visualization Problems

- For 2D slices, ensure nz=1 for thin simulations.
- Use even frame dimensions for MP4 encoding.

## Future Improvements

- GPU acceleration (CUDA).
- More boundary conditions.
- Parallel processing.
- Advanced materials (dispersive, nonlinear).

