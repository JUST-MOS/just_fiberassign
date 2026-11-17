## Installation

Install the package from GitHub:

```bash
pip install "just_fiberassign @ git+https://github.com/JUST-MOS/just_fiberassign.git"
```

To include the optional visualization dependencies (`matplotlib`, `healpy`, and `basemap`):

```bash
pip install "just_fiberassign[plot] @ git+https://github.com/JUST-MOS/just_fiberassign.git"
```

## Usage

Import `fiberassign` from `just_fiberassign` to run target-to-fiber assignment:

```python
from just_fiberassign import fiberassign

# Assign fibers: higher priority values are assigned first
target_assigned, target_patrolled = fiberassign(
    tile_coord, target_coord, target_pri,
    target_sub = None, tile_rotation = tile_rotation,
    algorithm = 'greedy-global', r_patrol = 6.0, r_exclude = 1.6, telescope = 'just'
)
```

`target_assigned` and `target_patrolled` are Boolean masks over the input targets. The patrol radius (`r_patrol`) and exclusion radius (`r_exclude`) are in millimeters. The available assignment algorithms are `greedy-classic`, `greedy-global` (default), `milp`, and `cp-sat`; `greedy` is an alias for `greedy-classic`. If omitted, `target_sub` is generated with a fixed random seed and `tile_rotation` defaults to zero.

Use `gen_catalog`, `gen_random`, and `gen_wang` to generate survey catalogues, random catalogues, and Wang angular corrections:

```python
from just_fiberassign import gen_catalog, gen_random, gen_wang

# Generate assignment masks, IIP weights, and PIP bitweights using 128 alternative subpriorities
assigned_mask, iip_weight, pip_bitweights, patrolled_mask = gen_catalog(
    fiber_coord, tile_coord, tile_rotation, target_coord, target_pri, target_sub,
    algorithm = 'greedy-global', r_patrol = 6.0, r_exclude = 1.6, telescope = 'just', thread = 16
)

# Generate random sky positions and sample their redshifts from the target catalogue
random_coord, random_z, random_patrolled = gen_random(
    fiber_coord, tile_coord, tile_rotation, target_z,
    density = 5000, tile_radius = 0.6, r_patrol = 6.0, telescope = 'just', seed = 0
)

# Calculate angular DD and DR correction factors
theta, wang_dd, wang_dr = gen_wang(
    target_coord[patrolled_mask], target_coord[assigned_mask],
    iip_weight[assigned_mask], pip_bitweights[assigned_mask], random_coord[random_patrolled], thread = 16
)
```

`gen_catalog()` returns Boolean masks and weights aligned with the input target catalogue. `gen_random()` generates random sky positions with density in deg^-2 and `tile_radius` in degrees, returning a separate patrol mask (`random_patrolled`) for the random catalogue. `gen_wang()` returns angular bin centres (`theta`, in degrees) and the corresponding DD and DR correction factors; the optional `theta_edges` argument specifies the angular bin edges in degrees.

With the `plot` dependencies installed, import the visualization functions from `just_fiberassign.visualize`:

```python
import matplotlib.pyplot as plt

from just_fiberassign.visualize import plot_focalplane, plot_coverage, plot_complete

# Plot the JUST focal-plane layout
plt.figure(figsize = (5, 5))
plot_focalplane(text = 'JUST', logo = True)
plt.show()

# Map the number of tiles that can patrol each sky position
plt.figure(figsize = (5, 5))
plot_coverage(tile_coord, 'JUST coverage', nside = 4096, nxy = 2001)
plt.show()

# Map the fraction of targets assigned in each HEALPix pixel
plt.figure(figsize = (5, 5))
plot_complete(target_coord, target_assigned, 'JUST completeness', nside = 1024, nxy = 2001)
plt.show()
```

All three visualization functions accept an optional `ax` argument to draw on an existing Matplotlib axes and return the axes object. None of the functions call `plt.show()` or `plt.savefig()` internally. `plot_focalplane()` also accepts custom text, logo, and focal-plane file paths. `plot_coverage()` and `plot_complete()` use a Lambert azimuthal equal-area projection from 0 to 90 degrees in RA and Dec; larger `nside` or `nxy` values increase computation time and memory usage.
