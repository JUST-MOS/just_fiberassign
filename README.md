## Installation

Install directly from GitHub:

```bash
pip install git+https://github.com/JUST-MOS/just_fiberassign
```

## Usage

The main functions can be imported directly from `just_fiberassign`:

```python
from just_fiberassign import fiberassign, load_focalplane, load_target, load_tile, show_focalplane

# Load the JUST focal plane
fiber_coord = load_focalplane('just')

# Load tiles: pass, (RA, Dec) [deg], rotation [deg]
tile_pass, tile_coord, tile_rotation = load_tile('tile.npz')

# Load targets: (RA, Dec) [deg], priority, subpriority, redshift, magnitude
target_coord, target_pri, target_sub, target_z, target_mr = load_target('target.npz')

# Assign fibers: higher priority values are assigned first
assigned, patrolled = fiberassign(
    fiber_coord, tile_coord, tile_rotation, target_coord, target_pri, target_sub,
    algorithm = 'greedy-classic', r_patrol = 6.0, r_exclude = 1.6, telescope = 'just'
)
```

Target and tile files can be local `.npz` files or HTTP(S) URLs.

Available assignment algorithms are `greedy-classic`, `greedy-global`, `milp`, and `cp-sat` (`greedy` is an alias for `greedy-classic`).

A custom focal plane can be loaded with:

```python
# Load /path/to/data/custom_focalplane.csv
fiber_coord = load_focalplane('custom', spec_dir = '/path/to/data')
```

The focal-plane layout can be displayed with:

```python
# Display the default JUST focal plane
show_focalplane()

# Customize text and logo
show_focalplane(
    text = 'JUST', text_scale = 0.9, text_dx = 0.0, text_dy = 0.0, text_buffer = 3.0,
    font = 'STIXGeneral', weight = 1000, text_ec = '0.75', text_fc = '0.75',
    logo = True, logo_scale = 0.7, logo_dx = 0.0, logo_dy = 180.0, logo_buffer = 1.0,
    logo_ec = '0.75', logo_fc = '#A71E2D', ec = '0.75', fc = 'none'
)

# Save with custom output settings
show_focalplane(
    dpi = 250, transparent = False, save_path = '/path/to/output',
    logo_path = '/path/to/logo.png', focalplane_path = '/path/to/focalplane.csv'
)
```

## Examples

The demonstrations use the default inputs in:

```text
examples/input/target/
examples/input/tiling/
```

Run the fiber-assignment and catalog demonstration with:

```bash
# Default fiber assignment
python examples/demo_fiberassign.py

# Custom target, tiling, algorithm, random density, workers, and resolution
python examples/demo_fiberassign.py \
    -target <target_file> -tile <tile_file> -algorithm greedy-classic \
    -n_random 5000 -thread 16 -dpi 250

# Progressively update the figure with focal-plane loading indicators
python examples/demo_fiberassign.py -fancy
```

Run the projected-correlation-function validation with:

```bash
# Use the default generated catalog
python examples/demo_wp_validation.py

# Custom catalog, workers, and resolution
python examples/demo_wp_validation.py -catalog <catalog_file> -thread 16 -dpi 250
```

Generated catalogs and figures are saved to:

```text
examples/output/
```