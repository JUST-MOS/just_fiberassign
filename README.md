## Installation

Install directly from GitHub:

```bash
pip install git+https://github.com/yzgu-git/just_fiberassign
```

## Usage

The main functions can be imported directly from `just_fiberassign`:

```python
from just_fiberassign import fiberassign, load_focalplane, load_target, load_tile, show_focalplane

fiber_coord = load_focalplane('just')
tile_pass, tile_coord, tile_rotation = load_tile('tile.npz')
target_coord, target_pri, target_sub, target_z, target_mr = load_target('target.npz')
assigned, patrolled = fiberassign(fiber_coord, tile_coord, tile_rotation, target_coord, target_pri, target_sub, algorithm = 'greedy-classic', r_patrol = 6.0, r_exclude = 1.6, telescope = 'just')
```

`fiber_coord` contains focal-plane `(x, y)` coordinates in mm. `target_coord` and `tile_coord` contain `(RA, Dec)` coordinates in degrees, and `tile_rotation` is also in degrees. Larger `target_pri` values have higher priority, with `target_sub` used as the secondary priority.

`algorithm` can be `greedy-classic`, `greedy-global`, `milp`, or `cp-sat`; `greedy` is an alias for `greedy-classic`. `r_patrol` is the fiber patrol radius in mm, and `r_exclude` is the minimum allowed target separation in the focal plane.

Target files are `.npz` files containing `target_coord`, `target_pri`, `target_sub`, `target_z`, and `target_mr`. Local paths and HTTP(S) URLs are supported. Tile files contain `tile_pass`, `tile_coord`, and `tile_rotation`.

A custom focal plane can be loaded with:

```python
fiber_coord = load_focalplane('name', spec_dir = '/path/to/data')
```

which reads `/path/to/data/name_focalplane.csv`.

The focal-plane layout can be displayed with:

```python
show_focalplane(text = None, text_scale = 0.9, text_dx = 0.0, text_dy = 0.0, text_buffer = 3.0, font = 'STIXGeneral', weight = 1000, text_ec = '0.75', text_fc = '0.75', logo = True, logo_scale = 0.7, logo_dx = 0.0, logo_dy = 180.0, logo_buffer = 1.0, logo_ec = '0.75', logo_fc = '#A71E2D', ec = '0.75', fc = 'none', dpi = 250, transparent = False, save_path = None, logo_path = ..., focalplane_path = ...)
```

`text_scale`, `text_dx`, `text_dy`, and `text_buffer` control the text size, position, and spacing; `font` and `weight` set the typeface, while `text_ec` and `text_fc` set the text edge and face colors. The equivalent `logo_*` arguments control the logo. `ec` and `fc` set the edge and face colors of the remaining fibers. `dpi` sets the saved resolution, `transparent` controls the saved background, `save_path` sets the output directory, and `logo_path` and `focalplane_path` replace the default input files.

## Examples

Run the demonstrations with:

```bash
python demo/demo_fiberassign.py
python demo/demo_catalog.py
python demo/demo_auto.py
```

The demonstrations use the default inputs in:

```text
demo/input/target/
demo/input/tiling/
```

Custom settings can be specified with:

```bash
python demo/demo_fiberassign.py -target <target_file> -tile <tile_file> -algorithm greedy-classic -dpi 250
python demo/demo_catalog.py -target <target_file> -tile <tile_file> -thread 16 -dpi 250
python demo/demo_auto.py -catalog <catalog_file> -thread 16 -dpi 250
```

`-thread` sets the number of parallel workers and `-dpi` sets the output resolution. Generated products are saved to `demo/output/`.