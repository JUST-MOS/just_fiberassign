## Installation

Install directly from GitHub:

```bash
pip install git+https://github.com/yzgu-git/just_fiberassign
```

## Examples

Run the fiber-assignment demonstration with:

```bash
python demo/demo_fiberassign.py
```

Run the catalog-generation demonstration with:

```bash
python demo/demo_catalog.py
```

Run the projected autocorrelation demonstration with:

```bash
python demo/demo_auto.py
```

The demonstrations use the default inputs in:

```text
demo/input/target/
demo/input/tiling/
```

Custom inputs can be specified with:

```bash
python demo/demo_fiberassign.py -target <target_file> -tile <tile_file> -algorithm greedy-classic -dpi 250
python demo/demo_catalog.py -target <target_file> -tile <tile_file> -thread 16 -dpi 250
python demo/demo_auto.py -catalog <catalog_file> -thread 16 -dpi 250
```

Generated products are saved to:

```text
demo/output/
```