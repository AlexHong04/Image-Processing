#!/usr/bin/env python3
"""
Execute a notebook's code cells in order, headlessly, in this folder.

Used to run 00_data_preparation.ipynb without Jupyter. It executes the SAME cells
the notebook contains, so there is no second copy of the logic to drift.

    MPLBACKEND=Agg python3 run_notebook_headless.py 00_data_preparation.ipynb
"""
import json, os, sys, traceback
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg")

nb_path = Path(sys.argv[1]).resolve()
os.chdir(nb_path.parent)

nb = json.loads(nb_path.read_text(encoding="utf-8"))
cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
ns = {"__name__": "__main__", "__file__": str(nb_path)}

print(f"=== running {nb_path.name}: {len(cells)} code cells ===", flush=True)
for i, cell in enumerate(cells, 1):
    src = "".join(cell["source"])
    if not src.strip():
        continue
    print(f"\n----- cell {i}/{len(cells)} -----", flush=True)
    try:
        exec(compile(src, f"<cell {i}>", "exec"), ns)
    except Exception:
        print(f"!!! CELL {i} FAILED", flush=True)
        traceback.print_exc()
        sys.exit(1)
    sys.stdout.flush()
print("\n=== ALL CELLS OK ===", flush=True)
