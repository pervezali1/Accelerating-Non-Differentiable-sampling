"""Emit self-contained Colab versions: the library is written by the notebook itself."""
import pathlib
import nbformat as nbf

HERE = pathlib.Path("/home/user/Accelerating-Non-Differentiable-sampling/sgld_experiment")
source = (HERE / "anchored_sgld.py").read_text()
assert "%%writefile" not in source

INTRO = """## Self-contained setup (Colab-safe)

This notebook needs `anchored_sgld.py`, which normally sits next to it in the
repository. The cell below **writes that module to the working directory**, so
the notebook runs anywhere — Colab, a fresh Jupyter kernel, a bare container —
with no file upload and no network access.

The content is the library verbatim; if you already have the file next to the
notebook, this simply rewrites the identical text. Everything Colab needs beyond
it (NumPy, SciPy, pandas, scikit-learn, Matplotlib) is preinstalled, so there is
nothing to `pip install`."""

for stem in ("accuracy_curves_only", "accuracy_curves_l1smooth"):
    nb = nbf.read(str(HERE / f"{stem}.ipynb"), as_version=4)
    # Strip any previously injected bootstrap so re-running is idempotent.
    nb.cells = [c for c in nb.cells
                if "%%writefile anchored_sgld.py" not in c.source
                and "Self-contained setup (Colab-safe)" not in c.source]
    bootstrap = nbf.v4.new_code_cell("%%writefile anchored_sgld.py\n" + source)
    nb.cells = [nbf.v4.new_markdown_cell(INTRO), bootstrap] + nb.cells
    out = HERE / f"{stem}_colab.ipynb"
    nbf.write(nb, str(out))
    print(f"wrote {out.name}: {len(nb.cells)} cells, {out.stat().st_size/1024:.0f} KB")
