"""Build `nonreversible_anchored_langevin.ipynb` and the reusable module `nral.py`
from the single source of truth `notebook_source.py` (jupytext "percent" format).

Cell markers
------------
    # %% [markdown] tags=...   -> markdown cell (leading "# " stripped)
    # %% lib                   -> code cell, ALSO exported to nral.py (library code, no side effects)
    # %% run                   -> code cell, driver only (not exported)

Usage:  python make_notebook.py
"""
from __future__ import annotations

import json
import re
import sys
import uuid

SRC = "notebook_source.py"
NB = "nonreversible_anchored_langevin.ipynb"
MODULE = "nral.py"

MARKER = re.compile(r"^# %%(.*)$")


def parse(path: str):
    cells = []
    kind, tag, buf = None, "", []
    for line in open(path).read().splitlines():
        m = MARKER.match(line)
        if m:
            if kind is not None:
                cells.append((kind, tag, buf))
            rest = m.group(1).strip()
            if rest.startswith("[markdown]"):
                kind, tag = "markdown", rest[len("[markdown]"):].strip()
            else:
                kind, tag = "code", rest
            buf = []
        else:
            buf.append(line)
    if kind is not None:
        cells.append((kind, tag, buf))
    return cells


def strip_md(buf):
    out = []
    for line in buf:
        if line.startswith("# "):
            out.append(line[2:])
        elif line.strip() == "#":
            out.append("")
        elif not line.strip():
            out.append("")
        else:
            out.append(line)
    while out and not out[0].strip():
        out.pop(0)
    while out and not out[-1].strip():
        out.pop()
    return out


def trim(buf):
    out = list(buf)
    while out and not out[0].strip():
        out.pop(0)
    while out and not out[-1].strip():
        out.pop()
    return out


def _cell_id(i: int) -> str:
    """Stable per-position cell id (nbformat >= 4.5 requires one)."""
    return f"cell-{i:03d}"


def main() -> int:
    cells = parse(SRC)
    nb_cells, lib_chunks = [], []
    for kind, tag, buf in cells:
        if kind == "markdown":
            body = strip_md(buf)
            if not body:
                continue
            nb_cells.append(dict(cell_type="markdown", id=_cell_id(len(nb_cells)), metadata={},
                                 source=[l + "\n" for l in body[:-1]] + [body[-1]]))
        else:
            body = trim(buf)
            if not body:
                continue
            nb_cells.append(dict(cell_type="code", id=_cell_id(len(nb_cells)), metadata={},
                                 execution_count=None, outputs=[],
                                 source=[l + "\n" for l in body[:-1]] + [body[-1]]))
            if tag.split()[0:1] == ["lib"]:
                lib_chunks.append("\n".join(body))

    nb = dict(
        cells=nb_cells,
        metadata=dict(
            kernelspec=dict(display_name="Python 3", language="python", name="python3"),
            language_info=dict(name="python", version=sys.version.split()[0]),
        ),
        nbformat=4, nbformat_minor=5,
    )
    with open(NB, "w") as fh:
        json.dump(nb, fh, indent=1)

    header = (
        '"""Reusable implementation of non-reversible anchored Langevin with block\n'
        "state-dependent skew-symmetric matrices for constrained Bayesian logistic regression.\n\n"
        "AUTO-GENERATED from notebook_source.py by make_notebook.py -- edit that file, not this one.\n"
        "Contains exactly the library cells of `nonreversible_anchored_langevin.ipynb`, so the\n"
        "notebook and this module can never drift apart. Importing it has no side effects beyond\n"
        "creating the results/figures directories.\n"
        '"""\n\n'
    )
    with open(MODULE, "w") as fh:
        fh.write(header + "\n\n".join(lib_chunks) + "\n")

    n_md = sum(1 for c in nb_cells if c["cell_type"] == "markdown")
    n_code = len(nb_cells) - n_md
    print(f"wrote {NB}: {n_md} markdown + {n_code} code cells")
    print(f"wrote {MODULE}: {len(lib_chunks)} library chunks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
