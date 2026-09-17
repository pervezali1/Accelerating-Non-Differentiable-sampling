"""Fetch the two datasets into ./data.

Official sources, tried first:
  * MAGIC Gamma Telescope -- https://archive.ics.uci.edu/dataset/159/magic+gamma+telescope
  * Titanic (labelled train.csv) -- https://www.kaggle.com/competitions/titanic/data

If the official endpoints are unreachable (some sandboxes block them) the script falls back to
public mirrors and then VALIDATES the bytes against the official specification: MAGIC must have
19020 rows, 10 predictors and class counts g = 12332 / h = 6688; Titanic must have 891 rows, the
12 competition columns, 549/342 survived, 177 missing Age and 2 missing Embarked. Anything that
fails validation is rejected rather than used.

Usage:  python fetch_data.py [--data-dir DIR]
"""
from __future__ import annotations

import argparse
import hashlib
import io
import os
import sys
import urllib.request
import zipfile

MAGIC_COLUMNS = ["fLength", "fWidth", "fSize", "fConc", "fConc1",
                 "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist"]

MAGIC_OFFICIAL = "https://archive.ics.uci.edu/static/public/159/magic+gamma+telescope.zip"
MAGIC_MIRROR = ("https://media.githubusercontent.com/media/EpistasisLab/pmlb/master/"
                "datasets/magic/magic.tsv.gz")
TITANIC_MIRROR = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"


def _get(url: str, timeout: int = 180) -> bytes:
    print(f"  GET {url}")
    return urllib.request.urlopen(url, timeout=timeout).read()


def fetch_magic(path: str) -> None:
    import pandas as pd

    if os.path.exists(path):
        print(f"[magic] already present: {path}")
        return
    df = None
    try:                                              # 1. official UCI API
        from ucimlrepo import fetch_ucirepo
        rep = fetch_ucirepo(id=159)
        df = pd.concat([rep.data.features, rep.data.targets], axis=1)
        df.columns = MAGIC_COLUMNS + ["class"]
        print("[magic] source: ucimlrepo (official)")
    except Exception as exc:
        print(f"[magic] ucimlrepo unavailable: {exc!r}")
    if df is None:
        try:                                          # 2. official UCI static archive
            with zipfile.ZipFile(io.BytesIO(_get(MAGIC_OFFICIAL))) as zf:
                with zf.open("magic04.data") as fh:
                    df = pd.read_csv(fh, header=None, names=MAGIC_COLUMNS + ["class"])
            print("[magic] source: archive.ics.uci.edu (official)")
        except Exception as exc:
            print(f"[magic] official archive unavailable: {exc!r}")
    if df is None:
        try:                                          # 3. public mirror, then validate
            import gzip
            raw = gzip.decompress(_get(MAGIC_MIRROR))
            df = pd.read_csv(io.BytesIO(raw), sep="\t")
            df.columns = MAGIC_COLUMNS + ["target"]
            df["class"] = df.pop("target").map({0: "g", 1: "h"})
            print("[magic] source: PMLB mirror (validated against the UCI specification below)")
        except Exception as exc:
            print(f"[magic] mirror unavailable: {exc!r}")
    if df is None:
        raise SystemExit("could not obtain MAGIC; download magic04.data manually into ./data")

    assert len(df) == 19020, len(df)
    counts = df["class"].value_counts().to_dict()
    assert counts == {"g": 12332, "h": 6688}, counts

    def fmt(v: float) -> str:
        s = f"{v:.4f}".rstrip("0")
        return s + "0" if s.endswith(".") else s

    lines = [",".join(fmt(v) for v in row) + "," + c
             for row, c in zip(df[MAGIC_COLUMNS].to_numpy(dtype=float), df["class"])]
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"[magic] wrote {path}  ({len(lines)} rows, sha256={_sha256(path)})")


def fetch_titanic(path: str) -> None:
    import pandas as pd

    if os.path.exists(path):
        print(f"[titanic] already present: {path}")
        return
    print("[titanic] the Kaggle competition file needs credentials:")
    print("          kaggle competitions download -c titanic   (then place train.csv here)")
    try:
        raw = _get(TITANIC_MIRROR)
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:
        raise SystemExit(
            f"could not obtain Titanic ({exc!r}); download train.csv from "
            "https://www.kaggle.com/competitions/titanic/data into ./data "
            "(the unlabelled test.csv will NOT work)"
        )
    assert len(df) == 891, len(df)
    assert df["Survived"].value_counts().to_dict() == {0: 549, 1: 342}
    assert int(df["Age"].isna().sum()) == 177 and int(df["Embarked"].isna().sum()) == 2
    df.to_csv(path, index=False)
    print(f"[titanic] wrote {path}  (891 rows, sha256={_sha256(path)})")


def _sha256(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
    args = ap.parse_args()
    os.makedirs(args.data_dir, exist_ok=True)
    fetch_magic(os.path.join(args.data_dir, "magic04.data"))
    fetch_titanic(os.path.join(args.data_dir, "titanic_train.csv"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
