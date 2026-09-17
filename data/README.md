# Data directory

Place the UCI **MAGIC Gamma Telescope** file here:

```
data/magic04.data
```

Download it from
<https://archive.ics.uci.edu/dataset/159/magic+gamma+telescope>
(the archive contains `magic04.data` and `magic04.names`).

The file is comma separated with no header and has 19 020 rows and 11 columns:

| column | name | meaning |
|--------|------|---------|
| 1  | `fLength`  | major axis of the ellipse (mm) |
| 2  | `fWidth`   | minor axis of the ellipse (mm) |
| 3  | `fSize`    | 10-log of the sum of the content of all pixels |
| 4  | `fConc`    | ratio of the sum of the two highest pixels over `fSize` |
| 5  | `fConc1`   | ratio of the highest pixel over `fSize` |
| 6  | `fAsym`    | distance from the highest pixel to the centre, projected onto the major axis |
| 7  | `fM3Long`  | third root of the third moment along the major axis |
| 8  | `fM3Trans` | third root of the third moment along the minor axis |
| 9  | `fAlpha`   | angle of the major axis with the vector to the origin |
| 10 | `fDist`    | distance from the origin to the centre of the ellipse |
| 11 | `class`    | `g` = gamma (signal), `h` = hadron (background) |

The code encodes `g -> 1` and `h -> 0`, and never substitutes any other data
set silently: if the file is missing, `pnral.data.load_raw_magic` raises with
this path in the message.  The `--synthetic` flag generates a clearly-labelled
surrogate of the same shape, which exists **only** to exercise the code path
and supports no scientific claim.

The data file itself is not committed (see `.gitignore`).
