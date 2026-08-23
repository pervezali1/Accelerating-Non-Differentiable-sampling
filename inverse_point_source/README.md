# Inverse point-source reconstruction with an RBF-DeepONet

Two point sources separated by less than half a wavelength, recovered from
Cauchy data on a measurement circle.

| file | what it is |
|---|---|
| `point_source_two_sources_improved.ipynb` | the revised pipeline, runnable top to bottom |
| `RESULTS.md` | the measurements behind every change |

## The short version

The original pipeline trains to a weighted MSE of `2.14e-02`. Predicting zero
everywhere scores `8.65e-02`, and the **best score any branch network could
achieve with the original RBF trunk is `1.24e-02`** — so most of the remaining
loss is the trunk's representational floor, not the network. That floor also
caps the peak of the reconstructed indicator at 0.763, which is why predictions
top out around 0.59 and why two well-separated sources come back as one smeared
lobe.

Meanwhile the measurement itself is nowhere near its limit: at `λ/4` separation,
1 % noise supports a localisation error of 0.009, roughly three times finer than
one grid cell. The accuracy is being lost in the representation, the training and
the peak picking — all three fixable.

## What changed

Tagged in the notebook as **[FIX]** (correctness), **[ACC]** (accuracy) and
**[DIAG]** (new diagnostic).

**Correctness**
- `CosineAnnealingLR(T_max=300)` with `num_epochs=100` — the learning rate never
  annealed. Now tied to `NUM_EPOCHS`.
- The RBF overlap parameter `s` was shadowed by a loop variable holding source
  coordinates. All configuration moved to one cell with uppercase names.
- Input standardisation is registered as buffers *inside* the model, so it is
  applied at inference too. Applied outside, every "new test data" figure would
  have been silently wrong.
- `Adam(weight_decay=…)` → `AdamW`.
- Standardisation statistics are computed on the training split only.

**Accuracy**
- Finer RBF trunk (`h = λ/12`): floor `1.24e-02` → `2.39e-03`, and the trained
  loss `1.57e-02` → `1.01e-02` with the activation held fixed. Measured; `λ/16`
  lowers the floor another 10x but scores slightly *worse* at this budget.
- Input standardisation: measured `2.23e-02` → `1.58e-02` in a matched A/B.
- Exact rotation augmentation — rotating the sources equals `np.roll` of the
  data to 6e-14, so every sample stands in for 256.
- Gradient clipping, early stopping, optional soft-Dice term, optional noise.

**Diagnostics** (these are the point of the rewrite)
- Forward-operator verification: finite-difference, Helmholtz residual,
  Sommerfeld condition.
- **Representational floor** of the trunk, via non-negative least squares. Run it
  before blaming the network for anything.
- Information budget of the measurement: how redundant `∂u/∂n` is, and what
  localisation error a given noise level supports.
- A three-panel split of the blur into "trunk" and "network" contributions.
- Localisation error against separation, which is the number worth reporting for
  this problem — not the training loss.

**Peak detection** — `peak_threshold = 0.60` / `neighborhood_size = 5` replaced
by topographic prominence, physical-units non-maximum suppression and sub-grid
parabolic refinement, with the cutoff calibrated on the validation split rather
than guessed. Correct source count on a 200-field benchmark: 85.5 % → 95.5 %.

## Running it

The notebook regenerates `point_source_data_2/cauchy_dataset_two_sources_close.npz`
if it is missing and reuses it otherwise. Cell 1 holds every knob. The trunk is
larger than before (`P = 400` against 169), so the branch network grows from
~0.7 M to ~2.5 M parameters — slower per epoch, and worth a GPU.

If memory is tight, drop `Y_train` and rely on `targets_from_sources` to rebuild
targets per batch; with `ROT_AUGMENT = True` they are rebuilt anyway.
