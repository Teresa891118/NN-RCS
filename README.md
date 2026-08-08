# NN-RCS

**Flexible modelling of secular trends in disease rates: a neural network with restricted cubic spline activation**

Give it a series of age-standardised incidence rates together with the variance
of their log-transformed values. It returns one smooth fitted trend that you can
read off at **any** time point — not only at the years you observed.

---

## Why this method

Long-term trends in disease rates are usually described with a log-linear model,
a polynomial, a restricted cubic spline with knots you have to place yourself,
or joinpoint regression. Each forces a shape on the data: a constant annual
percentage change, a fixed polynomial degree, knots at percentiles you chose, or
abrupt changes in slope at a small number of estimated breakpoints.

NN-RCS replaces the fixed basis with a small neural network. Its activation
function is a **three-knot restricted cubic spline**, so:

- the fitted curve has a **continuous second derivative** everywhere — no kinks;
- the curve is **linear beyond the outer knots**, which keeps it well behaved at
  the two ends of the series, where flexible methods usually misbehave;
- **you do not have to place any knots.** The network learns where the curve
  needs to bend.

The fitted trend is a smooth function of time, so once it is fitted you can
evaluate it wherever you like: at 1990, at 1990.5, or at 1990.01.

---

## Quick start

### Option A — run the built-in example (no data needed)

```python
DATA_SOURCE = 'demo_male'      # or 'demo_female'
```

```bash
python nn_rcs.py
```

Two demonstration series are built into the file: oesophageal cancer incidence
in Taiwan, men and women, 1980–2023.

### Option B — Google Colab, upload your own file

```python
DATA_SOURCE = 'upload'
```

Paste the whole file into a notebook cell and run the cell. A **Choose Files**
button appears.

> The button will **not** appear if you launch the file with `!python`, because
> it needs the notebook itself. Paste the code into a cell instead.

### Option C — your own machine

```python
DATA_SOURCE = 'file'
INPUT_CSV   = 'my_data.csv'
```

```bash
pip install numpy pandas matplotlib tensorflow
python nn_rcs.py
```

### Not sure about the format?

```python
WRITE_TEMPLATE = True
```

writes a blank `input_template.csv` in the right shape. Fill it in, set the flag
back to `False`, and run again.

---

## Input format

A file with a header row. Accepted formats: `.csv`, `.txt`, `.xlsx`, `.xls`.
Text encodings are tried in turn (UTF-8, UTF-8-BOM, Big5, CP1252, Latin-1), so a
file exported from software using a regional code page still opens.

| Column         | Required | Meaning |
|----------------|----------|---------|
| `asir`         | yes      | The age-standardised incidence rate itself, **not logged**. Must be > 0 |
| `var_log_asir` | yes      | Variance of the **log** of that rate. Must be > 0 |
| `time`         | no       | Time point. If absent, generated from `TIME_START` and `TIME_STEP` |

```csv
time,asir,var_log_asir
1980,6.503659,0.00254975
1981,6.739920,0.00275820
1982,5.821879,0.00294383
...
```

Column names are configurable (`COL_RATE`, `COL_VAR`, `COL_TIME`) if your
headers differ.

### Why the rate is not logged but the variance is

You already have the rate: it is what you report. What the model needs on the
log scale is the weighting, and the variance of a log rate is **not** obtained by
taking the log of a variance — it comes from the delta method. Asking for the
rate as you have it, and for the one derived quantity that genuinely has to be
computed, keeps the work on your side to a minimum. **The program takes the
logarithm for you.**

### `log` means the natural logarithm

Base e, `ln`, **not** log base 10, everywhere in this program and in the
paper.

### You do not tell the program how long your series is

The number of time points is read from the file. The fitted grid always spans
exactly your observed period. A 44-year series, a 19-year series and a 60-week
series all work with no changes.

| Your data | Span | Grid at `OUTPUT_STEP = 0.1` |
|---|---|---|
| Annual, 1980–2023 | 43 | 431 points, 1980.0 … 2023.0 |
| Annual, 2001–2020 | 19 | 191 points, 2001.0 … 2020.0 |
| Weekly, weeks 1–60 | 59 | 591 points, 1.0 … 60.0 |
| Quarterly, 2010.00–2019.75 | 9.75 | 99 points, 2010.00 … 2019.75 |

### Why the program does not age-standardise for you

Age standardisation depends on choices that are yours, not ours: which standard
population (WHO 2000, Segi, US 2000, …), whether the age range is truncated
(all ages, 20+, 40+, …), and how many age groups you use. Building it in would
force you into our choices, and would break for anyone who needs a different
one.

Computing an age-standardised rate and the variance of its logarithm is a
standard calculation. From Poisson counts:

```
ASIR_i      = Σ_a  w_a · (d_ai / n_ai)          w_a = standard population weights
Var(ASIR_i) = Σ_a  w_a² · d_ai / n_ai²
var_i       = Var(ASIR_i) / ASIR_i²             (delta method)
```

where `d_ai` and `n_ai` are the case count and the person-time in age group `a`
at time point `i`. You supply `ASIR_i` and `var_i`.

Nothing in the method is specific to incidence rates. Any quantity that is
positive and has a variance at each time point will work.

---

## Reading the curve off at any time point

This is the point of the method, so there are two ways to do it.

**A regular grid** — set the spacing you want:

```python
OUTPUT_STEP = 0.1     # 1980.0, 1980.1, 1980.2, ...
OUTPUT_STEP = 0.01    # 1980.00, 1980.01, 1980.02, ...
OUTPUT_STEP = 1       # 1980, 1981, 1982, ... only
```

**Specific points** — list exactly the ones you want:

```python
CUSTOM_TIMES = [1990.5, 2000.25, 2010.75]
```

A finer grid costs nothing. The curve is already fitted; this only decides how
densely it is written out.

> Points outside your observed range are extrapolation. The restricted cubic
> spline continues linearly beyond the outer knots, so a value is returned, but
> the program warns you that it is not supported by data.

---

## What you get

Everything is written to `OUTPUT_DIR` (default `nnrcs_output/`).

| File | Contents |
|---|---|
| `fitted_curve.csv` | The fitted curve on the grid: `time`, `rate_fitted`, `log_rate_fitted`, `slope_log_rate`, `percent_change_per_unit` |
| `fitted_at_observations.csv` | `time`, `rate_observed`, `rate_fitted`, `log_rate_observed`, `log_rate_fitted`, `residual_log_rate`, `var_log_rate`, `sd_log_rate` |
| `fitted_at_custom_times.csv` | Only if `CUSTOM_TIMES` is non-empty: `time`, `rate_fitted`, `log_rate_fitted`, `slope_log_rate`, `within_observed_range` |
| `model_parameters.csv` | Every estimated parameter — 62 with the default 20 nodes — plus the two time-scaling constants |
| `nn_rcs_fit.png` | Observed points and the fitted curve |

Column names use no statistical shorthand: there is no `y` and no `hat`.
Everything is either **observed** — exactly what you put in the file — or
**fitted** — what the model produced. `log_` prefixes the natural log of the
same quantity.

`slope_log_rate` is d(log rate)/d(time). Because the quantity is a log rate,
this is the relative change of the rate per time unit; ×100 it is
`percent_change_per_unit`, usually reported as the annual percentage change.

`within_observed_range` is `True` when the time point lies inside the period
you supplied and `False` when it does not. A `False` value is an
extrapolation: a number is returned, because the restricted cubic spline
continues linearly beyond the outer knots, but no data support it.

**No goodness-of-fit or model-comparison statistic is computed.** This program
fits one curve and hands you the curve.

---

## Settings

All of them are at the top of `nn_rcs.py`, above the line that says nothing
below it needs to be edited.

### Input

| Setting | Default | Meaning |
|---|---|---|
| `DATA_SOURCE` | `'demo_male'` | `'demo_male'`, `'demo_female'`, `'upload'`, `'file'` |
| `INPUT_CSV` | `'my_data.csv'` | Path to your file, when `DATA_SOURCE = 'file'` |
| `WRITE_TEMPLATE` | `False` | Write a blank input file and stop |
| `COL_RATE` | `'asir'` | Name of the rate column in your file |
| `COL_VAR` | `'var_log_asir'` | Name of the log-variance column |
| `COL_TIME` | `'time'` | Name of the time column, if you have one |
| `TIME_START` | `1.0` | First time point, when the file has no time column |
| `TIME_STEP` | `1.0` | Spacing between time points, same situation |

### Output

| Setting | Default | Meaning |
|---|---|---|
| `OUTPUT_DIR` | `'nnrcs_output'` | Where the files go |
| `OUTPUT_STEP` | `0.1` | Grid spacing for `fitted_curve.csv` |
| `CUSTOM_TIMES` | `[]` | Extra individual time points |
| `SAVE_PARAMETERS` | `True` | Write the estimated parameters |

### Figure

The defaults are tuned for **one large panel**, so you should get a clear,
publication-ready figure without changing anything. They differ from Figure 5 of
the paper, where six panels shared the width of a page.

| Setting | Default | Meaning |
|---|---|---|
| `MAKE_FIGURE` | `True` | Draw the figure at all |
| `FIG_WIDTH` | `8.0` | Width in inches |
| `FIG_HEIGHT` | `5.0` | Height in inches; wider than tall suits a time series |
| `FIG_DPI` | `300` | Resolution; 300 is print quality |
| `FIG_XLABEL` | `'Year'` | Horizontal axis label |
| `FIG_YLABEL` | `'Age-Standardized Incidence Rate\nper 100,000'` | Vertical axis label. **Change this if your quantity is not an incidence rate.** `\n` starts a second line |
| `FIG_TITLE` | `''` | Title; empty for none |
| `FIG_LOG_Y` | `True` | Logarithmic vertical axis, labelled with ordinary numbers. On a log axis a given vertical distance always means the same percentage change |
| `FIG_Y_LIMITS` | `None` | Fixed limits on the rate scale, e.g. `(4, 20)`; `None` to choose from the data |
| `FIG_END_XTICKS` | `False` | `False` chooses tick positions automatically; `True` shows only the first, middle and last time point, as in the paper |
| `FIG_LEGEND` | `True` | Legend naming the observed series and the fitted curve. On because this figure is usually looked at without a caption |
| `FIG_GRID` | `True` | Faint horizontal lines at the vertical-axis ticks, to help read values off |
| `FIG_POINT_SIZE` | `24` | Size of the observed points |
| `FIG_LINE_WIDTH` | `2.0` | Thickness of the fitted curve |
| `FIG_POINT_COLOR` | `'#2C5FD6'` | Blue, as in the paper |
| `FIG_LINE_COLOR` | `'#B0281C'` | Red, as in the paper |

To reproduce the exact look of Figure 5 of the paper:

```python
FIG_END_XTICKS = True
FIG_LEGEND     = False
FIG_GRID       = False
FIG_POINT_SIZE = 7
FIG_LINE_WIDTH = 1.1
```

### Model

The defaults are the ones used in the paper. They worked across every simulated
scenario studied and both real series, so you should not normally need to change
them.

| Setting | Default | Meaning |
|---|---|---|
| `N_HIDDEN_NODES` | `20` | Hidden nodes. Total parameters = 3 × nodes + 2 |
| `EPOCHS` | `2000` | Optimiser passes. Increase if the final loss is still falling |
| `LEARNING_RATE` | `0.001` | Adam step size |
| `GAMMA_INIT_SD` | `0.01` | SD of the random starting output weights |
| `NODE_SCALE_INIT` | `2.0` | Starting node scale, and initial spread of node positions |
| `W_MIN` | `0.01` | Lower bound on the node scales |
| `W_MAX` | `15.0` | Upper bound on the node scales |
| `B_CLIP_FRACTION` | `0.95` | Node positions kept within ±0.95·mean\|W\| |
| `RANDOM_SEED` | `42` | Any integer. Same seed, same result |
| `PRINT_EVERY` | `500` | Report the loss every N epochs; `0` for silence |

---

## The model

With time rescaled to *x* ∈ [−1, +1]:

```
ŷ(x) = α + β·x + Σ_j  γ_j · φ( W_j · x + B_j )
```

where φ is a three-knot restricted cubic spline with knots at *z* = −1, 0, +1.
Parameters: α, β, and (W_j, B_j, γ_j) for each of the *J* hidden nodes —
3*J* + 2 = 62 by default.

Fitted by minimising the inverse-variance weighted squared error on the log
scale,

```
Σ_i  (1 / var_i) · ( log(rate_i) − ŷ(x_i) )²
```

which is the maximum likelihood criterion when each log rate is normal with
variance `var_i`. Three things keep the curve well behaved:

1. **Structured initialisation.** Training starts from the inverse-variance
   weighted straight line, with small random output weights. The curve departs
   from that line only as far as the data require.
2. **Hard bounds instead of a penalty.** After every optimiser step the node
   scales and positions are clipped back into range, which keeps the hidden
   nodes inside the observed period.
3. **The RCS activation itself**, which is linear beyond its outer knots.

---

## Requirements

```
numpy
pandas
matplotlib
tensorflow
openpyxl        # only if you read .xlsx files
```

```bash
pip install numpy pandas matplotlib tensorflow
```

A GPU is used automatically if one is available, but is not needed. A single
model fit takes under ten seconds on a standard personal computer.

Results are exactly reproducible: same input and same `RANDOM_SEED` gives the
same numbers every time.

---

## Questions you may have

**Should I give you the rate or its logarithm?**
The rate. The column named by `COL_RATE` must hold the rate as you would report
it. The program logs it for you. Only the *variance* is on the log scale.

**How many time points do I need?**
The program stops below 6 and warns below 15. A network with 62 parameters is
very flexible for a short series. The method is intended for series of a few
decades.

**Must the time points be equally spaced?**
The curve is fitted correctly either way, but the program warns you if they are
not, because a fine output grid may then fall in places where you have no data.

**My data are not disease rates.**
That is fine. Anything positive with a variance at each time point works. Change
`FIG_YLABEL` so the axis does not say "incidence rate".

**What if a rate or a variance is zero?**
The program stops. A rate of zero has no logarithm, and a variance of zero would
give that point infinite weight. If your series has a genuinely error-free
point, give it a very small positive variance instead.

**Can I get a confidence interval for the fitted curve?**
Not from this program. That requires resampling, which is outside its scope.

**Why is there no measure of fit?**
Because deciding whether a fit is good is a separate question from producing
one, and it needs a comparison against something — held-out observations, a
known truth, or a competing model. Those comparisons are reported in the paper.
This program does the fitting.

---

## Citation

If you use this code, please cite:

> [Author]. *Flexible modeling of secular trends in disease rates: neural
> networks with restricted cubic spline.* [Journal / thesis], [year].

---

## Demonstration data

Oesophageal cancer incidence in Taiwan, 1980–2023, men and women, from the
Taiwan Cancer Registry. Age-standardised to the WHO 2000 World Standard
Population over 18 five-year age groups and expressed per 100,000; `asir` is the
rate and `var_log_asir` is the variance of its logarithm by the delta method.
Mean rates over the period are 10.10 per 100,000 in men and 0.89 in women. These
are aggregate published summary figures; no individual-level information is
involved.

---

## Licence

[Add your licence here — MIT is a common choice for research code.]
