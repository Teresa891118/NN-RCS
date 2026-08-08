"""
================================================================================
 NN-RCS: Flexible modelling of secular trends in disease rates
 A single-hidden-layer neural network with restricted-cubic-spline activation
================================================================================

WHAT THIS PROGRAM DOES
--------------------------------------------------------------------------------
You give it a time series of rates together with the variance of each log
rate. It fits one smooth curve through that series and lets you read the
fitted curve off at ANY time point you like -- not only at the observed
time points.

Everything is estimated with a single-hidden-layer neural network whose
activation function is a three-knot restricted cubic spline (RCS). The RCS
constraint forces the fitted curve to be linear beyond the outer knots, which
keeps the curve well behaved at the two ends of the series.


================================================================================
 1. WHAT YOU HAVE TO SUPPLY
================================================================================
Two numbers for every time point:

    rate_i     the age-standardised incidence rate itself, NOT logged
    var_i      the variance of its LOG-transformed value, i.e. Var( log rate )

That is all. The program takes the logarithm for you. The number of time points
n is read from your file -- you never have to tell the program how long your
series is.

WHY THE RATE IS NOT LOGGED BUT THE VARIANCE IS
    You already have the rate: it is what you report. What the model needs on
    the log scale is the weighting, and the variance of a log rate is not
    obtained by taking the log of a variance -- it comes from the delta method
    (see below). Asking for the rate as you have it, and for the one derived
    quantity that genuinely has to be computed, keeps the work on your side to
    a minimum.

"log" ALWAYS MEANS THE NATURAL LOGARITHM (base e, "ln"), never log base 10.

WHY THE PROGRAM DOES NOT AGE-STANDARDISE FOR YOU
    Age standardisation depends on choices that are yours, not ours: which
    standard population (WHO 2000, Segi, US 2000, ...), whether the age range is
    truncated (all ages, 20+, 40+, ...), and how many age groups you use. Doing
    it inside the program would force you into our choices. Computing an
    age-standardised rate and its variance is a standard, well-documented
    calculation, so we ask you to do that step yourself and hand us the result.

    For an age-standardised incidence rate (ASIR) from Poisson counts:
        ASIR_i      = sum_a  w_a * (d_ai / n_ai)          (w_a = standard weights)
        Var(ASIR_i) = sum_a  w_a^2 * d_ai / n_ai^2
        var_i       = Var(ASIR_i) / ASIR_i^2              (delta method)

    where d_ai and n_ai are the case count and the person-time in age group a
    at time point i. The last line is the delta-method approximation for the
    variance of a log-transformed quantity.

    Nothing in this program is specific to incidence rates. Any quantity that is
    positive and has a variance at each time point will work.

INPUT FILE FORMAT
    A file with a header row (.csv, .txt, .xlsx or .xls). Required columns:

        asir           the rate itself, not logged  (must be > 0)
        var_log_asir   variance of the log of that rate  (must be > 0)

    Optional column:

        time     the time point (calendar year, week, day, ...).
                 If this column is absent, the program creates the time axis
                 itself from TIME_START and TIME_STEP below.

    Example (my_data.csv):

        time,asir,var_log_asir
        1980,6.503659,0.00254975
        1981,6.739920,0.00275820
        1982,5.821879,0.00294383
        ...

    Rows must be in time order, and the time points must be equally spaced.
    Rows with a missing value in either required column are dropped, with a
    message.


================================================================================
 2. WHAT YOU GET BACK
================================================================================
All files are written to the folder named in OUTPUT_DIR.

    fitted_curve.csv          The fitted curve on a fine grid. This is the
                              red curve in the figure, as numbers. Columns:
                                  time                     time point
                                  rate_fitted              the fitted rate
                                  log_rate_fitted          its natural log
                                  slope_log_rate           d(log rate)/d(time),
                                                           i.e. the relative
                                                           change per time unit
                                  percent_change_per_unit  100 x slope_log_rate
                              The spacing of this grid is OUTPUT_STEP. Set
                              OUTPUT_STEP = 0.01 and you get 1.00, 1.01, 1.02,
                              ...; set it to 1 and you get 1, 2, 3, ...

    fitted_at_observations.csv    The fitted curve evaluated exactly at your
                              own time points, next to the two values you
                              supplied, so you can compare them directly.
                              Columns: time, rate_observed, rate_fitted,
                              log_rate_observed, log_rate_fitted,
                              residual_log_rate, var_log_rate, sd_log_rate.
                              "observed" always means what you gave the
                              program; "fitted" always means what the model
                              produced.

    fitted_at_custom_times.csv    Only written if you put time points in
                              CUSTOM_TIMES below. Use this when you want a few
                              specific times rather than a whole grid.

    model_parameters.csv      Every estimated parameter of the fitted network
                              (62 of them with the default 20 hidden nodes):
                              the intercept, the linear coefficient, and the
                              scale, position and output weight of each node.
                              See section 5 for how to reconstruct the curve
                              from these numbers.

    nn_rcs_fit.png            Observed points and the fitted curve.

The program does NOT compute any goodness-of-fit or model-comparison statistic.
It fits one curve and hands you the curve.


================================================================================
 3. HOW TO RUN IT
================================================================================
    In Google Colab
        Set DATA_SOURCE = 'upload', run the whole file, and a file-picker will
        appear. Or set DATA_SOURCE = 'demo_male' to try it with no file at all.

    On your own machine
        pip install numpy pandas matplotlib tensorflow
        Set DATA_SOURCE = 'file' and INPUT_CSV = 'path/to/my_data.csv'
        python nn_rcs.py

    Two demonstration series are built into this file, so it runs out of the
    box with no input at all: oesophageal cancer incidence in Taiwan, men and
    women, 1980-2023 (44 years). See section 6 for the source.

    Runtime is a few seconds to about a minute on a normal laptop, depending on
    EPOCHS. A GPU is used automatically if one is available, but is not needed.

    Results are exactly reproducible: with the same input and the same
    RANDOM_SEED you get the same numbers every time.
================================================================================
"""

# ##############################################################################
#
#                          U S E R   S E T T I N G S
#
#   Everything you may want to change is in this block. Nothing below it needs
#   to be edited to use the method on your own data.
#
# ##############################################################################

# ------------------------------------------------------------------------------
# 4.1  INPUT
# ------------------------------------------------------------------------------

# Where the data come from. One of:
#   'demo_male'    built-in example: Taiwanese men,   oesophageal cancer
#   'demo_female'  built-in example: Taiwanese women, oesophageal cancer
#   'upload'       open a file-picker (Google Colab only)
#   'file'         read the file named in INPUT_CSV
DATA_SOURCE = 'demo_male'

# Path to your data file. Only used when DATA_SOURCE = 'file'.
# Accepted formats: .csv, .txt, .xlsx, .xls
INPUT_CSV = 'my_data.csv'

# Set to True to write a blank input file (input_template.csv) showing the
# exact format expected, then stop. Fill it in with your own numbers, set this
# back to False, set DATA_SOURCE = 'file' and INPUT_CSV to that file, and run
# again. Useful the first time you prepare your own data.
WRITE_TEMPLATE = False

# Names of the columns in your file. Change these if your headers differ.
COL_RATE = 'asir'          # the rate itself, NOT logged, must be > 0  (required)
COL_VAR  = 'var_log_asir'  # variance of the LOG of that rate, > 0     (required)
COL_TIME = 'time'          # time point                                (optional)

# Used only when your file has no time column. The program then builds the time
# axis as TIME_START, TIME_START + TIME_STEP, TIME_START + 2*TIME_STEP, ...
# Set TIME_START = 1 if you want your first observation to be called "year 1",
# or 0 if you prefer to start at zero, or 1980 for a calendar year.
TIME_START = 1.0
TIME_STEP  = 1.0

# ------------------------------------------------------------------------------
# 4.2  OUTPUT
# ------------------------------------------------------------------------------

# Folder for all output files. Created automatically if it does not exist.
OUTPUT_DIR = 'nnrcs_output'

# Spacing of the fine grid written to fitted_curve.csv.
#   0.1  -> ..., 1.0, 1.1, 1.2, ...      (10 points per time unit)
#   0.01 -> ..., 1.00, 1.01, 1.02, ...   (100 points per time unit)
#   1    -> ..., 1, 2, 3, ...            (only the whole time points)
# A finer grid costs nothing: the curve is already fitted, this only decides
# how densely it is written out.
OUTPUT_STEP = 0.1

# Extra individual time points you want the fitted value for. Leave the list
# empty to skip. These may be anywhere, and need not lie on the grid above.
#   e.g. CUSTOM_TIMES = [1990.5, 2000.25, 2010.75]
CUSTOM_TIMES = []

# Write the estimated network parameters to model_parameters.csv?
SAVE_PARAMETERS = True

# ------------------------------------------------------------------------------
# 4.3  FIGURE
# ------------------------------------------------------------------------------

# The figure keeps the visual identity of the published paper -- serif type, no
# top or right frame, blue observed points, a red fitted curve, and a
# logarithmic vertical axis labelled with ordinary numbers -- but the defaults
# below are tuned for ONE large panel rather than for the six small side-by-side
# panels of the paper. You should get a clear, publication-ready figure without
# changing anything here.
#
# Every setting is explained where it appears. In brief:
#
#   size and resolution     FIG_WIDTH, FIG_HEIGHT, FIG_DPI
#   what the axes say       FIG_XLABEL, FIG_YLABEL, FIG_TITLE
#   vertical axis           FIG_LOG_Y, FIG_Y_LIMITS
#   horizontal axis         FIG_END_XTICKS
#   reading aids            FIG_LEGEND, FIG_GRID
#   ink                     FIG_POINT_SIZE, FIG_LINE_WIDTH,
#                           FIG_POINT_COLOR, FIG_LINE_COLOR
#
# To reproduce the exact look of Figure 5 of the paper instead, set
#   FIG_END_XTICKS = True, FIG_LEGEND = False, FIG_GRID = False,
#   FIG_POINT_SIZE = 7, FIG_LINE_WIDTH = 1.1

MAKE_FIGURE = True

# Size in inches and resolution in dots per inch. 8 x 5 is a little wider than
# it is tall, which suits a time series: it gives the horizontal axis room
# without flattening the vertical differences. 300 dpi is print quality.
FIG_WIDTH   = 8.0
FIG_HEIGHT  = 5.0
FIG_DPI     = 300

# Text on the axes. Change FIG_YLABEL if your quantity is not an
# age-standardised incidence rate. "\n" starts a second line.
FIG_XLABEL = 'Year'
FIG_YLABEL = 'Age-Standardized Incidence Rate\nper 100,000'
FIG_TITLE  = ''       # '' for no title

# Draw the vertical axis on a logarithmic scale, labelled with ordinary numbers
# rather than powers of ten. Recommended for rates: on a log axis a given
# vertical distance always means the same percentage change, so the steepness
# of the curve can be compared between the low and the high parts of the series.
FIG_LOG_Y = True

# Fixed limits for the vertical axis, on the RATE scale (not the log scale).
# Leave as None to let the program choose from the data.
#   e.g. FIG_Y_LIMITS = (4, 20)
FIG_Y_LIMITS = None

# Horizontal axis tick positions.
#   False  chosen automatically -- usually about five evenly spaced round
#          numbers, which is the easier read in a single large panel
#   True   only the first, middle and last time point, as in the paper, where
#          six panels shared the width of the page and labels would collide
FIG_END_XTICKS = False

# Draw a legend saying which is the observed series and which is the fitted
# curve. On by default because a figure produced by this program is usually
# looked at on its own, without the caption that carries that information in
# the paper.
FIG_LEGEND = True

# Draw faint horizontal lines at the vertical-axis ticks. These make it much
# easier to read a value off a single large panel, and are light enough not to
# compete with the data. Set to False for the unadorned look of the paper.
FIG_GRID = False

# Size of the observed points and thickness of the fitted curve. Both are
# larger than in the paper because here they occupy one full-size panel instead
# of one sixth of a page width.
FIG_POINT_SIZE = 24
FIG_LINE_WIDTH = 2.0

# Colours, as in the paper.
FIG_POINT_COLOR = '#2C5FD6'   # blue observed points
FIG_LINE_COLOR  = '#B0281C'   # red fitted curve

# ------------------------------------------------------------------------------
# 4.4  MODEL AND FITTING
# ------------------------------------------------------------------------------
# The defaults below are the ones used in the paper. They worked well across
# every simulated scenario we studied and both real series, so you should not
# normally need to change them. They are exposed here so that you can.

# Number of hidden nodes. More nodes give a more flexible curve. The total
# number of estimated parameters is 3 * N_HIDDEN_NODES + 2 (62 by default).
N_HIDDEN_NODES = 20

# Number of passes of the optimiser over the data. Increase if the reported
# final loss is still falling noticeably at the end of training.
EPOCHS = 2000

# Adam optimiser step size. Smaller is slower but steadier.
LEARNING_RATE = 0.001

# Standard deviation of the random starting values of the output weights.
# Training starts from an almost straight line and departs from it only as far
# as the data require; a small value here is what makes that happen.
GAMMA_INIT_SD = 0.01

# Starting value for every node scale W_j. Also sets the initial spread of the
# node positions B_j, which are laid out evenly over
# [-0.95 * NODE_SCALE_INIT, +0.95 * NODE_SCALE_INIT].
NODE_SCALE_INIT = 2.0

# Hard bounds applied after every optimiser step.
#   W stays inside [W_MIN, W_MAX]
#   B stays inside +/- B_CLIP_FRACTION * mean(|W|)
# These keep the hidden nodes inside the observed range of the data, which is
# what stops the curve from developing features where there are no data.
W_MIN            = 0.01
W_MAX            = 15.0
B_CLIP_FRACTION  = 0.95

# Random seed. Any integer. The same seed always gives the same result.
RANDOM_SEED = 42

# Print the training loss every PRINT_EVERY epochs (0 = silent).
PRINT_EVERY = 500


# ##############################################################################
#
#   Nothing below this line needs to be edited to use the method on your data.
#
# ##############################################################################

import os
import sys
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
from matplotlib import ticker as mticker
import tensorflow as tf
tf.get_logger().setLevel('ERROR')


# ==============================================================================
# 5. BUILT-IN DEMONSTRATION DATA
# ==============================================================================
# Oesophageal cancer in Taiwan, 1980-2023 (44 calendar years), from the Taiwan
# Cancer Registry. Age-standardised to the WHO 2000 World Standard Population
# over 18 five-year age groups and expressed per 100,000. The two arrays below
# are exactly what this program asks a user to supply:
#     _RATE   is the age-standardised incidence rate itself, NOT logged
#     _VARLOG is the variance of the LOG of that rate, by the delta method
# Mean rates over the period are 10.10 per 100,000 in men and 0.89 in women.
# These are aggregate, already-published summary figures; no individual-level
# information is involved.
#
# Note that the age standardisation has already been done -- this program never
# sees age groups, standard-population weights or case counts. That is the whole
# point of the interface: you hand it a rate and a variance, nothing else.
# ==============================================================================
_DEMO_YEARS = list(range(1980, 2024))

_DEMO_MALE_RATE = [
    6.503659, 6.739920, 5.821879, 6.607650, 6.092290, 5.312125, 6.024190,
    5.829126, 5.644655, 6.254405, 5.327020, 6.303588, 6.438013, 6.388688,
    6.555871, 6.670481, 7.917369, 7.819605, 8.077553, 8.251698, 8.855033,
    10.017444, 10.060029, 10.056981, 11.399768, 11.002681, 12.368835,
    12.514313, 13.401840, 13.327759, 14.295903, 13.601349, 14.064853,
    14.795070, 14.985889, 14.432231, 14.425089, 14.873201, 14.694305,
    14.521249, 14.492206, 13.938350, 13.585445, 13.971185]
_DEMO_MALE_VARLOG = [
    0.00254975, 0.00275820, 0.00294383, 0.00259365, 0.00272347, 0.00266244,
    0.00246489, 0.00236665, 0.00239620, 0.00207182, 0.00232107, 0.00191120,
    0.00175193, 0.00169684, 0.00164361, 0.00157807, 0.00131027, 0.00126883,
    0.00118842, 0.00112725, 0.00103143, 0.00088421, 0.00086519, 0.00082759,
    0.00071661, 0.00071547, 0.00061975, 0.00059421, 0.00054021, 0.00052921,
    0.00047602, 0.00048874, 0.00046211, 0.00042972, 0.00041367, 0.00042023,
    0.00041260, 0.00039204, 0.00038881, 0.00038323, 0.00038110, 0.00039022,
    0.00039466, 0.00037839]

_DEMO_FEMALE_RATE = [
    0.811691, 1.217764, 0.699128, 0.974103, 0.839691, 0.703473, 0.635161,
    0.773057, 0.599145, 0.790350, 0.750652, 0.793120, 1.032546, 0.755011,
    0.764776, 0.718599, 0.979858, 0.799688, 0.918157, 0.892843, 0.782204,
    0.811293, 1.008217, 0.911551, 0.823734, 0.744454, 0.832891, 0.994232,
    0.955750, 0.978082, 0.935894, 0.839879, 1.052231, 0.833661, 0.935686,
    0.927440, 0.946157, 1.041888, 1.000487, 0.919435, 0.989523, 1.207208,
    1.092676, 1.096946]
_DEMO_FEMALE_VARLOG = [
    0.02373404, 0.01624158, 0.02410558, 0.01782045, 0.01889619, 0.02237344,
    0.02230440, 0.01912357, 0.02207375, 0.01747287, 0.01701280, 0.01576777,
    0.01181576, 0.01562881, 0.01464490, 0.01492246, 0.01077973, 0.01275991,
    0.01063446, 0.01068031, 0.01159533, 0.01068604, 0.00840444, 0.00892480,
    0.00944232, 0.01008143, 0.00870676, 0.00703698, 0.00706458, 0.00668549,
    0.00683513, 0.00732060, 0.00569929, 0.00694989, 0.00635563, 0.00612083,
    0.00584308, 0.00516767, 0.00532230, 0.00567499, 0.00532050, 0.00416363,
    0.00490091, 0.00455325]


def _demo_frame(which):
    r = _DEMO_MALE_RATE if which == 'male' else _DEMO_FEMALE_RATE
    v = _DEMO_MALE_VARLOG if which == 'male' else _DEMO_FEMALE_VARLOG
    return pd.DataFrame({COL_TIME: _DEMO_YEARS, COL_RATE: r, COL_VAR: v})


def _read_table(path):
    """Read a data file. Accepts .csv, .txt, .xlsx and .xls.

    Spreadsheets are accepted because that is what most people actually have.
    For text files the encoding is tried in turn, so a file exported from
    software using a regional code page still opens.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.xlsx', '.xls', '.xlsm'):
        try:
            return pd.read_excel(path)
        except ImportError:
            sys.exit('Reading Excel files needs the "openpyxl" package.\n'
                     'Install it with:  pip install openpyxl\n'
                     'Or save your data as .csv and try again.')
    last = None
    for enc in ('utf-8-sig', 'utf-8', 'big5', 'cp1252', 'latin-1'):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError as e:
            last = e
        except Exception as e:
            sys.exit(f'Could not read {path}: {e}')
    sys.exit(f'Could not decode {path}. Last error: {last}\n'
             'Try re-saving the file as CSV UTF-8.')


def write_template(path):
    """Write an empty input file showing exactly the format expected."""
    tpl = pd.DataFrame({
        COL_TIME: [1980, 1981, 1982],
        COL_RATE: [6.503659, 6.739920, 5.821879],
        COL_VAR:  [0.00254975, 0.00275820, 0.00294383],
    })
    tpl.to_csv(path, index=False)
    print(f'  input template written to {path}\n'
          f'    {COL_TIME:<14} time point (optional column)\n'
          f'    {COL_RATE:<14} the rate itself, NOT logged (> 0)\n'
          f'    {COL_VAR:<14} variance of the LOG of that rate (> 0)\n'
          f'  Replace the three example rows with your own, as many as you have.')


# ==============================================================================
# 6. READING AND CHECKING THE DATA
# ==============================================================================
def load_data():
    """Read and check the data.

    Returns four 1-D arrays: (time, rate, log rate, variance of the log rate).
    The rate is what the user supplied; the log is taken here, once.
    """
    src = DATA_SOURCE.lower()

    if src == 'demo_male':
        df = _demo_frame('male')
        origin = 'built-in demonstration series: Taiwan, men, oesophageal cancer'
    elif src == 'demo_female':
        df = _demo_frame('female')
        origin = 'built-in demonstration series: Taiwan, women, oesophageal cancer'
    elif src == 'upload':
        try:
            from google.colab import files
        except ImportError:
            sys.exit(
                "DATA_SOURCE = 'upload' needs Google Colab.\n"
                "  In Colab : paste this file into a notebook cell and run the\n"
                "             cell. A 'Choose Files' button will appear. Note\n"
                "             that it will NOT appear if you launch the file\n"
                "             with !python, because the button needs the\n"
                "             notebook itself.\n"
                "  Elsewhere: set DATA_SOURCE = 'file' and put the path to your\n"
                "             data in INPUT_CSV.")
        print('A "Choose Files" button will appear below. Select your data file '
              '(.csv or .xlsx).')
        uploaded = files.upload()
        if not uploaded:
            sys.exit('No file was chosen.')
        fname = list(uploaded.keys())[0]
        if len(uploaded) > 1:
            print(f'  note: {len(uploaded)} files were chosen; using the first '
                  f'one, {fname}.')
        df = _read_table(fname)
        origin = f'uploaded file: {fname}'
    elif src == 'file':
        if not os.path.exists(INPUT_CSV):
            here = [f for f in sorted(os.listdir('.'))
                    if f.lower().endswith(('.csv', '.xlsx', '.xls', '.txt'))][:10]
            sys.exit(f'File not found: {INPUT_CSV}\n'
                     f'  Working directory: {os.path.abspath(".")}\n'
                     f'  Data files found here: {here if here else "none"}\n'
                     f'  Fix INPUT_CSV, or set DATA_SOURCE to a demo series, or\n'
                     f'  set WRITE_TEMPLATE = True to get a blank input file.')
        df = _read_table(INPUT_CSV)
        origin = f'file: {INPUT_CSV}'
    else:
        sys.exit(f"DATA_SOURCE must be 'demo_male', 'demo_female', 'upload' or "
                 f"'file'; got {DATA_SOURCE!r}.")

    # --- required columns -----------------------------------------------------
    missing = [c for c in (COL_RATE, COL_VAR) if c not in df.columns]
    if missing:
        sys.exit(f'Required column(s) not found in the data: {missing}\n'
                 f'Columns present: {list(df.columns)}\n'
                 f'Either rename your columns, or change COL_RATE / COL_VAR at the '
                 f'top of this file.')

    # --- time axis ------------------------------------------------------------
    if COL_TIME in df.columns:
        t = df[COL_TIME].to_numpy(float)
        time_from_file = True
    else:
        t = TIME_START + TIME_STEP * np.arange(len(df), dtype=float)
        time_from_file = False

    rate = df[COL_RATE].to_numpy(float)
    v = df[COL_VAR].to_numpy(float)

    # --- drop unusable rows ---------------------------------------------------
    ok = np.isfinite(t) & np.isfinite(rate) & np.isfinite(v)
    if (~ok).any():
        print(f'  note: {int((~ok).sum())} row(s) dropped because time, '
              f'{COL_RATE} or {COL_VAR} was missing or not a number.')
    t, rate, v = t[ok], rate[ok], v[ok]

    if (rate <= 0).any():
        sys.exit(f'{COL_RATE} contains values that are zero or negative. The '
                 f'model works on the log scale, so every rate must be strictly '
                 f'positive. Offending time points: {t[rate <= 0][:10]}\n'
                 f'Note that this column must hold the rate itself, NOT its '
                 f'logarithm -- the program takes the logarithm for you.')

    if (v <= 0).any():
        sys.exit(f'{COL_VAR} contains values that are zero or negative. A '
                 f'variance must be strictly positive. Offending time points: '
                 f'{t[v <= 0][:10]}')

    order = np.argsort(t)
    if not np.array_equal(order, np.arange(len(t))):
        print('  note: rows were not in time order; they have been sorted.')
        t, rate, v = t[order], rate[order], v[order]

    n = len(t)
    if n < 6:
        sys.exit(f'Only {n} usable time points. At least 6 are needed, and the '
                 f'method is intended for appreciably longer series.')
    if n < 15:
        print(f'  warning: only {n} time points. A network with '
              f'{3 * N_HIDDEN_NODES + 2} parameters is very flexible for a '
              f'series this short; treat the fitted curve with caution.')

    # --- equal spacing --------------------------------------------------------
    gaps = np.diff(t)
    if gaps.std() > 1e-8 * max(1.0, abs(gaps.mean())):
        print(f'  warning: the time points are not equally spaced '
              f'(gaps range from {gaps.min():g} to {gaps.max():g}). The curve '
              f'is still fitted correctly, but a fine output grid may fall in '
              f'places where you have no data.')

    # The model works on the log scale throughout; this is the only place the
    # logarithm is taken, and it is taken for you.
    y = np.log(rate)

    print(f'  source        : {origin}')
    print(f'  time column   : {"from file" if time_from_file else "generated from TIME_START / TIME_STEP"}')
    print(f'  n             : {n} time points, from {t[0]:g} to {t[-1]:g}')
    # Echo back what was read, on both scales. The commonest way to get this
    # wrong is to put log rates into the rate column; seeing the two ranges
    # side by side makes that obvious at a glance. (A negative or zero rate is
    # rejected outright above, which already catches the case of a series whose
    # rates are below 1.)
    print(f'  {COL_RATE:<14}: {rate.min():.4g} to {rate.max():.4g}   '
          f'(mean {rate.mean():.4g})  <- read as RATES, not logs')
    print(f'  log of it     : {y.min():.4f} to {y.max():.4f}   '
          f'(taken by this program)')
    print(f'  {COL_VAR:<14}: {v.min():.4g} to {v.max():.4g}')
    return t, rate, y, v


# ==============================================================================
# 7. THE MODEL
# ==============================================================================
def _rcs_activation(z):
    """Three-knot restricted cubic spline activation, knots at z = -1, 0, +1.

    Being a restricted cubic spline, it is linear outside [-1, +1] and has a
    continuous second derivative everywhere. Those two properties are what make
    the fitted curve smooth in the interior and well behaved at the ends.
    """
    return 0.5 * (tf.pow(tf.nn.relu(z + 1.), 3) / 3.
                  - 2 * tf.pow(tf.nn.relu(z), 3) / 3.
                  + tf.pow(tf.nn.relu(z - 1.), 3) / 3. - z)


class NNRCS(tf.keras.Model):
    """Single hidden layer, RCS activation, plus a direct linear term.

        y_hat(x) = alpha + beta * x + sum_j  gamma_j * phi( W_j * x + B_j )

    with x the time axis rescaled to [-1, +1]. The parameters are
    alpha, beta, and (W_j, B_j, gamma_j) for j = 1 .. N_HIDDEN_NODES.
    """

    def __init__(self, n_nodes, scale_init, seed, alpha0, beta0):
        super().__init__()
        b_init = np.linspace(-scale_init * B_CLIP_FRACTION,
                             scale_init * B_CLIP_FRACTION,
                             n_nodes).astype('float32')
        self.B = tf.Variable(b_init.reshape(1, n_nodes))
        self.W = tf.Variable(tf.ones((1, 1, n_nodes)) * float(scale_init))
        self.gamma = tf.Variable(
            tf.random.normal((1, n_nodes, 1), stddev=GAMMA_INIT_SD, seed=seed))
        self.beta = tf.Variable(np.asarray([beta0], 'f').reshape(1, 1, 1))
        self.alpha = tf.Variable(np.asarray([alpha0], 'f').reshape(1, 1, 1))
        self.trainable_vars_list = [self.B, self.gamma, self.beta,
                                    self.alpha, self.W]
        self.opt = tf.keras.optimizers.Adam(LEARNING_RATE)

    def forward(self, x):
        xt = tf.tile(tf.expand_dims(x, 0), [1, 1, 1])
        z = xt * self.W + tf.expand_dims(self.B, 1)
        a = _rcs_activation(z)
        return tf.matmul(a, self.gamma) + tf.matmul(xt, self.beta) + self.alpha

    @tf.function
    def train_step(self, x, y, w):
        # Inverse-variance weighted squared error on the log scale: the maximum
        # likelihood criterion when each y_i is normal with variance var_i.
        with tf.GradientTape() as tape:
            pred = self.forward(x)
            loss = tf.reduce_mean(tf.square(pred - y) * w)
        self.opt.apply_gradients(
            zip(tape.gradient(loss, self.trainable_vars_list),
                self.trainable_vars_list))
        # Hard bounds, applied after the step rather than as a penalty term.
        lim = tf.reduce_mean(tf.abs(self.W)) * B_CLIP_FRACTION
        self.B.assign(tf.clip_by_value(self.B, -lim, lim))
        self.W.assign(tf.clip_by_value(self.W, W_MIN, W_MAX))
        return loss

    def predict_log(self, x_np):
        """Fitted value on the log scale at arbitrary rescaled inputs."""
        xt = tf.constant(np.asarray(x_np, float).reshape(-1, 1), tf.float32)
        return np.squeeze(self.forward(xt).numpy())


def fit_nn_rcs(t, y, var_log_rate):
    """Fit the network. Returns (model, rescale, unrescale)."""
    t_lo, t_hi = float(t[0]), float(t[-1])

    def rescale(tt):                       # time  ->  x in [-1, +1]
        return 2.0 * (np.asarray(tt, float) - t_lo) / (t_hi - t_lo) - 1.0

    def unrescale(xx):                     # x     ->  time
        return t_lo + (np.asarray(xx, float) + 1.0) * (t_hi - t_lo) / 2.0

    x = rescale(t)
    inv = 1.0 / var_log_rate                      # precision of each observation
    w = inv / inv.mean()                   # normalised so the mean weight is 1

    # Structured start: an inverse-variance weighted straight line. Training
    # then only has to explain the departure from that line.
    X = np.column_stack([np.ones(len(x)), x])
    XW = X.T * inv
    alpha0, beta0 = np.linalg.solve(XW @ X, XW @ y)

    device = '/gpu:0' if tf.config.list_physical_devices('GPU') else '/cpu:0'
    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    x_tf = tf.constant(x.reshape(-1, 1), tf.float32)
    y_tf = tf.constant(y.reshape(1, -1, 1), tf.float32)
    w_tf = tf.constant(w.reshape(1, -1, 1), tf.float32)

    print(f'\nFitting NN-RCS  ({N_HIDDEN_NODES} hidden nodes, '
          f'{3 * N_HIDDEN_NODES + 2} parameters, {EPOCHS} epochs, '
          f'device {device})')
    print(f'  starting straight line: alpha = {alpha0:.6f}, beta = {beta0:.6f}')

    with tf.device(device):
        model = NNRCS(N_HIDDEN_NODES, NODE_SCALE_INIT, RANDOM_SEED,
                      float(alpha0), float(beta0))
        loss = None
        for epoch in range(1, EPOCHS + 1):
            loss = model.train_step(x_tf, y_tf, w_tf)
            if PRINT_EVERY and (epoch % PRINT_EVERY == 0 or epoch == 1):
                print(f'  epoch {epoch:>6d} / {EPOCHS}   weighted loss = '
                      f'{float(loss):.8f}')
    print(f'  final weighted loss = {float(loss):.8f}')
    return model, rescale, unrescale


# ==============================================================================
# 8. READING THE FITTED CURVE OFF AT ANY TIME POINT
# ==============================================================================
def _decimals_of(v, cap=10):
    """How many decimal places are needed to write v exactly (capped)."""
    s = f'{float(v):.{cap}f}'.rstrip('0')
    frac = s.split('.')[1] if '.' in s else ''
    return len(frac)


def build_grid(t_lo, t_hi, step):
    """Regular grid from t_lo to t_hi, inclusive of both ends.

    The obvious construction t_lo + step * arange(n) accumulates binary
    floating-point error: 0.1 cannot be represented exactly, so 1 + 0.1*7 comes
    out as 1.7000000000000002. The error is ~1e-16 and has no effect on the
    fitted values, but it makes the time column of the output ugly and awkward
    to join on. Rounding to the number of decimals actually implied by the step
    and the starting time removes it.
    """
    span = t_hi - t_lo
    n_step = int(np.floor(span / step + 1e-9))
    g = t_lo + step * np.arange(n_step + 1)
    note = ''
    if g[-1] < t_hi - 1e-9:
        note = (f'OUTPUT_STEP = {step:g} does not divide the observed period '
                f'({span:g}) a whole number of times. The final time point '
                f'{t_hi:g} has been added, so the last interval is '
                f'{t_hi - g[-1]:g} instead of {step:g}. Choose a step that '
                f'divides {span:g} exactly to avoid this.')
        g = np.append(g, t_hi)
    dec = max(_decimals_of(step), _decimals_of(t_lo), _decimals_of(t_hi))
    return np.round(g, dec), note


def evaluate(model, rescale, times):
    """Fitted log value and rate at any set of time points."""
    times = np.asarray(times, float)
    y_hat = np.atleast_1d(model.predict_log(rescale(times)))
    return y_hat, np.exp(y_hat)


def slope_of(model, rescale, times, h=None):
    """d(y_hat)/d(time), by a centred difference of half-width h.

    Because y_hat is a log rate, this slope is the relative change of the rate
    per time unit; multiplied by 100 it is the percentage change per time unit.
    """
    if h is None:
        h = OUTPUT_STEP if OUTPUT_STEP > 0 else 0.1
    times = np.asarray(times, float)
    up = np.atleast_1d(model.predict_log(rescale(times + h)))
    dn = np.atleast_1d(model.predict_log(rescale(times - h)))
    return (up - dn) / (2.0 * h)


def extract_parameters(model, t_lo, t_hi):
    """All estimated parameters, as a tidy table."""
    B = np.asarray(model.B.numpy()).ravel()
    W = np.asarray(model.W.numpy()).ravel()
    g = np.asarray(model.gamma.numpy()).ravel()
    alpha = float(np.asarray(model.alpha.numpy()).ravel()[0])
    beta = float(np.asarray(model.beta.numpy()).ravel()[0])
    rows = [dict(parameter='alpha', index='', value=alpha,
                 meaning='intercept of the direct linear term'),
            dict(parameter='beta', index='', value=beta,
                 meaning='slope of the direct linear term, per unit of x')]
    for j in range(len(W)):
        rows.append(dict(parameter='W', index=j + 1, value=float(W[j]),
                         meaning=f'scale of hidden node {j + 1}'))
    for j in range(len(B)):
        rows.append(dict(parameter='B', index=j + 1, value=float(B[j]),
                         meaning=f'position of hidden node {j + 1}'))
    for j in range(len(g)):
        rows.append(dict(parameter='gamma', index=j + 1, value=float(g[j]),
                         meaning=f'output weight of hidden node {j + 1}'))
    rows.append(dict(parameter='time_min', index='', value=t_lo,
                     meaning='time mapped to x = -1'))
    rows.append(dict(parameter='time_max', index='', value=t_hi,
                     meaning='time mapped to x = +1'))
    return pd.DataFrame(rows)


# ==============================================================================
# 9. FIGURE
# ==============================================================================
def _plain_log_ticks(lo, hi):
    base = [1, 1.2, 1.4, 1.6, 1.8, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 7, 8, 9]
    cands = sorted(set(round(b * 10 ** e, 8) for e in range(-6, 8) for b in base))
    ticks = [v for v in cands if lo <= v <= hi]
    if len(ticks) > 9:
        idx = sorted(set(np.linspace(0, len(ticks) - 1, 8).round().astype(int)))
        ticks = [ticks[i] for i in idx]
    return ticks


def _fmt(v):
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    return f'{v:.4g}'


def make_figure(t, rate_obs, curve_t, curve_rate, path):
    """Observed points and the fitted curve, sized for a single large panel."""
    # Type is a little larger than in the paper: there the figure held six
    # panels across a page width and was reproduced small, whereas this one is
    # usually looked at on its own.
    FS_BASE, FS_LABEL, FS_TICK, FS_TITLE = 12, 12.5, 11, 13.5
    matplotlib.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': FS_BASE,
        'axes.grid': False,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    if FIG_GRID:
        # Behind everything, and faint enough to read past.
        ax.set_axisbelow(True)
        ax.grid(axis='y', which='major', color='0.90', lw=0.8, zorder=0)
    ax.scatter(t, rate_obs, s=FIG_POINT_SIZE, color=FIG_POINT_COLOR,
               linewidth=0, zorder=3, label='Observed')
    ax.plot(curve_t, curve_rate, color=FIG_LINE_COLOR, lw=FIG_LINE_WIDTH,
            solid_capstyle='round', zorder=4, label='NN-RCS fitted trend')

    if FIG_Y_LIMITS is not None:
        lo, hi = FIG_Y_LIMITS
    else:
        allv = np.concatenate([rate_obs, curve_rate])
        allv = allv[allv > 0]
        lo, hi = allv.min() * 0.85, allv.max() * 1.15
    if FIG_LOG_Y:
        ax.set_yscale('log')
        ax.set_ylim(lo, hi)
        tk = _plain_log_ticks(lo, hi)
        if len(tk) >= 2:
            ax.set_yticks(tk)
            ax.set_yticklabels([_fmt(v) for v in tk])
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    else:
        ax.set_ylim(lo, hi)

    # Horizontal axis: first, middle and last time point, as in the paper.
    t0, t1 = float(t[0]), float(t[-1])
    ax.set_xlim(t0 - (t1 - t0) * 0.02, t1 + (t1 - t0) * 0.02)
    if FIG_END_XTICKS:
        mid = (t0 + t1) / 2
        ticks = [t0, round(mid) if abs(mid - round(mid)) < 0.5 else mid, t1]
        ax.set_xticks(ticks)
        ax.set_xticklabels([_fmt(v) for v in ticks])

    ax.tick_params(labelsize=FS_TICK)
    ax.set_xlabel(FIG_XLABEL, fontsize=FS_LABEL)
    ax.set_ylabel(FIG_YLABEL, fontsize=FS_LABEL)
    if FIG_TITLE:
        ax.set_title(FIG_TITLE, fontsize=FS_TITLE, fontweight='bold')
    if FIG_LEGEND:
        ax.legend(frameon=False, fontsize=FS_TICK, loc='best')
    plt.tight_layout()
    fig.savefig(path, dpi=FIG_DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)


# ==============================================================================
# 10. MAIN
# ==============================================================================
def main():
    print('=' * 78)
    print(' NN-RCS: flexible modelling of secular trends in disease rates')
    print(' (log means the NATURAL logarithm throughout)')
    print('=' * 78)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if WRITE_TEMPLATE:
        write_template(os.path.join(OUTPUT_DIR, 'input_template.csv'))
        print('\nSet WRITE_TEMPLATE = False once your file is ready, then run '
              'again.')
        return None, None

    t, rate_obs, y, var_log_rate = load_data()

    model, rescale, unrescale = fit_nn_rcs(t, y, var_log_rate)
    t_lo, t_hi = float(t[0]), float(t[-1])

    written = []

    # ---- the fitted curve on a fine grid ------------------------------------
    if OUTPUT_STEP <= 0:
        sys.exit('OUTPUT_STEP must be greater than zero.')
    # The grid always spans exactly the observed period: it starts at your
    # first time point and ends at your last one, whatever they are. Only the
    # spacing is yours to choose (OUTPUT_STEP).
    grid_t, grid_note = build_grid(t_lo, t_hi, OUTPUT_STEP)
    if grid_note:
        print(f'\n  note: {grid_note}')
    g_y, g_rate = evaluate(model, rescale, grid_t)
    g_slope = slope_of(model, rescale, grid_t)
    # Column names avoid statistical shorthand: no "y", no "hat". Everything
    # is either "observed" (what you supplied) or "fitted" (what the model says).
    curve = pd.DataFrame({
        'time': grid_t,
        'rate_fitted': g_rate,
        'log_rate_fitted': g_y,
        'slope_log_rate': g_slope,
        'percent_change_per_unit': 100.0 * g_slope,
    })
    p = os.path.join(OUTPUT_DIR, 'fitted_curve.csv')
    curve.to_csv(p, index=False)
    written.append((p, f'fitted curve on a grid of step {OUTPUT_STEP:g} '
                       f'({len(curve)} points)'))

    # ---- the fitted curve at the observed time points ------------------------
    o_y, o_rate = evaluate(model, rescale, t)
    obs = pd.DataFrame({
        'time': t,
        'rate_observed': rate_obs,          # exactly what you supplied
        'rate_fitted': o_rate,
        'log_rate_observed': y,             # log of what you supplied
        'log_rate_fitted': o_y,
        'residual_log_rate': y - o_y,       # observed minus fitted
        'var_log_rate': var_log_rate,              # exactly what you supplied
        'sd_log_rate': np.sqrt(var_log_rate),
    })
    p = os.path.join(OUTPUT_DIR, 'fitted_at_observations.csv')
    obs.to_csv(p, index=False)
    written.append((p, 'fitted curve at your own time points'))

    # ---- user-requested individual time points -------------------------------
    if len(CUSTOM_TIMES) > 0:
        ct = np.asarray(CUSTOM_TIMES, float)
        outside = ct[(ct < t_lo) | (ct > t_hi)]
        if outside.size:
            print(f'\n  warning: {outside.size} of the CUSTOM_TIMES lie outside '
                  f'the observed range {t_lo:g} to {t_hi:g}. The restricted '
                  f'cubic spline continues linearly beyond the outer knots, so '
                  f'a value is returned, but it is an extrapolation and is not '
                  f'supported by data.')
        c_y, c_rate = evaluate(model, rescale, ct)
        cus = pd.DataFrame({
            'time': ct,
            'rate_fitted': c_rate,
            'log_rate_fitted': c_y,
            'slope_log_rate': slope_of(model, rescale, ct),
            'within_observed_range': (ct >= t_lo) & (ct <= t_hi),
        })
        p = os.path.join(OUTPUT_DIR, 'fitted_at_custom_times.csv')
        cus.to_csv(p, index=False)
        written.append((p, f'fitted curve at your {len(ct)} chosen time points'))

    # ---- estimated parameters ------------------------------------------------
    if SAVE_PARAMETERS:
        pars = extract_parameters(model, t_lo, t_hi)
        p = os.path.join(OUTPUT_DIR, 'model_parameters.csv')
        pars.to_csv(p, index=False)
        written.append((p, f'{len(pars) - 2} estimated parameters '
                           f'(plus the two time-scaling constants)'))

    # ---- figure --------------------------------------------------------------
    if MAKE_FIGURE:
        p = os.path.join(OUTPUT_DIR, 'nn_rcs_fit.png')
        make_figure(t, rate_obs, grid_t, g_rate, p)
        written.append((p, 'observed points and the fitted curve'))

    # ---- what the fitted curve says -----------------------------------------
    print('\nFitted trend')
    print(f'  at time {t_lo:g} : rate = {g_rate[0]:.6g}')
    print(f'  at time {t_hi:g} : rate = {g_rate[-1]:.6g}')
    print(f'  overall ratio    : {g_rate[-1] / g_rate[0]:.4f}')
    i_lo, i_hi = int(np.argmin(g_rate)), int(np.argmax(g_rate))
    print(f'  lowest  fitted rate {g_rate[i_lo]:.6g} at time {grid_t[i_lo]:g}')
    print(f'  highest fitted rate {g_rate[i_hi]:.6g} at time {grid_t[i_hi]:g}')
    print(f'  change per time unit ranges from '
          f'{100 * g_slope.min():+.3f}% to {100 * g_slope.max():+.3f}%')

    print(f'\nFiles written to {os.path.abspath(OUTPUT_DIR)}')
    for path, what in written:
        print(f'  {os.path.basename(path):<32} {what}')

    print('\nTo read the curve off at other time points, either set OUTPUT_STEP '
          'to a\nfiner spacing (0.01 gives 1.00, 1.01, 1.02, ...) or list the '
          'exact points\nyou want in CUSTOM_TIMES, then run this file again.')
    return curve, obs


if __name__ == '__main__':
    main()
