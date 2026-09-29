from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from scipy.signal import find_peaks, peak_prominences, peak_widths
from scipy.stats import skew, kurtosis
import numpy as np
import pandas as pd
from scipy.signal import medfilt, savgol_filter
from scipy.integrate import trapezoid
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from skfda import FDataGrid
import pywt


class UncertaintySelector(BaseEstimator, TransformerMixin):
    def __init__(self, C=1.0, penalty='l1', threshold=None):
        self.C=C
        self.penalty=penalty
        self.threshold=threshold
        self.solver='saga' if penalty=='l1' else 'lbfgs'
    def fit(self, X, y):
        uncerts=X.shape[1]//2
        X_uncerts=X[:,:uncerts]
        self.log_reg_= LogisticRegression(C=self.C, penalty=self.penalty, solver=self.solver, random_state=42, max_iter=500)
        self.log_reg_.fit(X_uncerts, y)
        self.selector_= SelectFromModel(self.log_reg_,threshold=self.threshold,prefit=True)
        self.support_mask_ = self.selector_.get_support()
        return self
    def transform(self, X):
        check_is_fitted(self, ['support_mask_'])
        uncerts=X.shape[1]//2
        X_vels=X[:,uncerts:]
        return self.selector_.transform(X_vels)

class Spall_GlobalGeometryFeatureExtractor(BaseEstimator, TransformerMixin):

    def __init__(self,
                 sigma=50,
                 prominence=None,
                 plateau_slope_thresh=0.01,
                 plateau_min_length=5,
                 hist_bins=10):

        self.sigma = sigma
        self.prominence = prominence
        self.plateau_slope_thresh = plateau_slope_thresh
        self.plateau_min_length = plateau_min_length
        self.hist_bins = hist_bins

    def fit(self, X, y=None):
        return self

    ####################################################################
    # Utilities
    ####################################################################

    def smooth(self, trace):
        return gaussian_filter1d(trace.astype(float), self.sigma)

    def plateau_lengths(self, slope):

        flat = np.abs(slope) < self.plateau_slope_thresh

        lengths = []
        count = 0

        for f in flat:

            if f:
                count += 1
            else:
                if count >= self.plateau_min_length:
                    lengths.append(count)
                count = 0

        if count >= self.plateau_min_length:
            lengths.append(count)

        return np.array(lengths)

    ####################################################################
    # Feature Extraction
    ####################################################################

    def extract_trace_features(self, trace):

        trace = self.smooth(trace)

        slope = np.gradient(trace)
        curvature = np.gradient(slope)

        ################################################################
        # Peaks
        ################################################################

        peaks, _ = find_peaks(
            trace,
            prominence=self.prominence
        )

        valleys, _ = find_peaks(
            -trace,
            prominence=self.prominence
        )

        peak_prom = (
            peak_prominences(trace, peaks)[0]
            if len(peaks)
            else np.array([0])
        )

        valley_prom = (
            peak_prominences(-trace, valleys)[0]
            if len(valleys)
            else np.array([0])
        )

        peak_width = (
            peak_widths(trace, peaks)[0]
            if len(peaks)
            else np.array([0])
        )

        valley_width = (
            peak_widths(-trace, valleys)[0]
            if len(valleys)
            else np.array([0])
        )

        ################################################################
        # Dominant peak/valley
        ################################################################

        if len(peaks):

            p = peaks[np.argmax(peak_prom)]

        else:

            p = np.argmax(trace)

        if len(valleys):

            v = valleys[np.argmax(valley_prom)]

        else:

            v = np.argmin(trace)

        dt = abs(p - v)
        dv = abs(trace[p] - trace[v])

        slope_between = (trace[p] - trace[v]) / (dt + 1e-12)

        segment = trace[min(p, v):max(p, v)+1]

        area = np.trapezoid(segment)

        arc = np.sum(
            np.sqrt(
                1 +
                np.diff(segment)**2
            )
        )

        ################################################################
        # Plateaus
        ################################################################

        plateaus = self.plateau_lengths(slope)

        ################################################################
        # Inflection points
        ################################################################

        inflections = np.where(
            np.diff(np.sign(curvature))
        )[0]

        ################################################################
        # Histograms
        ################################################################

        slope_hist, _ = np.histogram(
            slope,
            bins=self.hist_bins,
            density=True
        )

        curvature_hist, _ = np.histogram(
            curvature,
            bins=self.hist_bins,
            density=True
        )

        ################################################################
        # Complexity
        ################################################################

        total_variation = np.sum(np.abs(np.diff(trace)))

        arc_length = np.sum(
            np.sqrt(
                1 +
                np.diff(trace)**2
            )
        )

        energy = np.sum(trace**2)

        ################################################################
        # Assemble feature vector
        ################################################################

        features = [

            ##############################
            # extrema
            ##############################

            len(peaks),
            len(valleys),

            np.max(peak_prom),
            np.max(valley_prom),

            np.mean(peak_prom),
            np.mean(valley_prom),

            np.std(peak_prom),
            np.std(valley_prom),

            np.max(peak_width),
            np.max(valley_width),

            ##############################
            # peak-valley
            ##############################

            dt,
            dv,
            slope_between,
            area,
            arc,

            ##############################
            # plateaus
            ##############################

            len(plateaus),

            np.max(plateaus) if len(plateaus) else 0,

            np.mean(plateaus) if len(plateaus) else 0,

            np.std(plateaus) if len(plateaus) else 0,

            np.sum(plateaus) / len(trace),

            ##############################
            # inflections
            ##############################

            len(inflections),

            np.mean(np.diff(inflections))
            if len(inflections) > 1 else 0,

            np.std(np.diff(inflections))
            if len(inflections) > 1 else 0,

            ##############################
            # slope stats
            ##############################

            np.mean(slope),
            np.std(slope),
            np.max(slope),
            np.min(slope),
            skew(slope),
            kurtosis(slope),

            ##############################
            # curvature stats
            ##############################

            np.mean(curvature),
            np.std(curvature),
            np.max(curvature),
            np.min(curvature),
            skew(curvature),
            kurtosis(curvature),

            ##############################
            # complexity
            ##############################

            total_variation,
            arc_length,
            energy,
            np.max(trace) - np.min(trace)

        ]

        features.extend(slope_hist)
        features.extend(curvature_hist)

        return np.asarray(features)

    ####################################################################
    # sklearn interface
    ####################################################################

    def transform(self, X):

        return np.vstack(
            [
                self.extract_trace_features(trace)
                for trace in X
            ]
        )

class HELBreakpointFeaturizer(BaseEstimator, TransformerMixin):
    """
    For a batch of free-surface velocity (FSV) traces, scan each trace
    (restricted to the region before its own peak) for candidate HEL
    breakpoints using continuous piecewise-linear ("hinge") regression,
    and return the concatenated metrics of the top-K breakpoint
    candidates as one feature vector per trace.

    Parameters
    ----------
    top_k : int, default=3
        Number of best breakpoint candidates per trace to include in the
        output feature vector. Missing slots (if a trace's segment is too
        short) are filled with NaN -- use an imputer downstream.
    min_separation : int, default=3
        Minimum spacing, in samples, enforced between selected top_k
        breakpoints, so they aren't all clustered on the same kink.
    min_segment : int, default=4
        Absolute floor on samples required on EACH side of a candidate
        breakpoint. The effective minimum used is
        max(min_segment, ceil(min_segment_frac * n)).
    min_segment_frac : float, default=0.15
        Minimum segment size on each side of a breakpoint, as a fraction
        of the (post-trimming) segment length. Prevents candidates right
        at the edge of the search region from winning by trivially
        fitting a near-perfect line through just a handful of points.
    stride : int, default=1
        Step between candidate breakpoint indices tested. Use stride > 1
        to speed up scanning of long traces.
    dt : float, default=1.0
        Sample spacing (time between consecutive samples).
    restrict_to_pre_peak : bool, default=True
        If True (recommended), each trace is truncated to end at its main
        peak before scanning. The peak is found with scipy's find_peaks
        on a smoothed copy of the trace (window = peak_smoothing_window),
        scoring each detected peak by prominence x height and keeping the
        highest-scoring one -- robust to noise spikes and post-peak
        ringdown overshoot, unlike a raw argmax.
    peak_smoothing_window : int, default=5
        Width (in samples) of the moving-average filter used only for
        peak-finding. Must be odd; even values are incremented by 1.
        Does not affect the data used for the breakpoint fit itself.

    Attributes (set after fit/transform)
    -------------------------------------
    breakpoints_ : list of list of dict
        breakpoints_[i] = every candidate breakpoint evaluated for trace i.
    top_breakpoints_ : list of list of dict
        top_breakpoints_[i] = the top_k breakpoints for trace i (padded
        with None if fewer than top_k were found).

    Per-breakpoint metrics (each becomes `b{rank}_<name>` in the output)
    ------------------------------------------------------------------
    t_break                   : candidate HEL time (the hinge location)
    idx_break                  : sample index of the candidate
    v_at_break                  : fitted velocity at the breakpoint
    slope_left, slope_right   : dv/dt before / after the breakpoint
    slope_change                : slope_right - slope_left (large
                                   magnitude = sharp kink)
    sse                          : residual sum of squares of the fit
    r2                            : 1 - sse / total variance of the segment
    improvement_over_line        : fractional SSE reduction vs. the best
                                    single straight-line fit (near zero =
                                    probably not a real kink)
    n_left, n_right               : samples on each side of the breakpoint
    """

    _METRIC_NAMES = [
        't_break', 'idx_break', 'v_at_break',
        'slope_left', 'slope_right', 'slope_change',
        'sse', 'r2', 'improvement_over_line',
        'n_left', 'n_right',
    ]

    def __init__(self, top_k=3, min_separation=3, min_segment=4, min_segment_frac=0.15,
                 stride=20, dt=20, restrict_to_pre_peak=True, peak_smoothing_window=100):
        self.top_k = top_k
        self.min_separation = min_separation
        self.min_segment = min_segment
        self.min_segment_frac = min_segment_frac
        self.stride = stride
        self.dt = dt
        self.restrict_to_pre_peak = restrict_to_pre_peak
        self.peak_smoothing_window = peak_smoothing_window

    # ------------------------------------------------------------------ #
    def _validate(self):
        if self.min_segment < 3:
            raise ValueError("min_segment must be >= 3 for a well-determined hinge fit.")
        if self.top_k < 1:
            raise ValueError("top_k must be >= 1.")

    def fit(self, X, y=None):
        # Stateless transformer -- fit() just validates parameters.
        self._validate()
        return self

    # ------------------------------------------------------------------ #
    def _find_peak_idx(self, v):
        """
        Find the main peak: smooth v, detect all peaks with find_peaks,
        score each by prominence x height, and return the index of the
        highest-scoring one. Falls back to raw argmax if no peak is
        detected (e.g. a monotonically rising trace with no rollover).
        """
        n = len(v)
        window = max(3, self.peak_smoothing_window)
        if window % 2 == 0:
            window += 1
        window = min(window, n if n % 2 == 1 else n - 1)
        window = max(window, 3)

        if n < window:
            return int(np.argmax(v))

        kernel = np.ones(window) / window
        v_smooth = np.convolve(v, kernel, mode='same')

        peaks, props = find_peaks(v_smooth, prominence=0)
        if len(peaks) == 0:
            return int(np.argmax(v_smooth))

        score = props['prominences'] * v_smooth[peaks]
        return int(peaks[np.argmax(score)])

    def _single_line_sse(self, t, v):
        """SSE of the best single straight-line fit (the 'no breakpoint' baseline)."""
        A = np.column_stack([np.ones_like(t), t])
        coeffs, _, _, _ = np.linalg.lstsq(A, v, rcond=None)
        resid = v - A @ coeffs
        return float(np.sum(resid ** 2))

    def _hinge_fit(self, t, v, tau):
        """Fit v = b0 + b1*t + b2*max(0, t - tau); return coeffs and SSE."""
        hinge = np.clip(t - tau, 0.0, None)
        A = np.column_stack([np.ones_like(t), t, hinge])
        coeffs, _, _, _ = np.linalg.lstsq(A, v, rcond=None)
        resid = v - A @ coeffs
        sse = float(np.sum(resid ** 2))
        return coeffs, sse

    def _scan_trace(self, t, v):
        self._validate()

        if self.restrict_to_pre_peak:
            peak_idx = self._find_peak_idx(v)
            t = t[:peak_idx + 1]
            v = v[:peak_idx + 1]

        n = len(t)
        candidates = []

        effective_min_segment = max(self.min_segment, int(np.ceil(self.min_segment_frac * n)))
        lo, hi = effective_min_segment, n - effective_min_segment
        if hi > lo:
            ss_tot = float(np.sum((v - np.mean(v)) ** 2))
            single_line_sse = self._single_line_sse(t, v)

            for idx in range(lo, hi, self.stride):
                tau = t[idx]
                coeffs, sse = self._hinge_fit(t, v, tau)
                b0, b1, b2 = coeffs
                slope_left = float(b1)
                slope_right = float(b1 + b2)
                v_at_break = float(b0 + b1 * tau)
                r2 = 1.0 - sse / ss_tot if ss_tot > 0 else np.nan
                improvement = (
                    (single_line_sse - sse) / single_line_sse
                    if single_line_sse > 0 else np.nan
                )
                candidates.append({
                    't_break': float(tau),
                    'idx_break': idx,
                    'v_at_break': v_at_break,
                    'slope_left': slope_left,
                    'slope_right': slope_right,
                    'slope_change': float(b2),
                    'sse': sse,
                    'r2': r2,
                    'improvement_over_line': improvement,
                    'n_left': idx,
                    'n_right': n - idx,
                })

        # rank by lowest SSE (cleanest kink fit), enforcing min_separation
        # so the top_k aren't all clustered on the same true kink
        ranked = sorted(candidates, key=lambda c: c['sse'])
        selected = []
        for c in ranked:
            if all(abs(c['idx_break'] - s['idx_break']) >= self.min_separation
                   for s in selected):
                selected.append(c)
            if len(selected) == self.top_k:
                break
        selected += [None] * (self.top_k - len(selected))

        return candidates, selected

    def _candidate_to_vector(self, c):
        if c is None:
            return [np.nan] * len(self._METRIC_NAMES)
        return [c[name] for name in self._METRIC_NAMES]

    # ------------------------------------------------------------------ #
    def transform(self, X, y=None):
        """
        X : array-like, shape (n_traces, n_samples)
            Each row is one FSV trace.

        Returns
        -------
        np.ndarray, shape (n_traces, top_k * n_features)
        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(
                "X must be 2D with shape (n_traces, n_samples). "
                "Reshape a single trace with X.reshape(1, -1)."
            )
        n_traces, n_samples = X.shape
        t = np.arange(n_samples) * self.dt

        all_candidates, all_top, feature_rows = [], [], []
        for i in range(n_traces):
            candidates, top = self._scan_trace(t, X[i])
            all_candidates.append(candidates)
            all_top.append(top)
            row = []
            for c in top:
                row.extend(self._candidate_to_vector(c))
            feature_rows.append(row)

        self.breakpoints_ = all_candidates
        self.top_breakpoints_ = all_top
        return np.array(feature_rows, dtype=float)

    def get_feature_names_out(self, input_features=None):
        """Column labels matching transform()'s output, e.g. 'b1_slope_change', 'b2_t_break', ..."""
        names = []
        for rank in range(1, self.top_k + 1):
            names.extend(f"b{rank}_{m}" for m in self._METRIC_NAMES)
        return np.array(names)

    def top_breakpoints_df(self, trace_index):
        """Convenience: top_k breakpoints for one trace as a DataFrame (for inspection/plotting)."""
        if not hasattr(self, 'top_breakpoints_'):
            raise RuntimeError("Call fit_transform()/transform() first.")
        rows = [c for c in self.top_breakpoints_[trace_index] if c is not None]
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------- #
class ToFDataGrid(BaseEstimator, TransformerMixin):
    def __init__(self, grid_points=None):
        self.grid_points = grid_points

    def fit(self, X, y=None):
        if self.grid_points is None:
            self.grid_points_ = np.arange(X.shape[1])
        else:
            self.grid_points_ = self.grid_points
        return self

    def transform(self, X):
        return FDataGrid(data_matrix=X, grid_points=self.grid_points_)

#from skfda import FDataGrid
#from skfda.preprocessing.dim_reduction.feature_extraction import FPCA
from sklearn.ensemble import HistGradientBoostingClassifier


class FPCAHGBCModel:
    def __init__(self, n_components=5, grid_points=None, random_state=42, **hgbc_kwargs):
        self.n_components = n_components
        self.grid_points = grid_points
        self.random_state = random_state
        self.hgbc_kwargs = hgbc_kwargs

        self.fpca = None
        self.clf = None

    def fit(self, X, y):
        if self.grid_points is None:
            self.grid_points_ = np.arange(X.shape[1])
        else:
            self.grid_points_ = self.grid_points

        fd = FDataGrid(data_matrix=X, grid_points=self.grid_points_)

        self.fpca = FPCA(n_components=self.n_components)
        X_fpca = self.fpca.fit_transform(fd)

        self.clf = HistGradientBoostingClassifier(
            random_state=self.random_state,
            **self.hgbc_kwargs
        )
        self.clf.fit(X_fpca, y)
        return self

    def transform(self, X):
        fd = FDataGrid(data_matrix=X, grid_points=self.grid_points_)
        return self.fpca.transform(fd)

    def predict(self, X):
        X_fpca = self.transform(X)
        return self.clf.predict(X_fpca)

    def predict_proba(self, X):
        X_fpca = self.transform(X)
        return self.clf.predict_proba(X_fpca)

class WaveletFeatureExtractor(BaseEstimator, TransformerMixin):
    """Custom Scikit-Learn Transformer to extract wavelet-based features

    from 1D signals (e.g., velocity-time traces for HEL detection).
    """

    def __init__(self, wavelet="db4", level=3):
        self.wavelet = wavelet
        self.level = level

    def fit(self, X, y=None):
        # Unsupervised feature extraction; no fitting required
        return self

    def transform(self, X):
        """Extracts energy, relative energy, and standard deviation from

        approximation and detail wavelet coefficients for each sample.
        """
        X = np.asarray(X)
        features = []

        for signal in X:
            # Perform Discrete Wavelet Transform (DWT) decomposition
            coeffs = pywt.wavedec(
                signal, wavelet=self.wavelet, level=self.level
            )

            sample_features = []
            energies = []

            # 1. Compute Energy and Standard Deviation for each coefficient array
            for c in coeffs:
                energy = np.sum(c**2)
                std_dev = np.std(c)

                energies.append(energy)
                sample_features.extend([energy, std_dev])

            # 2. Compute Relative Energies (Normalized against total energy)
            total_energy = np.sum(energies) if np.sum(energies) > 0 else 1.0
            relative_energies = [e / total_energy for e in energies]

            sample_features.extend(relative_energies)
            features.append(sample_features)

        return np.array(features)

#import numpy as np
#import pywt
#from sklearn.base import BaseEstimator, TransformerMixin


class HELPresenceFeatureExtractorWavelet(BaseEstimator, TransformerMixin):
    """
    Extracts features indicating whether a structural break (HEL) is
    present in a free-surface velocity trace, WITHOUT needing to locate it.

    Designed for binary classification: HEL present vs. not present.
    """

    def __init__(self, dt=1.0, wavelet='cmor1.5-1.0',
                 scales=None, min_segment=5,
                 baseline_frac=0.05, plateau_frac=0.1):
        self.dt = dt
        self.wavelet = wavelet
        self.scales = scales if scales is not None else np.arange(2, 64)
        self.min_segment = min_segment
        self.baseline_frac = baseline_frac
        self.plateau_frac = plateau_frac

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return np.array([self._extract(trace) for trace in X])

    # ------------------------------------------------------------------
    def _extract(self, signal):
        signal = np.asarray(signal, dtype=float)
        n = len(signal)
        t = np.arange(n) * self.dt

        # --- Identify rise region self-referentially (no metadata) ---
        n_base = max(1, int(self.baseline_frac * n))
        n_plateau = max(1, int(self.plateau_frac * n))
        baseline = np.median(signal[:n_base])
        plateau = np.median(signal[-n_plateau:])
        amp = plateau - baseline + 1e-12

        onset_idx, end_idx = self._find_rise_region(signal, baseline, plateau)
        seg = signal[onset_idx:end_idx]
        t_seg = t[onset_idx:end_idx]
        n_seg = len(seg)

        if n_seg < 2 * self.min_segment:
            # too short to meaningfully test for a break
            return self._null_features()

        # --- Sup-F / Chow test scanned over all candidate breakpoints ---
        f_stats, resid_ratios = self._scan_breaks(seg, t_seg)
        max_f = np.max(f_stats) if len(f_stats) else 0.0
        max_resid_ratio = np.max(resid_ratios) if len(resid_ratios) else 0.0

        # --- WTMM persistence (max singularity strength anywhere in segment) ---
        persistence_max, holder_at_max = self._wtmm_features(seg)

        # --- Residual shape features (single-line fit residual distribution) ---
        skew, kurt = self._residual_shape(seg, t_seg)

        # --- Normalize amplitude-dependent quantities ---
        max_f_norm = max_f  # F-stat already unit-free
        persistence_norm = persistence_max / (amp + 1e-12)

        return [
            max_f_norm,
            max_resid_ratio,
            persistence_norm,
            holder_at_max,
            skew,
            kurt,
            n_seg,  # sometimes useful: how long is the rise region itself
        ]

    def _null_features(self):
        return [0.0, 0.0, 0.0, np.nan, 0.0, 0.0, 0]

    # ------------------------------------------------------------------
    def _find_rise_region(self, signal, baseline, plateau):
        thresh_lo = baseline + 0.05 * (plateau - baseline)
        thresh_hi = baseline + 0.95 * (plateau - baseline)
        above_lo = np.where(signal > thresh_lo)[0]
        above_hi = np.where(signal > thresh_hi)[0]
        onset = above_lo[0] if len(above_lo) else 0
        end = above_hi[0] if len(above_hi) else len(signal) - 1
        return onset, max(end, onset + 2 * self.min_segment)

    def _scan_breaks(self, seg, t_seg):
        n = len(seg)
        f_stats, resid_ratios = [], []

        # null model: single line fit over whole segment
        A_full = np.vstack([t_seg, np.ones_like(t_seg)]).T
        coef_full, *_ = np.linalg.lstsq(A_full, seg, rcond=None)
        rss_pooled = np.sum((A_full @ coef_full - seg) ** 2) + 1e-12

        for idx in range(self.min_segment, n - self.min_segment):
            t1, y1 = t_seg[:idx], seg[:idx]
            t2, y2 = t_seg[idx:], seg[idx:]

            A1 = np.vstack([t1, np.ones_like(t1)]).T
            A2 = np.vstack([t2, np.ones_like(t2)]).T

            c1, *_ = np.linalg.lstsq(A1, y1, rcond=None)
            c2, *_ = np.linalg.lstsq(A2, y2, rcond=None)

            rss1 = np.sum((A1 @ c1 - y1) ** 2)
            rss2 = np.sum((A2 @ c2 - y2) ** 2)
            rss_split = rss1 + rss2 + 1e-12

            k = 2
            f_stat = ((rss_pooled - rss_split) / k) / (rss_split / (n - 2 * k))
            resid_ratio = (rss_pooled - rss_split) / rss_pooled

            f_stats.append(f_stat)
            resid_ratios.append(resid_ratio)

        return np.array(f_stats), np.array(resid_ratios)

    def _wtmm_features(self, seg):
        if len(seg) < 8:
            return 0.0, np.nan
        coeffs, _ = pywt.cwt(seg, self.scales, self.wavelet,
                              sampling_period=self.dt)
        modulus = np.abs(coeffs)
        # persistence score per time index: sum of modulus across scales
        persistence = modulus.sum(axis=0)
        max_idx = np.argmax(persistence)
        max_val = persistence[max_idx]

        # holder exponent at the point of strongest persistence
        amps = modulus[:, max_idx]
        valid = amps > 1e-12
        if valid.sum() >= 2:
            log_s = np.log(self.scales[valid])
            log_a = np.log(amps[valid])
            holder, _ = np.polyfit(log_s, log_a, 1)
        else:
            holder = np.nan

        return max_val, holder

    def _residual_shape(self, seg, t_seg):
        A = np.vstack([t_seg, np.ones_like(t_seg)]).T
        coef, *_ = np.linalg.lstsq(A, seg, rcond=None)
        resid = seg - A @ coef
        std = resid.std() + 1e-12
        skew = np.mean(((resid - resid.mean()) / std) ** 3)
        kurt = np.mean(((resid - resid.mean()) / std) ** 4) - 3
        return skew, kurt

import numpy as np
from scipy.signal import savgol_filter
from sklearn.base import BaseEstimator, TransformerMixin


class ResidualHELFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Detects HEL signature in a free-surface velocity trace by fitting a
    single straight line from shock onset to peak compression, then
    characterizing the residual pattern (deviation from linearity).

    Physical motivation
    --------------------
    If the trace is purely elastic-to-plastic with NO distinct HEL, the
    rise from onset to peak is close to a single straight line (or a
    single smooth curve) and residuals from a linear fit will be small
    and unstructured.

    If a genuine HEL is present, the trace bends (elastic slope -> plastic
    slope), so forcing a single straight line through the whole rise
    creates a systematic, structured residual: positive on one side of
    the kink, negative on the other (or vice versa) -- i.e., an S-shaped
    or hump-shaped residual pattern, NOT random noise.

    This transformer quantifies that residual structure without ever
    needing to locate the breakpoint itself.

    Parameters
    ----------
    dt : float
        Time step between samples.
    smooth : bool
        Whether to apply Savitzky-Golay smoothing before fitting/onset
        detection (recommended if data is noisy).
    smooth_window, smooth_poly : int
        Savitzky-Golay filter parameters.
    onset_frac, peak_frac : float
        Fractional thresholds (relative to baseline->peak amplitude) used
        to self-referentially locate onset and peak-compression indices,
        without needing external metadata.
    """

    def __init__(self, dt=1.0, smooth=True, smooth_window=11,
                 smooth_poly=3, onset_frac=0.05, peak_frac=0.98):
        self.dt = dt
        self.smooth = smooth
        self.smooth_window = smooth_window
        self.smooth_poly = smooth_poly
        self.onset_frac = onset_frac
        self.peak_frac = peak_frac

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return np.array([self._extract(trace) for trace in X])

    # ------------------------------------------------------------------
    def _extract(self, signal):
        signal = np.asarray(signal, dtype=float)
        n = len(signal)

        if self.smooth and n > self.smooth_window:
            sig_smooth = savgol_filter(
                signal, self.smooth_window, self.smooth_poly
            )
        else:
            sig_smooth = signal

        onset_idx, peak_idx = self._find_onset_and_peak(sig_smooth)

        if peak_idx - onset_idx < 5:
            # segment too short to fit meaningfully
            return self._null_features()

        seg = sig_smooth[onset_idx:peak_idx + 1]
        t_seg = np.arange(len(seg)) * self.dt

        return self._residual_features(seg, t_seg)

    # ------------------------------------------------------------------
    def _find_onset_and_peak(self, signal):
        baseline = np.median(signal[:max(1, int(0.05 * len(signal)))])
        peak_val = np.max(signal)
        amp = peak_val - baseline + 1e-12

        thresh_onset = baseline + self.onset_frac * amp
        thresh_peak = baseline + self.peak_frac * amp

        above_onset = np.where(signal > thresh_onset)[0]
        above_peak = np.where(signal > thresh_peak)[0]

        onset_idx = above_onset[0] if len(above_onset) else 0
        peak_idx = above_peak[0] if len(above_peak) else len(signal) - 1

        # fallback if peak comes before onset somehow
        if peak_idx <= onset_idx:
            peak_idx = np.argmax(signal)

        return onset_idx, peak_idx

    # ------------------------------------------------------------------
    def _residual_features(self, seg, t_seg):
        n = len(seg)
        amp = (seg.max() - seg.min()) + 1e-12
        duration = (t_seg[-1] - t_seg[0]) + 1e-12

        # --- Fit single straight line, onset -> peak ---
        A = np.vstack([t_seg, np.ones_like(t_seg)]).T
        coef, _, _, _ = np.linalg.lstsq(A, seg, rcond=None)
        fit_line = A @ coef
        resid = seg - fit_line

        # Normalize residual by trace amplitude (scale-free)
        resid_norm = resid / amp

        # --- Basic magnitude features ---
        max_abs_resid = np.max(np.abs(resid_norm))
        rms_resid = np.sqrt(np.mean(resid_norm ** 2))

        # --- Shape features: is the residual structured or random? ---
        # A real kink produces a smooth hump/S-curve, not noise.
        # Skewness/kurtosis of residual distribution:
        std = resid_norm.std() + 1e-12
        skew = np.mean(((resid_norm - resid_norm.mean()) / std) ** 3)
        kurt = np.mean(((resid_norm - resid_norm.mean()) / std) ** 4) - 3

        # --- Sign-change count (structured residuals have FEW sign
        # changes -- e.g., one big hump; noise has MANY sign changes) ---
        signs = np.sign(resid)
        sign_changes = np.sum(np.abs(np.diff(signs)) > 0)
        # normalize by segment length for scale-invariance across
        # different trace lengths
        sign_change_rate = sign_changes / n

        # --- Autocorrelation at lag 1: structured residuals are smooth
        # (high positive autocorrelation); pure noise has low/no
        # autocorrelation ---
        if n > 2:
            resid_centered = resid_norm - resid_norm.mean()
            denom = np.sum(resid_centered ** 2) + 1e-12
            autocorr_lag1 = np.sum(
                resid_centered[:-1] * resid_centered[1:]
            ) / denom
        else:
            autocorr_lag1 = 0.0

        # --- Location of max positive and max negative residual ---
        # (captures "where in the segment" the bend happens, without
        # needing an explicit breakpoint search)
        idx_max_pos = np.argmax(resid_norm)
        idx_max_neg = np.argmin(resid_norm)
        frac_pos = idx_max_pos / n
        frac_neg = idx_max_neg / n

        # --- Energy ratio: how much of total variance is explained by
        # the linear fit vs. left as residual (low R^2 => more likely
        # a genuine nonlinearity/kink exists) ---
        ss_tot = np.sum((seg - seg.mean()) ** 2) + 1e-12
        ss_res = np.sum(resid ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        return [
            max_abs_resid,
            rms_resid,
            skew,
            kurt,
            sign_change_rate,
            autocorr_lag1,
            frac_pos,
            frac_neg,
            r_squared,
            n,  # segment length, weak auxiliary feature
        ]

    def _null_features(self):
        return [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0]

    # ------------------------------------------------------------------
    def get_feature_names_out(self, input_features=None):
        return np.array([
            'max_abs_resid',
            'rms_resid',
            'resid_skew',
            'resid_kurt',
            'sign_change_rate',
            'autocorr_lag1',
            'frac_pos_peak_loc',
            'frac_neg_peak_loc',
            'r_squared',
            'segment_length',
        ])

import numpy as np
from scipy.signal import savgol_filter
from sklearn.base import BaseEstimator, TransformerMixin


class WeightedResidualHELFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts HEL-detection features using a WEIGHTED linear fit
    (weights = 1/uncertainty^2) from shock onset to peak compression.

    Input format
    -------------
    X is expected to be a 2D array-like of shape (n_samples, 2*L), where
    for each row:
        - the first L values are the velocity trace
        - the last L values are the corresponding uncertainty trace
    (both on the same time base, length L each).

    Parameters
    ----------
    dt : float
        Time step between samples.
    smooth : bool
        Whether to apply Savitzky-Golay smoothing before onset/peak
        detection (uses velocity only; uncertainty is left untouched).
    smooth_window, smooth_poly : int
        Savitzky-Golay filter parameters.
    onset_frac, peak_frac : float
        Fractional thresholds used to self-referentially locate onset
        and peak-compression indices.
    min_uncert : float
        Floor applied to uncertainty values to avoid divide-by-zero /
        exploding weights.
    """

    def __init__(self, dt=1.0, smooth=True, smooth_window=11,
                 smooth_poly=3, onset_frac=0.05, peak_frac=0.98,
                 eps=1e-12, min_uncert=1e-6):
        self.dt = dt
        self.smooth = smooth
        self.smooth_window = smooth_window
        self.smooth_poly = smooth_poly
        self.onset_frac = onset_frac
        self.peak_frac = peak_frac
        self.eps = eps
        self.min_uncert = min_uncert

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        '''X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        n_cols = X.shape[1]
        if n_cols % 2 != 0:
            raise ValueError(
                f"Expected an even number of columns (velocity + "
                f"uncertainty concatenated), got {n_cols}."
            )
        L = n_cols // 2'''

        features = []
        for row in X:
            row=np.asarray(row, dtype=float) 
            L=len(row)//2
            velocity = row[:L]
            uncertainty = row[L:]
            features.append(self._extract(velocity, uncertainty))

        return np.array(features)

    # ------------------------------------------------------------------
    def _extract(self, velocity, uncertainty):
        velocity = np.asarray(velocity, dtype=float)
        uncertainty = np.asarray(uncertainty, dtype=float)
        n = len(velocity)

        if self.smooth and n > self.smooth_window:
            sig_smooth = savgol_filter(
                velocity, self.smooth_window, self.smooth_poly
            )
        else:
            sig_smooth = velocity

        onset_idx, peak_idx = self._find_onset_and_peak(sig_smooth)

        if peak_idx - onset_idx < 5:
            return self._null_features()

        seg = sig_smooth[onset_idx:peak_idx + 1]
        uncert_seg = uncertainty[onset_idx:peak_idx + 1]
        t_seg = np.arange(len(seg)) * self.dt

        uncert_seg = np.clip(uncert_seg, self.min_uncert, None)

        return self._weighted_residual_features(seg, uncert_seg, t_seg)

    # ------------------------------------------------------------------
    def _find_onset_and_peak(self, signal):
        baseline = np.median(signal[:max(1, int(0.05 * len(signal)))])
        peak_val = np.max(signal)
        amp = peak_val - baseline + self.eps

        thresh_onset = baseline + self.onset_frac * amp
        thresh_peak = baseline + self.peak_frac * amp

        above_onset = np.where(signal > thresh_onset)[0]
        above_peak = np.where(signal > thresh_peak)[0]

        onset_idx = above_onset[0] if len(above_onset) else 0
        peak_idx = above_peak[0] if len(above_peak) else len(signal) - 1

        if peak_idx <= onset_idx:
            peak_idx = np.argmax(signal)

        return onset_idx, peak_idx

    # ------------------------------------------------------------------
    def _weighted_residual_features(self, seg, uncert_seg, t_seg):
        n = len(seg)
        amp = (seg.max() - seg.min()) + self.eps

        w = 1.0 / (uncert_seg ** 2)

        A = np.vstack([t_seg, np.ones_like(t_seg)]).T
        W = np.diag(w)

        AtWA = A.T @ W @ A
        AtWy = A.T @ W @ seg
        coef = np.linalg.solve(AtWA, AtWy)

        fit_line = A @ coef
        resid = seg - fit_line
        resid_norm = resid / amp

        chi_sq = np.sum((resid / uncert_seg) ** 2)
        dof = max(n - 2, 1)
        chi_sq_reduced = chi_sq / dof

        y_mean_w = np.sum(w * seg) / np.sum(w)
        ss_tot_w = np.sum(w * (seg - y_mean_w) ** 2) + self.eps
        ss_res_w = np.sum(w * resid ** 2)
        r_squared_w = 1 - (ss_res_w / ss_tot_w)

        max_abs_resid = np.max(np.abs(resid_norm))
        rms_resid = np.sqrt(np.mean(resid_norm ** 2))

        std = resid_norm.std() + self.eps
        skew = np.mean(((resid_norm - resid_norm.mean()) / std) ** 3)
        kurt = np.mean(((resid_norm - resid_norm.mean()) / std) ** 4) - 3

        signs = np.sign(resid)
        sign_changes = np.sum(np.abs(np.diff(signs)) > 0)
        sign_change_rate = sign_changes / n

        if n > 2:
            resid_centered = resid_norm - resid_norm.mean()
            denom = np.sum(resid_centered ** 2) + self.eps
            autocorr_lag1 = np.sum(
                resid_centered[:-1] * resid_centered[1:]
            ) / denom
        else:
            autocorr_lag1 = 0.0

        pointwise_snr = np.abs(resid) / uncert_seg
        snr_max = np.max(pointwise_snr)
        snr_mean = np.mean(pointwise_snr)

        frac_significant = np.mean(pointwise_snr > 1.0)

        significant = pointwise_snr > 1.0
        longest_run = self._longest_true_run(significant) / n

        idx_max_pos = np.argmax(resid)
        idx_max_neg = np.argmin(resid)
        frac_pos = idx_max_pos / n
        frac_neg = idx_max_neg / n
        snr_at_pos_peak = pointwise_snr[idx_max_pos]
        snr_at_neg_peak = pointwise_snr[idx_max_neg]

        return [
            max_abs_resid,
            rms_resid,
            skew,
            kurt,
            sign_change_rate,
            autocorr_lag1,
            frac_pos,
            frac_neg,
            r_squared_w,
            chi_sq_reduced,
            snr_max,
            snr_mean,
            frac_significant,
            longest_run,
            snr_at_pos_peak,
            snr_at_neg_peak,
            n,
        ]

    @staticmethod
    def _longest_true_run(bool_array):
        if not np.any(bool_array):
            return 0
        max_run = 0
        current_run = 0
        for val in bool_array:
            if val:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 0
        return max_run

    def _null_features(self):
        return [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0]

    # ------------------------------------------------------------------
    def get_feature_names_out(self, input_features=None):
        return np.array([
            'max_abs_resid',
            'rms_resid',
            'resid_skew',
            'resid_kurt',
            'sign_change_rate',
            'autocorr_lag1',
            'frac_pos_peak_loc',
            'frac_neg_peak_loc',
            'r_squared_weighted',
            'chi_sq_reduced',
            'snr_max',
            'snr_mean',
            'frac_significant',
            'longest_significant_run',
            'snr_at_pos_peak',
            'snr_at_neg_peak',
            'segment_length',
        ])

class Spall_GlobalGeometryFeatureExtractor_uncerts(BaseEstimator, TransformerMixin):

    def __init__(self,
                 sigma=50,
                 prominence=None,
                 plateau_slope_thresh=0.01,
                 plateau_min_length=5,
                 hist_bins=10,
                 post_peak_hist_bins=5,
                 min_peak_prominence_frac=0.1):

        self.sigma = sigma
        self.prominence = prominence
        self.plateau_slope_thresh = plateau_slope_thresh
        self.plateau_min_length = plateau_min_length
        self.hist_bins = hist_bins
        self.post_peak_hist_bins = post_peak_hist_bins
        self.min_peak_prominence_frac = min_peak_prominence_frac

    def fit(self, X, y=None):
        return self

    ####################################################################
    # Utilities
    ####################################################################

    def smooth(self, trace):
        return gaussian_filter1d(trace.astype(float), self.sigma)

    def plateau_lengths(self, slope):

        flat = np.abs(slope) < self.plateau_slope_thresh

        lengths = []
        count = 0

        for f in flat:

            if f:
                count += 1
            else:
                if count >= self.plateau_min_length:
                    lengths.append(count)
                count = 0

        if count >= self.plateau_min_length:
            lengths.append(count)

        return np.array(lengths)

    def select_first_major_peak(self, trace, peaks, peak_prom):
        """
        Selects the first peak whose prominence exceeds
        min_peak_prominence_frac * (trace amplitude range).
        Falls back to the most prominent peak overall if none qualify,
        and to argmax(trace) if no peaks were detected at all.
        """

        if not len(peaks):
            return np.argmax(trace)

        amp_range = (np.max(trace) - np.min(trace)) + 1e-12
        min_prom = self.min_peak_prominence_frac * amp_range

        qualifying = np.where(peak_prom >= min_prom)[0]

        if len(qualifying):
            # first (earliest in time) peak that meets the threshold
            first_qualifying_idx = qualifying[0]
            return peaks[first_qualifying_idx]
        else:
            # nothing met the threshold -- fall back to most prominent
            return peaks[np.argmax(peak_prom)]

    ####################################################################
    # Feature Extraction
    ####################################################################

    def extract_trace_features(self, trace, uncertainty):

        raw_trace = trace.astype(float)
        trace = self.smooth(trace)

        slope = np.gradient(trace)
        curvature = np.gradient(slope)

        ################################################################
        # Peaks
        ################################################################

        peaks, _ = find_peaks(
            trace,
            prominence=self.prominence
        )

        valleys, _ = find_peaks(
            -trace,
            prominence=self.prominence
        )

        peak_prom = (
            peak_prominences(trace, peaks)[0]
            if len(peaks)
            else np.array([0])
        )

        valley_prom = (
            peak_prominences(-trace, valleys)[0]
            if len(valleys)
            else np.array([0])
        )

        peak_width = (
            peak_widths(trace, peaks)[0]
            if len(peaks)
            else np.array([0])
        )

        valley_width = (
            peak_widths(-trace, valleys)[0]
            if len(valleys)
            else np.array([0])
        )

        ################################################################
        # Dominant peak/valley
        ################################################################

        if len(peaks):

            p = peaks[np.argmax(peak_prom)]

        else:

            p = np.argmax(trace)

        if len(valleys):

            v = valleys[np.argmax(valley_prom)]

        else:

            v = np.argmin(trace)

        dt = abs(p - v)
        dv = abs(trace[p] - trace[v])

        slope_between = (trace[p] - trace[v]) / (dt + 1e-12)

        segment = trace[min(p, v):max(p, v)+1]

        area = np.trapezoid(segment)

        arc = np.sum(
            np.sqrt(
                1 +
                np.diff(segment)**2
            )
        )

        ################################################################
        # Plateaus
        ################################################################

        plateaus = self.plateau_lengths(slope)

        ################################################################
        # Inflection points
        ################################################################

        inflections = np.where(
            np.diff(np.sign(curvature))
        )[0]

        ################################################################
        # Histograms
        ################################################################

        slope_hist, _ = np.histogram(
            slope,
            bins=self.hist_bins,
            density=True
        )

        curvature_hist, _ = np.histogram(
            curvature,
            bins=self.hist_bins,
            density=True
        )

        ################################################################
        # Complexity
        ################################################################

        total_variation = np.sum(np.abs(np.diff(trace)))

        arc_length = np.sum(
            np.sqrt(
                1 +
                np.diff(trace)**2
            )
        )

        energy = np.sum(trace**2)

        ################################################################
        # Post-first-major-peak noise distribution (uncertainty-based)
        ################################################################
        # Uses a PROMINENCE-QUALIFIED first peak (initial shock arrival),
        # not simply peaks[0], to avoid locking onto small noise-driven
        # bumps early in the trace before the real shock arrives.

        first_peak_idx = self.select_first_major_peak(
            trace, peaks, peak_prom
        )

        post_peak_uncert = uncertainty[first_peak_idx:]
        post_peak_len = len(post_peak_uncert)

        if post_peak_len > 1:
            pp_mean = np.mean(post_peak_uncert)
            pp_std = np.std(post_peak_uncert)
            pp_max = np.max(post_peak_uncert)
            pp_min = np.min(post_peak_uncert)
            pp_skew = skew(post_peak_uncert)
            pp_kurt = kurtosis(post_peak_uncert)
            pp_hist, _ = np.histogram(
                post_peak_uncert,
                bins=self.post_peak_hist_bins,
                density=True
            )
        else:
            pp_mean = pp_std = pp_max = pp_min = pp_skew = pp_kurt = 0.0
            pp_hist = np.zeros(self.post_peak_hist_bins)

        post_peak_frac = post_peak_len / len(trace)

        ################################################################
        # Assemble feature vector
        ################################################################

        features = [

            ##############################
            # extrema
            ##############################

            len(peaks),
            len(valleys),

            np.max(peak_prom),
            np.max(valley_prom),

            np.mean(peak_prom),
            np.mean(valley_prom),

            np.std(peak_prom),
            np.std(valley_prom),

            np.max(peak_width),
            np.max(valley_width),

            ##############################
            # peak-valley
            ##############################

            dt,
            dv,
            slope_between,
            area,
            arc,

            ##############################
            # plateaus
            ##############################

            len(plateaus),

            np.max(plateaus) if len(plateaus) else 0,

            np.mean(plateaus) if len(plateaus) else 0,

            np.std(plateaus) if len(plateaus) else 0,

            np.sum(plateaus) / len(trace),

            ##############################
            # inflections
            ##############################

            len(inflections),

            np.mean(np.diff(inflections))
            if len(inflections) > 1 else 0,

            np.std(np.diff(inflections))
            if len(inflections) > 1 else 0,

            ##############################
            # slope stats
            ##############################

            np.mean(slope),
            np.std(slope),
            np.max(slope),
            np.min(slope),
            skew(slope),
            kurtosis(slope),

            ##############################
            # curvature stats
            ##############################

            np.mean(curvature),
            np.std(curvature),
            np.max(curvature),
            np.min(curvature),
            skew(curvature),
            kurtosis(curvature),

            ##############################
            # complexity
            ##############################

            total_variation,
            arc_length,
            energy,
            np.max(trace) - np.min(trace),

            ##############################
            # post-first-major-peak noise distribution
            ##############################

            pp_mean,
            pp_std,
            pp_max,
            pp_min,
            pp_skew,
            pp_kurt,
            post_peak_frac,

        ]

        features.extend(slope_hist)
        features.extend(curvature_hist)
        features.extend(pp_hist)

        return np.asarray(features)

    ####################################################################
    # sklearn interface
    ####################################################################

    def transform(self, X):
        '''
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        n_cols = X.shape[1]
        if n_cols % 2 != 0:
            raise ValueError(
                f"Expected an even number of columns (velocity + "
                f"uncertainty concatenated), got {n_cols}."
            )
        L = n_cols // 2'''

        features_list = []
        for row in X:
            row=row = np.asarray(row, dtype=float)   # <-- must happen BEFORE slicing
            L = len(row) // 2
            trace = row[:L]
            uncertainty = row[L:]
            features_list.append(
                self.extract_trace_features(trace, uncertainty)
            )

        return np.vstack(features_list)

    ####################################################################
    # Feature names
    ####################################################################

    def get_feature_names_out(self, input_features=None):

            base_names = [
                'n_peaks', 'n_valleys',
                'main_peak_prominence', 'main_peak_width',
                'main_valley_prominence', 'main_valley_width',
                'plateau_frac', 'n_inflections',
                'total_variation', 'arc_length', 'energy',
                'pp_mean', 'pp_std', 'pp_max', 'pp_min',
                'pp_skew', 'pp_kurt',
                'post_peak_frac',
            ]

            slope_hist_names = [
                f'slope_hist_{i}' for i in range(self.hist_bins)
            ]
            curvature_hist_names = [
                f'curvature_hist_{i}' for i in range(self.hist_bins)
            ]
            post_peak_hist_names = [
                f'post_peak_uncert_hist_{i}'
                for i in range(self.post_peak_hist_bins)
            ]

            return np.array(base_names + slope_hist_names + curvature_hist_names + post_peak_hist_names)

import numpy as np
from scipy.signal import find_peaks
from sklearn.base import BaseEstimator, TransformerMixin


class GBFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts two families of features from a concatenated
    [velocity..., uncertainty...] trace of length 2L:

    1. Step-significance / persistence features: for each consecutive
       pair of points, compute a z-score of the step size relative to
       combined uncertainty, then characterize the run-length structure
       of "statistically distinct" vs "overlapping" steps across the
       whole trace.

    2. Peak-location features: location of the global maximum, plus
       distributional statistics of all detected local peaks
       (self-normalized, prominence-filtered).

    Parameters
    ----------
    k : float
        Number of combined-sigma required for a step to be considered
        "statistically distinct" (default 1.0).
    min_peak_prominence_frac : float
        Minimum peak prominence, as a fraction of the trace's amplitude
        range, required for a local maximum to count as a "peak"
        (default 0.05).
    n_hist_bins : int
        Number of bins used for the peak-location histogram.
    """

    def __init__(self, k=1.0, min_peak_prominence_frac=0.05, n_hist_bins=10):
        self.k = k
        self.min_peak_prominence_frac = min_peak_prominence_frac
        self.n_hist_bins = n_hist_bins

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        '''X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("Expected 2D array of shape (n_samples, 2*L).")
        n_cols = X.shape[1]
        if n_cols % 2 != 0:
            raise ValueError("Expected concatenated [velocity, uncertainty] of equal length.")
        L = n_cols // 2'''

        feats = []
        for row in X:
            row = np.asarray(row, dtype=float)
            L=len(row)//2
            v = row[:L]
            u = row[L:]
            step_feats = self._step_significance_features(v, u)
            peak_feats = self._peak_features(v)
            feats.append(np.concatenate([step_feats, peak_feats]))

        return np.array(feats)

    # ------------------------------------------------------------------
    # Step significance / persistence features
    # ------------------------------------------------------------------

    def _step_significance_features(self, v, u):
        n = len(v)
        if n < 3:
            return np.zeros(10)

        denom = np.sqrt(u[:-1] ** 2 + u[1:] ** 2)
        denom[denom <= 0] = 1e-12
        z = np.diff(v) / denom

        significant = np.abs(z) > self.k
        n_steps = len(significant)

        frac_significant = np.mean(significant)

        longest_overlap_run = self._longest_run(~significant)
        longest_distinct_run = self._longest_run(significant)

        n_transitions = np.sum(np.diff(significant.astype(int)) != 0)
        transition_rate = n_transitions / n_steps

        start, end = self._longest_run_bounds(significant)
        if end > start:
            frac_loc_start = start / n_steps
            frac_loc_end = end / n_steps
            distinct_run_span = (end - start) / n_steps
            signs = np.sign(z[start:end])
            sign_consistency = np.abs(np.mean(signs))
        else:
            frac_loc_start = 0.0
            frac_loc_end = 0.0
            distinct_run_span = 0.0
            sign_consistency = 0.0

        z_abs_max = np.max(np.abs(z))
        z_abs_mean = np.mean(np.abs(z))

        return np.array([
            frac_significant,
            longest_overlap_run / n_steps,
            longest_distinct_run / n_steps,
            transition_rate,
            frac_loc_start,
            frac_loc_end,
            distinct_run_span,
            sign_consistency,
            z_abs_max,
            z_abs_mean,
        ])

    # ------------------------------------------------------------------
    # Peak location / distribution features
    # ------------------------------------------------------------------

    def _peak_features(self, v):
        n = len(v)
        v_range = np.max(v) - np.min(v)

        max_peak_loc_frac = np.argmax(v) / (n - 1) if n > 1 else 0.0

        if v_range <= 0:
            return np.concatenate([
                [max_peak_loc_frac, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                np.zeros(self.n_hist_bins),
            ])

        prominence_thresh = self.min_peak_prominence_frac * v_range
        peaks, props = find_peaks(v, prominence=prominence_thresh)

        if len(peaks) == 0:
            n_peaks = 0
            peak_loc_mean = 0.0
            peak_loc_std = 0.0
            peak_loc_entropy = 0.0
            frac_first_half = 0.0
            frac_second_half = 0.0
            mean_prom_norm = 0.0
            max_prom_norm = 0.0
            mean_spacing = 0.0
            std_spacing = 0.0
            max_peak_dist_from_mean = 0.0
            hist = np.zeros(self.n_hist_bins)
        else:
            peak_locs_frac = peaks / (n - 1) if n > 1 else np.zeros_like(peaks, dtype=float)
            n_peaks = len(peaks)

            hist_counts, _ = np.histogram(peak_locs_frac, bins=self.n_hist_bins, range=(0, 1))
            hist = hist_counts / n_peaks

            peak_loc_mean = np.mean(peak_locs_frac)
            peak_loc_std = np.std(peak_locs_frac)
            peak_loc_entropy = self._entropy(hist_counts)

            frac_first_half = np.mean(peak_locs_frac < 0.5)
            frac_second_half = np.mean(peak_locs_frac >= 0.5)

            prominences = props['prominences']
            mean_prom_norm = np.mean(prominences) / v_range
            max_prom_norm = np.max(prominences) / v_range

            if n_peaks > 1:
                spacing = np.diff(np.sort(peak_locs_frac))
                mean_spacing = np.mean(spacing)
                std_spacing = np.std(spacing)
            else:
                mean_spacing = 0.0
                std_spacing = 0.0

            max_peak_dist_from_mean = np.abs(max_peak_loc_frac - peak_loc_mean)

        peak_feats = np.array([
            max_peak_loc_frac,
            n_peaks,
            peak_loc_mean,
            peak_loc_std,
            peak_loc_entropy,
            frac_first_half,
            frac_second_half,
            mean_prom_norm,
            max_prom_norm,
            mean_spacing,
            std_spacing,
        ])

        return np.concatenate([peak_feats, hist])

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _longest_run(bool_array):
        if len(bool_array) == 0:
            return 0
        max_run = 0
        current_run = 0
        for val in bool_array:
            if val:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 0
        return max_run

    @staticmethod
    def _longest_run_bounds(bool_array):
        """Return (start, end) index bounds [start, end) of the longest True run."""
        if len(bool_array) == 0:
            return 0, 0
        max_run = 0
        current_run = 0
        current_start = 0
        best_start = 0
        best_end = 0
        for i, val in enumerate(bool_array):
            if val:
                if current_run == 0:
                    current_start = i
                current_run += 1
                if current_run > max_run:
                    max_run = current_run
                    best_start = current_start
                    best_end = i + 1
            else:
                current_run = 0
        return best_start, best_end

    @staticmethod
    def _entropy(counts):
        counts = np.asarray(counts, dtype=float)
        total = np.sum(counts)
        if total <= 0:
            return 0.0
        p = counts / total
        p = p[p > 0]
        return -np.sum(p * np.log(p))

    # ------------------------------------------------------------------
    # Feature names
    # ------------------------------------------------------------------

    def get_feature_names_out(self, input_features=None):
        step_names = [
            'frac_significant_steps',
            'longest_overlap_run_frac',
            'longest_distinct_run_frac',
            'transition_rate',
            'distinct_run_loc_start',
            'distinct_run_loc_end',
            'distinct_run_span',
            'distinct_run_sign_consistency',
            'z_abs_max',
            'z_abs_mean',
        ]

        peak_names = [
            'max_peak_loc_frac',
            'n_peaks',
            'peak_loc_mean',
            'peak_loc_std',
            'peak_loc_entropy',
            'frac_peaks_first_half',
            'frac_peaks_second_half',
            'mean_peak_prominence_norm',
            'max_peak_prominence_norm',
            'mean_peak_spacing',
            'std_peak_spacing',
        ]

        hist_names = [f'peak_loc_hist_{i}' for i in range(self.n_hist_bins)]

        return np.array(step_names + peak_names + hist_names)


