import numpy as np
from astropy.table import Table
from lightkurve import LightCurve
from astropy.timeseries import LombScargle, BoxLeastSquares
from scipy.signal import find_peaks, correlate
from scipy.ndimage import gaussian_filter1d
import clara_benchmark.clara.astrodata_helpers.clara2_phase_fold as c2pf


def count_ls_peaks(time, flux, max_points=8000, min_period=0.2, max_period=20, power_prominence=0.05):
    if len(time) < 100:
        return 0, None, None, None
    time, flux = time[:max_points], flux[:max_points]
    mask = np.isfinite(time) & np.isfinite(flux)
    time, flux = time[mask], flux[mask]
    if len(time) < 100:
        return 0, None, None, None
    flux = (flux - np.median(flux)) / np.std(flux)
    ls = LombScargle(time, flux)
    frequency, power = ls.autopower(minimum_frequency=1/max_period, maximum_frequency=1/min_period)
    peaks, _ = find_peaks(power, prominence=power_prominence)
    return len(peaks), frequency, power, peaks


def calculate_fwhm(phase, flux):
    min_flux = np.min(flux)
    max_flux = np.median(flux)
    half_depth = (max_flux + min_flux) / 2
    indices = np.where(flux < half_depth)[0]
    return (phase[indices[-1]] - phase[indices[0]]) if len(indices) >= 2 else 0


def calculate_asymmetry_index(flux):
    mid = len(flux) // 2
    return np.mean(np.abs(flux[:mid] - flux[::-1][:mid]))


def calculate_curvature_at_minimum(phase, flux):
    smoothed_flux = gaussian_filter1d(flux, sigma=3)
    min_idx = np.argmin(smoothed_flux)
    if 2 < min_idx < len(flux) - 3:
        return smoothed_flux[min_idx - 1] - 2 * smoothed_flux[min_idx] + smoothed_flux[min_idx + 1]
    return 0


def calculate_autocorr_peak(flux):
    norm_flux = (flux - np.mean(flux)) / np.std(flux)
    ac = correlate(norm_flux, norm_flux, mode='full')[len(flux):]
    ac[0] = 0
    peaks, _ = find_peaks(ac)
    return ac[peaks[0]] if len(peaks) else 0


def extract_features_from_fits(fpath, min_period=0.2, max_period=20):
    try:
        table = Table.read(fpath, hdu=1)
        time = np.array(table["TIME"]).copy()
        flux = np.array(table["PDCSAP_FLUX"]).copy()

        if hasattr(time, "unit"): time = time.value
        if hasattr(flux, "unit"): flux = flux.value

        mask = np.isfinite(time) & np.isfinite(flux)
        time, flux = time[mask], flux[mask]
        flux = (flux - np.median(flux)) / np.std(flux)

        if len(time) < 10:
            raise ValueError("Too few valid points after cleaning.")

        # --- Lomb-Scargle period ---
        frequency, power = LombScargle(time, flux).autopower(
            minimum_frequency=1/max_period, maximum_frequency=1/min_period
        )
        best_period = 1 / frequency[np.argmax(power)]

        # --- Phase folding using CLARA utility ---
        result = c2pf.phase_fold(fpath, ls_power_thresh=0.05, bins=100, show=False)
        if result is None:
            raise ValueError("Phase folding failed.")
        phase, folded_flux, bin_centers, bin_means, flux_segment = result
        phase, flux = bin_centers, bin_means  # Use binned flux

        # --- Morphological features ---
        transit_depth = np.min(flux) - np.median(flux)
        transit_width = calculate_fwhm(phase, flux)
        baseline_std = np.std(flux[flux > np.percentile(flux, 80)])
        asymmetry = calculate_asymmetry_index(flux)
        sharpness = calculate_curvature_at_minimum(phase, flux)
        autocorr_strength = calculate_autocorr_peak(flux)

        # --- LS Transit count ---
        # Re-read full-resolution flux for LS-based transit count
        flux_full = np.array(table["PDCSAP_FLUX"]).copy()
        if hasattr(flux_full, "unit"): flux_full = flux_full.value
        flux_full = (flux_full - np.median(flux_full)) / np.std(flux_full)
        transit_count, _, _, _ = count_ls_peaks(time, flux_full)

        # --- BLS Transit Features ---
        bls_flux = table["PDCSAP_FLUX"]
        if hasattr(bls_flux, "unit"): bls_flux = bls_flux.value
        bls_flux = bls_flux[mask]
        bls_flux = (bls_flux - np.median(bls_flux)) / np.std(bls_flux)

        bls = BoxLeastSquares(time, bls_flux)
        bls_result = bls.autopower(0.1)
        best_idx = np.argmax(bls_result.power)
        bls_depth = bls_result.depth[best_idx]
        bls_duration = bls_result.duration[best_idx]
        
        # Compute SNR manually if not present
        if hasattr(bls_result, "snr"):
            bls_snr = bls_result.snr[best_idx]
        else:
            # Approximate SNR = depth / scatter (simple fallback)
            scatter = np.std(bls_flux)
            bls_snr = bls_depth / scatter if scatter > 0 else 0


        return [
            transit_depth,         # 0
            transit_width,         # 1
            baseline_std,          # 2
            asymmetry,             # 3
            sharpness,             # 4
            autocorr_strength,     # 5
            transit_count,         # 6
            bls_depth,             # 7
            bls_snr,               # 8
            bls_duration           # 9
        ]

    except Exception as e:
        # print(f"⚠️ Failed to process {fpath}: {e}")
        return None
