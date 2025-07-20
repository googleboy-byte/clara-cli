import numpy as np
import batman
import matplotlib.pyplot as plt
import os
from lightkurve import LightCurve

def load_synthetic_lcs_from_npy(folder_path, filename_prefix="transit", sector_tag="SIM"):
    """
    Load synthetic light curves (saved as .npy) and return them as LightCurve objects.

    Assumes:
    - Each LC file is named like: {prefix}_{sector_tag}_{index}_{model}.npy
    - Time vector is saved as:   {prefix}_{sector_tag}_time.npy

    Args:
        folder_path (str): Directory where synthetic files are stored.
        filename_prefix (str): Prefix used in filenames (e.g., "transit").
        sector_tag (str): Sector or synthetic tag used (e.g., "SIM").

    Returns:
        list of LightCurve: List of LightCurve objects with flux and shared time.
        list of str: Corresponding filenames (useful for tracking model type).
    """
    time_file = os.path.join(folder_path, f"{filename_prefix}_{sector_tag}_time.npy")
    if not os.path.exists(time_file):
        raise FileNotFoundError(f"Time vector not found: {time_file}")

    time = np.load(time_file)
    lc_files = sorted([
        f for f in os.listdir(folder_path)
        if f.endswith(".npy") and "time" not in f
    ])

    lightcurves = []
    filenames = []

    for fname in lc_files:
        flux = np.load(os.path.join(folder_path, fname))
        lc = LightCurve(time=time, flux=flux)
        lightcurves.append(lc)
        filenames.append(fname)

    # print(f"✅ Loaded {len(lightcurves)} synthetic LightCurve objects from {folder_path}")
    return lightcurves, filenames

def generate_mixed_transit_lcs(n_curves=200,
                               lc_length_days=27.0,
                               cadence_minutes=2.0,
                               noise_level=100e-6,
                               random_state=42):
    """
    Generate synthetic light curves using a random mix of box-shaped and Mandel–Agol transits.

    Args:
        n_curves (int): Total number of synthetic LCs to generate.
        lc_length_days (float): Duration of light curve in days.
        cadence_minutes (float): Sampling cadence in minutes.
        noise_level (float): Standard deviation of white noise (normalized units).
        random_state (int): Seed for reproducibility.

    Returns:
        list of np.ndarray: List of synthetic flux arrays.
        np.ndarray: Shared time vector.
        list of str: Model type used for each curve ("box" or "mandel-agol")
    """
    np.random.seed(random_state)

    # Time vector
    cadence_days = cadence_minutes / (60 * 24)
    time = np.arange(0, lc_length_days, cadence_days)

    synthetic_lcs = []
    model_types = []

    for _ in range(n_curves):
        # Randomly choose transit model type
        model_type = np.random.choice(["box", "mandel-agol"])
        model_types.append(model_type)

        # Sample physical parameters
        period = np.random.uniform(1, 15)        # days
        duration = np.random.uniform(1, 10) / 24 # hours → days
        depth = np.random.uniform(0.0005, 0.02)  # 500 ppm to 2%
        t0 = np.random.uniform(0, period)        # transit center

        if model_type == "box":
            flux = np.ones_like(time)
            in_transit = np.abs((time - t0 + 0.5*period) % period - 0.5*period) < duration/2
            flux[in_transit] -= depth

        elif model_type == "mandel-agol":
            params = batman.TransitParams()
            params.t0 = t0
            params.per = period
            params.rp = np.sqrt(depth)            # radius ratio
            params.a = 15                         # semi-major axis (stellar radii)
            params.inc = 89                       # inclination in degrees
            params.ecc = 0
            params.w = 90
            params.u = [0.1, 0.3]                 # limb darkening
            params.limb_dark = "quadratic"

            m = batman.TransitModel(params, time)
            flux = m.light_curve(params)

        # Add white noise
        flux += np.random.normal(0, noise_level, size=flux.shape)
        synthetic_lcs.append(flux)

    return synthetic_lcs, time, model_types

def save_synthetic_lcs_npy(synthetic_lcs, time_vector, model_types,
                           output_folder="../data/synthetic_lcs_box_mandel-agol/",
                           filename_prefix="synthetic",
                           sector_tag="SYN",
                           save_times=True):
    """
    Save synthetic light curves (flux arrays) with time vectors and model types to .npy files.

    Each file is saved as:
        {prefix}_{sector_tag}_{index}_{model_type}.npy

    Args:
        synthetic_lcs (list of np.ndarray): List of flux arrays.
        time_vector (np.ndarray): Shared time vector.
        model_types (list of str): Model type ("box" or "mandel-agol") for each LC.
        output_folder (str): Folder to save .npy files.
        filename_prefix (str): Filename prefix.
        sector_tag (str): Optional identifier in filenames (e.g. sector or "SYN").
        save_times (bool): If True, save a corresponding _time.npy file once.
    """
    os.makedirs(output_folder, exist_ok=True)

    if save_times:
        time_path = os.path.join(output_folder, f"{filename_prefix}_{sector_tag}_time.npy")
        np.save(time_path, time_vector)
        print(f"🕒 Saved shared time vector: {time_path}")

    for i, (flux, model) in enumerate(zip(synthetic_lcs, model_types)):
        fname = f"{filename_prefix}_{sector_tag}_{i:04d}_{model}.npy"
        fpath = os.path.join(output_folder, fname)
        np.save(fpath, flux)

    print(f"✅ Saved {len(synthetic_lcs)} synthetic light curves to: {output_folder}")
