from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import os
from astropy.table import Table
from lightkurve import LightCurve
from astropy.timeseries import LombScargle
import numpy as np

# --- Define frequency range: 4 hours to 27 days ---
num_flux_points = 3000 # MG21 3000 flux light curve points
num_freq_points = 1000 # MG21 1000 points corresponding to the periodogram
min_period = 4 / 24  # 4 hours in days
max_period = 27      # 27 days
frequencies = np.logspace(np.log10(1 / max_period), np.log10(1 / min_period), num_freq_points)

def computeLombScargleTessLC(this_lc, 
                             num_flux_points=num_flux_points,
                             num_freq_points=num_freq_points):

    this_lc.flux = this_lc.flux.value
    this_lc.flux_err = this_lc.flux_err.value
    
    # --- Preprocess --- mask removes any NaNs, Infs and any missing or corrupted points
    time = this_lc.time.value
    flux = this_lc.flux
    mask = np.isfinite(time) & np.isfinite(flux)
    time, flux = time[mask], flux[mask]
    flux = (flux - np.mean(flux)) / np.std(flux)
    
    # --- Truncate or pad flux vector ---
    flux_vector = flux[:num_flux_points]
    if len(flux_vector) < num_flux_points:
        flux_vector = np.pad(flux_vector, (0, num_flux_points - len(flux_vector)))

    ls = LombScargle(time, flux)
    power = ls.power(frequencies)
    power_vector = power[:num_freq_points]

    feature_vector = np.concatenate([flux_vector, power_vector])
    # print(f"Feature vector shape: {feature_vector.shape}")

    return feature_vector, flux_vector, power_vector

def process_single_fits(full_path):
    try:
        table = Table.read(full_path, hdu=1)
        this_fits_lc = LightCurve(time=table["TIME"], flux=table["PDCSAP_FLUX"])
        feature_vector, _, _ = computeLombScargleTessLC(this_fits_lc)
        if feature_vector.shape[0] == 4000:
            return feature_vector, os.path.basename(full_path)
    except Exception:
        return None

def build_feature_matrix_from_folder_tess_parallel(folder_path,
                                                   limit=None,
                                                   max_workers=8,
                                                   fits_files_list=None,
                                                   save_features_path=None,
                                                   save_feature_name=None,
                                                   toifits=None
                                                  ):
    
    fits_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".fits")])
    if fits_files_list is not None:
        fits_files = fits_files_list
    if limit:
        fits_files = fits_files[:limit]
    if toifits is not None:
        fits_files = [x for x in fits_files if x not in toifits]

    full_paths = [os.path.join(folder_path, f) for f in fits_files]
    feature_list = []
    filenames = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(process_single_fits, full_paths),
                            total=len(full_paths),
                            desc="Parallel feature extraction"))

    for res in results:
        if res is not None:
            fvec, fname = res
            feature_list.append(fvec)
            filenames.append(fname)

    if save_features_path is not None and save_feature_name is not None:
        os.makedirs(save_features_path, exist_ok=True) 
        filenames_savefname = os.path.join(save_features_path, save_feature_name) +"filenames.txt"
        features_savefname = os.path.join(save_features_path, save_feature_name) + ".npy" 
        np.save(features_savefname, feature_list)
        print(f"Features written to {features_savefname}")
        with open(filenames_savefname, "w") as f:
            for name in filenames:
                f.write(name + "\n")
        print(f"Filenames written to {filenames_savefname}")

    return feature_list, filenames
