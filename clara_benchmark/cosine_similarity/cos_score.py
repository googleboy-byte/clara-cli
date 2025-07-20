import pandas as pd
import pickle
from astropy.table import Table
from lightkurve import LightCurve
from sklearn.metrics.pairwise import cosine_similarity
from clara_benchmark.utils.logging import log_main
import numpy as np
from astropy.timeseries import LombScargle
import os
from clara_benchmark.utils.logging import log_system_status, log
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial

# Define label groups as a constant (not global variable)
LABEL_GROUPS = {
    "planet_like": ["Pl", "Pl?", "BD*", "s*b"],
    "binary_star": ["SB*", "**", "EB*", "LM*"],
    "stellar": ["*", "PM*", "Er*"],
    "outlier": ["outlier", "err"]
}

def create_label_to_group_mapping():
    """Create label to group mapping from LABEL_GROUPS"""
    return {
        label: group
        for group, labels in LABEL_GROUPS.items()
        for label in labels
}

def run_cosine_matching_batch(filenames, fits_dir, simbad_labelled_pca_model_path, simbad_labelled_pca_features_path, compute_func, min_similarity_threshold=0.3, workers=None):
    """
    Multiprocess cosine similarity match on light curves.
    """
    
    # Create label mapping
    label_to_group = create_label_to_group_mapping()

    print(f"Running cosine similarity matching on {len(filenames)} files...")

    # Create a partial function with all necessary parameters that can be pickled
    process_func = partial(
        process_single_file,
        fits_dir=fits_dir,
        compute_func=compute_func,
        min_similarity_threshold=min_similarity_threshold,
        simbad_labelled_pca_model_path=simbad_labelled_pca_model_path,
        simbad_labelled_pca_features_path=simbad_labelled_pca_features_path,
        label_to_group=label_to_group
    )

    with Pool(processes=workers or cpu_count()) as pool:
        results = list(tqdm(pool.imap(process_func, filenames), total=len(filenames)))

    results = [r for r in results if r is not None]
    return pd.DataFrame(results)

def process_single_file(fname, fits_dir, compute_func, min_similarity_threshold, simbad_labelled_pca_model_path, simbad_labelled_pca_features_path, label_to_group):
    """Process a single file with all parameters passed explicitly"""

    try:
        fits_path = os.path.join(fits_dir, fname)
        result = match_fits_by_cosine_similarity(
            fits_path=fits_path,
            num_flux=3000,
            num_freq=1000,
            min_similarity_threshold=min_similarity_threshold,
            simbad_labelled_pca_model_path=simbad_labelled_pca_model_path,
            simbad_labelled_pca_features_path=simbad_labelled_pca_features_path
        )

        sim_label = result["label"]
        group_label = label_to_group.get(sim_label, "unclassified")

        return {
            "filename": fname,
            "matched_label": sim_label,
            "label_group": group_label,
            "similarity_score": result["similarity"]
        }

    except Exception as e:
        print(f"[ERROR] {fname}: {e}")
        return None

def computeLombScargleTessLC(this_lc, 
                             num_flux_points=3000,
                             num_freq_points=1000):

    min_period = 4 / 24  # 4 hours in days
    max_period = 27      # 27 days
    
    frequencies = np.logspace(np.log10(1 / max_period), np.log10(1 / min_period), num_freq_points)

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


def match_fits_by_cosine_similarity(
    fits_path,
    num_flux=3000,
    num_freq=1000,
    min_similarity_threshold=0.6,
    simbad_labelled_pca_model_path=None,
    simbad_labelled_pca_features_path=None
):
    """Match FITS file by cosine similarity with explicit parameters"""

    # Validate required parameters
    if simbad_labelled_pca_model_path is None or simbad_labelled_pca_features_path is None:
        raise ValueError("simbad_labelled_pca_model_path and simbad_labelled_pca_features_path must be provided")

    # make sure all directories exist
    os.makedirs(os.path.dirname(simbad_labelled_pca_model_path), exist_ok=True)
    os.makedirs(os.path.dirname(simbad_labelled_pca_features_path), exist_ok=True)
    
    # Load PCA and labeled feature set
    with open(simbad_labelled_pca_model_path, "rb") as f:
        pca = pickle.load(f)

    df = pd.read_csv(simbad_labelled_pca_features_path)

    # Extract features from input file
    table = Table.read(fits_path, hdu=1)
    lc = LightCurve(time=table["TIME"], flux=table["PDCSAP_FLUX"])
    feature_vec, _, _ = computeLombScargleTessLC(lc, num_flux, num_freq)

    # PCA projection
    reduced_vec = pca.transform([feature_vec])

    # Cosine similarity with labeled dataset
    similarities = cosine_similarity(reduced_vec, df.iloc[:, :-2])[0]  # exclude last 2 columns (label, filename)
    df["similarity"] = similarities

    # Get top match
    top_match = df.sort_values(by="similarity", ascending=False).iloc[0]

    if top_match["similarity"] < min_similarity_threshold:
        log(f"Threshold not met. Similarity: {top_match['similarity']} < {min_similarity_threshold}. Matched label: {top_match['label']}", log_file="./logs/cosine_sim.log", call_log_main=False)
        return {
            "label": "outlier",
            "similarity": top_match["similarity"],
            "filename": top_match["filename"]
        }
    else:
        log(f"Threshold met. Similarity: {top_match['similarity']} >= {min_similarity_threshold}. Matched label: {top_match['label']}", log_file="./logs/cosine_sim.log", call_log_main=False)
        return top_match[["label", "similarity", "filename"]]