import pandas as pd
import requests
from typing import List, Dict, Union
from clara_feature_extraction_parallel import computeLombScargleTessLC
from lightkurve import LightCurve
from astropy.table import Table
import numpy as np
from tqdm import tqdm
from clara_utils import tic_from_fits_fname
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

def get_toi_importance_scores_from_filenames(
    toi_filenames: List[str],
    sector_tic_ids: List[int],
    verbose: bool = True
) -> pd.DataFrame:
    """
    Compute importance scores (raw and normalized) for TOIs given by FITS filenames.

    Parameters:
    -----------
    toi_filenames : List[str]
        List of FITS filenames of TOIs (e.g., 'tess2020234131234-s0002-123456789-0123-s_lc.fits')
    sector_tic_ids : List[int]
        All TIC IDs for TOIs in the current sector, used for normalization
    verbose : bool
        Whether to print progress and summary

    Returns:
    --------
    pd.DataFrame
        DataFrame with columns: ['filename', 'tic_id', 'importance_score', 'normalized_score']
    """
    import pandas as pd

    # Step 1: Parse TIC IDs and create filename→TIC map
    filename_tic_pairs = [(fname, int(fname.split("-")[2])) for fname in toi_filenames]
    filenames, tic_ids = zip(*filename_tic_pairs)
    filename_map = {tic: fname for fname, tic in filename_tic_pairs}

    # Step 2: Download and filter TOI catalog
    url = "https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv"
    if verbose:
        print("Downloading TOI catalog...")
    toi_df = pd.read_csv(url, comment="#")

    required_cols = ["TIC ID", "Planet SNR", "Depth (ppm)", "Duration (hours)"]
    for col in required_cols:
        if col not in toi_df.columns:
            raise KeyError(f"TOI catalog missing required column: {col}")

    # Step 3: Filter to sector TICs and target TICs
    sector_df = toi_df[toi_df["TIC ID"].isin(sector_tic_ids)].copy()
    target_df = sector_df[sector_df["TIC ID"].isin(tic_ids)].copy()

    # Step 4: Drop rows with missing data
    sector_df = sector_df.dropna(subset=["Planet SNR", "Depth (ppm)", "Duration (hours)"])
    target_df = target_df.dropna(subset=["Planet SNR", "Depth (ppm)", "Duration (hours)"])

    # Step 5: Compute importance scores
    sector_df["importance_score"] = (
        sector_df["Planet SNR"] *
        sector_df["Depth (ppm)"] *
        sector_df["Duration (hours)"]
    )
    target_df["importance_score"] = (
        target_df["Planet SNR"] *
        target_df["Depth (ppm)"] *
        target_df["Duration (hours)"]
    )

    # Step 6: Normalize scores
    min_score = sector_df["importance_score"].min()
    max_score = sector_df["importance_score"].max()
    target_df["normalized_score"] = (
        (target_df["importance_score"] - min_score) / (max_score - min_score)
    )

    # Step 7: Add filenames
    target_df["filename"] = target_df["TIC ID"].map(filename_map)

    if verbose:
        print(f"✓ Found {len(target_df)} TOIs with importance scores.")

    return target_df[["filename", "TIC ID", "importance_score", "normalized_score"]].rename(columns={"TIC ID": "tic_id"})
