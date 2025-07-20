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
from IPython.display import clear_output
    
def check_tois(tic_ids: List[int], verbose: bool = True) -> pd.DataFrame:
    """
    Check if given TIC IDs are TESS Objects of Interest (TOIs).
    
    Parameters:
    -----------
    tic_ids : List[int]
        List of TIC IDs to check
    verbose : bool, default True
        If True, prints results for each TIC ID
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns: TIC_ID, Is_TOI, TOI_Numbers, TOI_Count
    
    Example:
    --------
    >>> tic_ids = [123456789, 987654321, 456789123]
    >>> results = check_tois(tic_ids)
    >>> print(results)
    """
    
    try:
        # Download the latest TOI catalog
        url = "https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv"
        if verbose:
            print("Downloading latest TOI catalog...")
        
        toi_df = pd.read_csv(url, comment='#')
        
        if verbose:
            print(f"TOI catalog loaded with {len(toi_df)} entries")
            print("-" * 50)
        
        # Check which TIC IDs are TOIs
        results = []
        
        for tic_id in tic_ids:
            toi_match = toi_df[toi_df['TIC ID'] == tic_id]
            
            if not toi_match.empty:
                toi_numbers = toi_match['TOI'].tolist()
                results.append({
                    'TIC_ID': tic_id, 
                    'Is_TOI': True, 
                    'TOI_Numbers': toi_numbers,
                    'TOI_Count': len(toi_numbers)
                })
                if verbose:
                    print(f"✓ TIC {tic_id} is TOI: {toi_numbers}")
            else:
                results.append({
                    'TIC_ID': tic_id, 
                    'Is_TOI': False, 
                    'TOI_Numbers': [],
                    'TOI_Count': 0
                })
                if verbose:
                    print(f"✗ TIC {tic_id} is NOT a TOI")
        
        results_df = pd.DataFrame(results)
        
        if verbose:
            print("-" * 50)
            toi_count = results_df['Is_TOI'].sum()
            print(f"Summary: {toi_count}/{len(tic_ids)} objects are TOIs")
        
        return results_df
        
    except Exception as e:
        print(f"Error downloading or processing TOI catalog: {e}")
        # Return empty DataFrame with correct structure
        return pd.DataFrame(columns=['TIC_ID', 'Is_TOI', 'TOI_Numbers', 'TOI_Count'])


def get_toi_details(tic_ids: List[int]) -> pd.DataFrame:
    """
    Get detailed TOI information for given TIC IDs.
    
    Parameters:
    -----------
    tic_ids : List[int]
        List of TIC IDs to get TOI details for
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with detailed TOI information for matching TIC IDs
    """
    
    try:
        # Download the latest TOI catalog
        url = "https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv"
        toi_df = pd.read_csv(url, comment='#')
        
        # Filter for the requested TIC IDs
        matching_tois = toi_df[toi_df['TIC ID'].isin(tic_ids)]
        
        return matching_tois
        
    except Exception as e:
        print(f"Error downloading or processing TOI catalog: {e}")
        return pd.DataFrame()


def get_sector_tois(sector: int, verbose: bool = True) -> pd.DataFrame:
    """
    Get all TOIs that were observed in a specific TESS sector.
    
    Parameters:
    -----------
    sector : int
        TESS sector number to search for TOIs
    verbose : bool, default True
        If True, prints summary information
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with TOI information for objects observed in the specified sector
    """
    
    try:
        # Download the latest TOI catalog
        url = "https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv"
        if verbose:
            print(f"Downloading TOI catalog to search for Sector {sector} TOIs...")
        
        toi_df = pd.read_csv(url, comment='#')
        
        # The TOI catalog has a 'Sectors' column that contains sector information
        # It might be formatted as comma-separated values like "1,2,3" or "1"
        sector_tois = toi_df[toi_df['Sectors'].astype(str).str.contains(f'\\b{sector}\\b', na=False)]
        
        if verbose:
            print(f"Found {len(sector_tois)} TOIs observed in Sector {sector}")
            
            if len(sector_tois) > 0:
                print(f"TIC IDs: {sorted(sector_tois['TIC ID'].unique())}")
                print(f"TOI numbers: {sorted(sector_tois['TOI'].unique())}")
        
        return sector_tois
        
    except Exception as e:
        print(f"Error downloading or processing TOI catalog: {e}")
        return pd.DataFrame()


def get_sector_tic_ids(sector: int) -> List[int]:
    """
    Get just the TIC IDs of TOIs observed in a specific TESS sector.
    
    Parameters:
    -----------
    sector : int
        TESS sector number to search for TOIs
    
    Returns:
    --------
    List[int]
        List of TIC IDs for TOIs observed in the specified sector
    """
    
    sector_tois = get_sector_tois(sector, verbose=False)
    
    if len(sector_tois) > 0:
        return sorted(sector_tois['TIC ID'].unique().tolist())
    else:
        return []

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
