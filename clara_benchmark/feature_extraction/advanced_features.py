import clara_benchmark.clara.clara2_features as clara2_features
import numpy as np
import os
import pandas as pd
from datetime import datetime
from clara_benchmark.utils.logging import *
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm
import warnings

# Suppress numpy warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")

def process_single_advanced_features(fits_filename, fits_dir):
    """
    Process a single FITS file to extract advanced features.
    
    Args:
        fits_filename: Name of the FITS file
        fits_dir: Directory containing the FITS file
    
    Returns:
        tuple: (features, filename) or (None, filename) if failed
    """
    try:
        fits_path = os.path.join(fits_dir, fits_filename)
        features = clara2_features.extract_features_from_fits(fits_path)
        if features is not None:
            # Check for NaN values
            if np.any(np.isnan(features)) or np.any(np.isinf(features)):
                return None, fits_filename
            return features, fits_filename
        else:
            return None, fits_filename
    except Exception as e:
        return None, fits_filename

def features_advanced_10_feature_set(df, fits_dir, filename_column_in_df, output_dir, max_workers=4, desired_length=None, reduce_features=False, reduce_features_n_features=50):
    """
    Extract advanced 10-feature set using multiprocessing.
    
    Args:
        df: DataFrame containing filenames
        fits_dir: Directory containing FITS files
        filename_column_in_df: Column name containing filenames
        output_dir: Output directory for results
        max_workers: Number of worker processes
        desired_length: Desired length for feature vectors (optional, not used for advanced features)
        reduce_features: Whether to apply PCA reduction (ignored for advanced features)
        reduce_features_n_features: Number of PCA components to keep (ignored for advanced features)
    
    Returns:
        save_path: Path to saved features .npy file
    """
    # Create subdirectory for advanced features
    advanced_output_dir = os.path.join(output_dir, "advanced_features")
    os.makedirs(advanced_output_dir, exist_ok=True)
    
    # Initialize feature extraction log
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    feature_log_file = f"./logs/advanced_features.log"
    
    log(f"Starting advanced 10-feature set extraction", feature_log_file, call_log_main=True)
    log(f"Input DataFrame shape: {df.shape}", feature_log_file, call_log_main=False)
    log(f"FITS directory: {fits_dir}", feature_log_file, call_log_main=False)
    log(f"Output directory: {advanced_output_dir}", feature_log_file, call_log_main=False)
    log(f"Max workers: {max_workers}", feature_log_file, call_log_main=False)
    log(f"Note: PCA reduction is not available for advanced features", feature_log_file, call_log_main=False)
    
    fitslist = df[filename_column_in_df].tolist()
    log(f"Processing {len(fitslist)} files with multiprocessing", feature_log_file, call_log_main=True)
    
    # Create partial function for multiprocessing
    process_func = partial(process_single_advanced_features, fits_dir=fits_dir)
    
    # Process files with multiprocessing
    log(f"Starting parallel feature extraction with {max_workers} workers", feature_log_file, call_log_main=True)
    
    with Pool(processes=max_workers) as pool:
        results = list(tqdm(
            pool.imap(process_func, fitslist), 
            total=len(fitslist),
            desc="Extracting advanced features"
        ))
    
    # Separate successful and failed results
    successful_features = []
    successful_filenames = []
    failed_filenames = []
    nan_count = 0
    
    for features, filename in results:
        if features is not None:
            # Additional NaN check
            if np.any(np.isnan(features)) or np.any(np.isinf(features)):
                failed_filenames.append(filename)
                nan_count += 1
                continue
            successful_features.append(features)
            successful_filenames.append(filename)
        else:
            failed_filenames.append(filename)
    
    log(f"Advanced feature extraction complete: {len(successful_features)} successful, {len(failed_filenames)} failed ({nan_count} due to NaN/Inf values)", 
        feature_log_file, call_log_main=True)
    
    if not successful_features:
        log("No valid features extracted", feature_log_file, call_log_main=True, error=True)
        return None
    
    # Convert to numpy array for easier manipulation
    feature_array = np.array(successful_features)
    log(f"Feature array shape: {feature_array.shape}", feature_log_file, call_log_main=False)
    
    # Save features as .npy and filenames as .txt
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    features_save_path = os.path.join(advanced_output_dir, f"advanced_features_{timestamp}.npy")
    filenames_save_path = os.path.join(advanced_output_dir, f"advanced_filenames_{timestamp}.txt")
    
    # Save features as numpy array
    np.save(features_save_path, feature_array)
    log(f"Saved advanced features to: {features_save_path}", feature_log_file, call_log_main=True)
    
    # Save filenames as text file
    with open(filenames_save_path, "w") as f:
        for filename in successful_filenames:
            f.write(filename + "\n")
    log(f"Saved advanced filenames to: {filenames_save_path}", feature_log_file, call_log_main=True)
    
    log(f"Final feature matrix shape: {feature_array.shape}", feature_log_file, call_log_main=False)
    log(f"Advanced feature extraction pipeline complete", feature_log_file, call_log_main=True)
    
    # Log feature statistics
    log(f"Feature statistics:", feature_log_file, call_log_main=False)
    feature_names = ['transit_depth', 'transit_width', 'baseline_std', 'asymmetry', 'sharpness', 
                    'autocorr_strength', 'transit_count', 'bls_depth', 'bls_snr', 'bls_duration']
    for i, col in enumerate(feature_names):
        mean_val = np.mean(feature_array[:, i])
        std_val = np.std(feature_array[:, i])
        log(f"  {col}: mean={mean_val:.4f}, std={std_val:.4f}", feature_log_file, call_log_main=False)
    
    return features_save_path