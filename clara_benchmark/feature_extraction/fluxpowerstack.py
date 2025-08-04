import clara_benchmark.clara.astrodata_helpers.clara2_phase_fold as clara2_phase_fold
import numpy as np
import os
import pandas as pd
from datetime import datetime
from clara_benchmark.clara.clara_feature_extraction_parallel import *
from clara_benchmark.utils.logging import *
import warnings

# Suppress numpy warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")

def features_minbeans_folded(df, fits_dir, filename_column_in_df, output_dir, max_workers=4, desired_length=None):
    # Create subdirectory for bin_means features
    bin_means_output_dir = os.path.join(output_dir, "bin_means")
    os.makedirs(bin_means_output_dir, exist_ok=True)
    
    # Initialize feature extraction log
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    feature_log_file = f"./logs/feature_extraction.log"
    
    log(f"Starting fluxpowerstack folded feature extraction", feature_log_file, call_log_main=True)
    log(f"Input DataFrame shape: {df.shape}", feature_log_file, call_log_main=False)
    log(f"FITS directory: {fits_dir}", feature_log_file, call_log_main=False)
    log(f"Output directory: {bin_means_output_dir}", feature_log_file, call_log_main=False)
    log(f"Max workers: {max_workers}", feature_log_file, call_log_main=False)
    log(f"Desired length: {desired_length}", feature_log_file, call_log_main=False)
    
    filenames = df[filename_column_in_df].tolist()
    log(f"Processing {len(filenames)} files", feature_log_file, call_log_main=True)
    
    features = []
    valid_filenames = []
    failed_files = []
    
    for i, filename in enumerate(filenames):
        if i % 100 == 0:  # Log progress every 100 files
            log(f"Processed {i}/{len(filenames)} files ({len(valid_filenames)} successful, {len(failed_files)} failed)", 
                feature_log_file, call_log_main=True)
        
        fits_file = os.path.join(fits_dir, filename)
        try:
            # Use the correct phase_fold function
            phase_fold_results = clara2_phase_fold.phase_fold(fits_file)
            if phase_fold_results is None:
                failed_files.append(filename)
                continue
            phase, folded_flux, bin_centers, bin_means, flux_segment = phase_fold_results
            vec = np.array(bin_means)
            if desired_length is not None:
                if len(vec) >= desired_length:
                    vec = vec[:desired_length]
                else:
                    vec = np.pad(vec, (0, desired_length - len(vec)), mode='constant', constant_values=0)
            
            norm_vec = (vec - np.mean(vec)) / np.std(vec)
            features.append(norm_vec)
            valid_filenames.append(filename)
        except Exception as e:
            log(f"Failed to process {filename}: {str(e)}", feature_log_file, call_log_main=False)
            failed_files.append(filename)

    log(f"Feature extraction complete: {len(valid_filenames)} successful, {len(failed_files)} failed", 
        feature_log_file, call_log_main=True)
    
    if not features:
        log("No valid features extracted", feature_log_file, call_log_main=True, error=True)
        return None

    # Save features as .npy and filenames as .txt
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    features_save_path = os.path.join(bin_means_output_dir, f"bin_means_features_{timestamp}.npy")
    filenames_save_path = os.path.join(bin_means_output_dir, f"bin_means_filenames_{timestamp}.txt")
    
    # Save features as numpy array
    np.save(features_save_path, np.array(features))
    log(f"Saved bin_means features to: {features_save_path}", feature_log_file, call_log_main=True)
    
    # Save filenames as text file
    with open(filenames_save_path, "w") as f:
        for filename in valid_filenames:
            f.write(filename + "\n")
    log(f"Saved bin_means filenames to: {filenames_save_path}", feature_log_file, call_log_main=True)
    
    log(f"Feature matrix shape: {np.array(features).shape}", feature_log_file, call_log_main=False)
    return features_save_path

def features_fluxpowerstack_mp(df, fits_dir, filename_column_in_df, output_dir, max_workers=4, desired_length=None, reduce_features=False, reduce_features_n_features=50):
    """
    Extract flux power stack features using multiprocessing.
    
    Args:
        df: DataFrame containing filenames
        fits_dir: Directory containing FITS files
        filename_column_in_df: Column name containing filenames
        output_dir: Output directory for results
        max_workers: Number of worker processes
        desired_length: Desired length for feature vectors (optional)
        reduce_features: Whether to apply PCA reduction (only for fluxpowerstack)
        reduce_features_n_features: Number of PCA components to keep
    
    Returns:
        save_path: Path to saved features .npy file
    """
    # Create subdirectory for fluxpowerstack features
    fluxpowerstack_output_dir = os.path.join(output_dir, "fluxpowerstack")
    os.makedirs(fluxpowerstack_output_dir, exist_ok=True)
    
    # Initialize feature extraction log
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    feature_log_file = f"./logs/feature_extraction_mp.log"
    
    log(f"Starting fluxpowerstack multiprocessing feature extraction", feature_log_file, call_log_main=True)
    log(f"Input DataFrame shape: {df.shape}", feature_log_file, call_log_main=False)
    log(f"FITS directory: {fits_dir}", feature_log_file, call_log_main=False)
    log(f"Output directory: {fluxpowerstack_output_dir}", feature_log_file, call_log_main=False)
    log(f"Max workers: {max_workers}", feature_log_file, call_log_main=False)
    log(f"Desired length: {desired_length}", feature_log_file, call_log_main=False)
    log(f"Reduce features: {reduce_features}", feature_log_file, call_log_main=False)
    if reduce_features:
        log(f"PCA components: {reduce_features_n_features}", feature_log_file, call_log_main=False)
    
    fitslist = df[filename_column_in_df].tolist()
    log(f"Processing {len(fitslist)} files with multiprocessing", feature_log_file, call_log_main=True)
    
    # Use the multiprocessing function from clara_feature_extraction_parallel with save options
    log(f"Starting parallel feature extraction with {max_workers} workers", feature_log_file, call_log_main=True)
    
    # Generate save paths
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_feature_name = f"fluxpowerstack_features_{timestamp}"
    
    feature_list, filenames = build_feature_matrix_from_folder_tess_parallel(
        folder_path=fits_dir,
        max_workers=max_workers,
        fits_files_list=fitslist,
        save_features_path=fluxpowerstack_output_dir,
        save_feature_name=save_feature_name
    )
    
    log(f"Parallel feature extraction complete", feature_log_file, call_log_main=True)
    log(f"Extracted features for {len(feature_list)} files", feature_log_file, call_log_main=False)
    
    if not feature_list:
        log("No valid features extracted", feature_log_file, call_log_main=True, error=True)
        return None
    
    # Apply PCA reduction if requested (only for fluxpowerstack)
    if reduce_features:
        log(f"Applying PCA reduction to {reduce_features_n_features} components", feature_log_file, call_log_main=True)
        from sklearn.decomposition import PCA
        pca = PCA(n_components=reduce_features_n_features)
        feature_array = np.array(feature_list)
        feature_array = pca.fit_transform(feature_array)
        log(f"PCA reduction complete. Explained variance ratio: {sum(pca.explained_variance_ratio_):.3f}", 
            feature_log_file, call_log_main=False)
        
        # Save PCA-reduced features
        pca_save_path = os.path.join(fluxpowerstack_output_dir, f"{save_feature_name}_pca_{reduce_features_n_features}.npy")
        np.save(pca_save_path, feature_array)
        log(f"Saved PCA-reduced features to: {pca_save_path}", feature_log_file, call_log_main=True)
        return pca_save_path
    
    # Apply desired_length if specified
    if desired_length is not None:
        log(f"Applying desired_length adjustment to {desired_length}", feature_log_file, call_log_main=True)
        feature_array = np.array(feature_list)
        if feature_array.shape[1] >= desired_length:
            feature_array = feature_array[:, :desired_length]
        else:
            # Pad with zeros if desired_length is larger than feature dimension
            padded_array = np.zeros((feature_array.shape[0], desired_length))
            padded_array[:, :feature_array.shape[1]] = feature_array
            feature_array = padded_array
        
        # Save adjusted features
        adjusted_save_path = os.path.join(fluxpowerstack_output_dir, f"{save_feature_name}_adjusted_{desired_length}.npy")
        np.save(adjusted_save_path, feature_array)
        log(f"Saved adjusted features to: {adjusted_save_path}", feature_log_file, call_log_main=True)
        log(f"Feature length adjustment complete", feature_log_file, call_log_main=False)
        return adjusted_save_path
    
    # Return the path to the saved features
    features_save_path = os.path.join(fluxpowerstack_output_dir, f"{save_feature_name}.npy")
    log(f"Saved fluxpowerstack multiprocessing features to: {features_save_path}", feature_log_file, call_log_main=True)
    log(f"Final feature matrix shape: {np.array(feature_list).shape}", feature_log_file, call_log_main=False)
    log(f"Feature extraction pipeline complete", feature_log_file, call_log_main=True)
    
    return features_save_path