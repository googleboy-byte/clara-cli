"""
Clustering utilities for CLARA astronomical feature analysis.
Provides multiple clustering algorithms with standardized interface.
"""

import numpy as np
import pandas as pd
import os
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns
from clara_benchmark.utils.logging import *

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False

logfile = "./logs/clustering.log"

def load_features_and_filenames(features_path, filenames_path):
    """
    Load features and filenames from .npy and .txt files.
    
    Args:
        features_path: Path to .npy features file
        filenames_path: Path to .txt filenames file
    
    Returns:
        tuple: (features_array, filenames_list)
    """
    try:
        # Load features
        features = np.load(features_path)
        
        # Load filenames
        with open(filenames_path, 'r') as f:
            filenames = [line.strip() for line in f.readlines()]
        
        # Validate consistency
        if len(features) != len(filenames):
            raise ValueError(f"Feature count ({len(features)}) doesn't match filename count ({len(filenames)})")
        
        return features, filenames
    
    except Exception as e:
        raise ValueError(f"Error loading features/filenames: {str(e)}")

def preprocess_features(features, method='standard'):
    """
    Preprocess features for clustering.
    
    Args:
        features: Feature array (n_samples, n_features)
        method: Preprocessing method ('standard', 'minmax', 'robust', 'none')
    
    Returns:
        tuple: (scaled_features, scaler_object)
    """
    if method == 'none':
        return features, None
    
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
    elif method == 'robust':
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unknown preprocessing method: {method}")
    
    scaled_features = scaler.fit_transform(features)
    return scaled_features, scaler

def kmeans_clustering(X, n_clusters=3, random_state=42, **kwargs):
    """
    Perform KMeans clustering.
    
    Args:
        X: Feature array (n_samples, n_features)
        n_clusters: Number of clusters
        random_state: Random seed
        **kwargs: Additional KMeans parameters
    
    Returns:
        dict: Clustering results
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, **kwargs)
    labels = kmeans.fit_predict(X)
    
    return {
        'algorithm': 'KMeans',
        'labels': labels,
        'n_clusters': len(set(labels)),
        'model': kmeans,
        'inertia': kmeans.inertia_,
        'silhouette': silhouette_score(X, labels) if len(set(labels)) > 1 else np.nan,
        'calinski_harabasz': calinski_harabasz_score(X, labels) if len(set(labels)) > 1 else np.nan,
        'davies_bouldin': davies_bouldin_score(X, labels) if len(set(labels)) > 1 else np.nan
    }

def dbscan_clustering(X, eps=0.5, min_samples=5, **kwargs):
    """
    Perform DBSCAN clustering.
    
    Args:
        X: Feature array (n_samples, n_features)
        eps: Epsilon parameter for DBSCAN
        min_samples: Minimum samples parameter for DBSCAN
        **kwargs: Additional DBSCAN parameters
    
    Returns:
        dict: Clustering results
    """
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, **kwargs)
    labels = dbscan.fit_predict(X)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = np.sum(labels == -1)
    
    return {
        'algorithm': 'DBSCAN',
        'labels': labels,
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'model': dbscan,
        'silhouette': silhouette_score(X, labels) if n_clusters > 1 else np.nan,
        'calinski_harabasz': calinski_harabasz_score(X, labels) if n_clusters > 1 else np.nan,
        'davies_bouldin': davies_bouldin_score(X, labels) if n_clusters > 1 else np.nan
    }

def hdbscan_clustering(X, min_cluster_size=15, min_samples=5, **kwargs):
    """
    Perform HDBSCAN clustering.
    
    Args:
        X: Feature array (n_samples, n_features)
        min_cluster_size: Minimum cluster size
        min_samples: Minimum samples parameter
        **kwargs: Additional HDBSCAN parameters
    
    Returns:
        dict: Clustering results
    """
    if not HDBSCAN_AVAILABLE:
        raise ImportError("HDBSCAN not available. Install with: pip install hdbscan")
    
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, **kwargs)
    labels = clusterer.fit_predict(X)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = np.sum(labels == -1)
    
    return {
        'algorithm': 'HDBSCAN',
        'labels': labels,
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'model': clusterer,
        'silhouette': silhouette_score(X, labels) if n_clusters > 1 else np.nan,
        'calinski_harabasz': calinski_harabasz_score(X, labels) if n_clusters > 1 else np.nan,
        'davies_bouldin': davies_bouldin_score(X, labels) if n_clusters > 1 else np.nan
    }

def gaussian_mixture_clustering(X, n_components=3, covariance_type='full', random_state=42, **kwargs):
    """
    Perform Gaussian Mixture Model clustering.
    
    Args:
        X: Feature array (n_samples, n_features)
        n_components: Number of mixture components
        covariance_type: Type of covariance ('full', 'tied', 'diag', 'spherical')
        random_state: Random seed
        **kwargs: Additional GMM parameters
    
    Returns:
        dict: Clustering results
    """
    gmm = GaussianMixture(n_components=n_components, covariance_type=covariance_type, 
                          random_state=random_state, **kwargs)
    labels = gmm.fit_predict(X)
    
    return {
        'algorithm': 'GaussianMixture',
        'labels': labels,
        'n_clusters': len(set(labels)),
        'model': gmm,
        'aic': gmm.aic(X),
        'bic': gmm.bic(X),
        'silhouette': silhouette_score(X, labels) if len(set(labels)) > 1 else np.nan,
        'calinski_harabasz': calinski_harabasz_score(X, labels) if len(set(labels)) > 1 else np.nan,
        'davies_bouldin': davies_bouldin_score(X, labels) if len(set(labels)) > 1 else np.nan
    }

def bayesian_gaussian_mixture_clustering(X, n_components=3, covariance_type='full', 
                                        weight_concentration_prior_type='dirichlet_process',
                                        weight_concentration_prior=0.1, random_state=42, **kwargs):
    """
    Perform Bayesian Gaussian Mixture Model clustering.
    
    Args:
        X: Feature array (n_samples, n_features)
        n_components: Number of mixture components
        covariance_type: Type of covariance ('full', 'tied', 'diag', 'spherical')
        weight_concentration_prior_type: Type of prior ('dirichlet_process', 'dirichlet_distribution')
        weight_concentration_prior: Concentration parameter
        random_state: Random seed
        **kwargs: Additional BGMM parameters
    
    Returns:
        dict: Clustering results
    """
    bgmm = BayesianGaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        weight_concentration_prior_type=weight_concentration_prior_type,
        weight_concentration_prior=weight_concentration_prior,
        random_state=random_state,
        **kwargs
    )
    labels = bgmm.fit_predict(X)
    
    return {
        'algorithm': 'BayesianGaussianMixture',
        'labels': labels,
        'n_clusters': len(set(labels)),
        'model': bgmm,
        'aic': bgmm.aic(X),
        'bic': bgmm.bic(X),
        'silhouette': silhouette_score(X, labels) if len(set(labels)) > 1 else np.nan,
        'calinski_harabasz': calinski_harabasz_score(X, labels) if len(set(labels)) > 1 else np.nan,
        'davies_bouldin': davies_bouldin_score(X, labels) if len(set(labels)) > 1 else np.nan
    }

def compare_clustering_algorithms(X, algorithms=['kmeans', 'dbscan', 'hdbscan', 'gmm', 'bgmm'], 
                                output_dir="./results/clustering/", **kwargs):
    """
    Compare multiple clustering algorithms.
    
    Args:
        X: Feature array (n_samples, n_features)
        algorithms: List of algorithms to compare
        output_dir: Output directory for results
        **kwargs: Algorithm-specific parameters
    
    Returns:
        dict: Comparison results
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    results = {}
    
    for algorithm in algorithms:
        try:
            if algorithm == 'kmeans':
                n_clusters = kwargs.get('n_clusters', 3)
                results[algorithm] = kmeans_clustering(X, n_clusters=n_clusters, **kwargs)
            elif algorithm == 'dbscan':
                eps = kwargs.get('eps', 0.5)
                min_samples = kwargs.get('min_samples', 5)
                results[algorithm] = dbscan_clustering(X, eps=eps, min_samples=min_samples, **kwargs)
            elif algorithm == 'hdbscan':
                min_cluster_size = kwargs.get('min_cluster_size', 15)
                min_samples = kwargs.get('min_samples', 5)
                results[algorithm] = hdbscan_clustering(X, min_cluster_size=min_cluster_size, 
                                                      min_samples=min_samples, **kwargs)
            elif algorithm == 'gmm':
                n_components = kwargs.get('n_components', 3)
                results[algorithm] = gaussian_mixture_clustering(X, n_components=n_components, **kwargs)
            elif algorithm == 'bgmm':
                n_components = kwargs.get('n_components', 3)
                results[algorithm] = bayesian_gaussian_mixture_clustering(X, n_components=n_components, **kwargs)
            else:
                log(f"Unknown algorithm: {algorithm}", logfile)
                continue
                
        except Exception as e:
            log(f"Error with {algorithm}: {str(e)}", logfile)
            continue
    
    # Create comparison plots
    if len(results) > 1:
        create_comparison_plots(results, output_dir, timestamp)
    
    # Save results
    save_clustering_results(results, output_dir, timestamp)
    
    return results

def create_comparison_plots(results, output_dir, timestamp):
    """
    Create comparison plots for clustering results.
    
    Args:
        results: Dictionary of clustering results
        output_dir: Output directory
        timestamp: Timestamp for file naming
    """
    try:
        # Metrics comparison
        metrics = ['silhouette', 'calinski_harabasz', 'davies_bouldin']
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for i, metric in enumerate(metrics):
            values = [results[alg][metric] for alg in results.keys() if metric in results[alg]]
            labels = [alg for alg in results.keys() if metric in results[alg]]
            
            if values:
                axes[i].bar(labels, values)
                axes[i].set_title(f'{metric.replace("_", " ").title()}')
                axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'clustering_comparison_{timestamp}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        log(f"Error creating comparison plots: {str(e)}", logfile)
def save_clustering_results(results, output_dir, timestamp):
    """
    Save clustering results to separate metadata and CSV files for each algorithm.
    
    Args:
        results: Dictionary of clustering results
        output_dir: Output directory
        timestamp: Timestamp for file naming
    """
    try:
        for alg, result in results.items():
            # Create summary stats for metadata file
            summary_stats = []
            summary_stats.append(f"Clustering Algorithm: {result['algorithm']}")
            summary_stats.append(f"Number of Clusters: {result['n_clusters']}")
            if 'n_noise' in result:
                summary_stats.append(f"Number of Noise Points: {result['n_noise']}")
            if 'silhouette' in result and not np.isnan(result['silhouette']):
                summary_stats.append(f"Silhouette Score: {result['silhouette']:.3f}")
            if 'calinski_harabasz' in result and not np.isnan(result['calinski_harabasz']):
                summary_stats.append(f"Calinski-Harabasz Score: {result['calinski_harabasz']:.3f}")
            if 'davies_bouldin' in result and not np.isnan(result['davies_bouldin']):
                summary_stats.append(f"Davies-Bouldin Score: {result['davies_bouldin']:.3f}")
            if 'inertia' in result:
                summary_stats.append(f"Inertia: {result['inertia']:.3f}")
            if 'aic' in result:
                summary_stats.append(f"AIC: {result['aic']:.3f}")
            if 'bic' in result:
                summary_stats.append(f"BIC: {result['bic']:.3f}")
            
            # Add cluster size statistics
            labels = result['labels']
            unique_labels = sorted(set(labels))
            summary_stats.append("Cluster Sizes:")
            for label in unique_labels:
                count = np.sum(labels == label)
                if label == -1:
                    summary_stats.append(f"  Noise (label -1): {count} curves")
                else:
                    summary_stats.append(f"  Cluster {label}: {count} curves")
            
            # Write metadata file
            metadata_file = os.path.join(output_dir, f'{alg}_metadata_{timestamp}.txt')
            with open(metadata_file, 'w') as f:
                for stat in summary_stats:
                    f.write(f"{stat}\n")
                    
            # Write pure CSV file
            csv_file = os.path.join(output_dir, f'{alg}_results_{timestamp}.csv')
            with open(csv_file, 'w') as f:
                # Write header
                f.write("filename,cluster_label\n")
                # Write data
                for filename, label in zip(result['filenames'], result['labels']):
                    f.write(f"{filename},{label}\n")
                    
            log(f"Saved {alg} results to:", logfile)
            log(f"  Metadata: {metadata_file}", logfile)
            log(f"  CSV Data: {csv_file}", logfile)
                    
    except Exception as e:
        log(f"Error saving clustering results: {str(e)}", logfile)

def save_filenames_with_labels(results, filenames, output_dir):
    """
    This function is now deprecated - results are saved in save_clustering_results.
    """
    pass

def cluster_features(features_path, filenames_path, output_dir="./results/clustering/", 
                   cluster_algorithm='kmeans',
                   preprocessing='standard', is_distance_matrix=False, **kwargs):
    """
    Main function to cluster features from saved files.
    
    Args:
        features_path: Path to .npy features file
        filenames_path: Path to .txt filenames file
        output_dir: Output directory for results
        cluster_algorithm: Clustering algorithm to use (kmeans, dbscan, hdbscan, gmm, bgmm)
        preprocessing: Preprocessing method ('standard', 'minmax', 'robust', 'none')
        is_distance_matrix: If True, features are a precomputed distance matrix (no scaling, only DBSCAN/HDBSCAN allowed)
        **kwargs: Algorithm-specific parameters
    
    Returns:
        dict: Clustering results
    """
    # Load features and filenames
    X, filenames = load_features_and_filenames(features_path, filenames_path)
    
    # If using a distance matrix, skip scaling and restrict algorithms
    if is_distance_matrix:
        if X.shape[0] != X.shape[1]:
            raise ValueError("Distance matrix must be square (n x n)")
        if cluster_algorithm not in ["dbscan", "hdbscan"]:
            raise ValueError("When using a distance matrix, only DBSCAN or HDBSCAN are supported.")
        X_input = X
    else:
        # Preprocess features
        X_input, scaler = preprocess_features(X, method=preprocessing)
    
    # Define allowed parameters for each algorithm
    allowed_params = {
        'kmeans': {'n_clusters', 'random_state', 'init', 'n_init', 'max_iter', 'tol', 'algorithm'},
        'dbscan': {'eps', 'min_samples', 'metric', 'algorithm', 'leaf_size', 'p', 'n_jobs'},
        'hdbscan': {'min_cluster_size', 'min_samples', 'metric', 'cluster_selection_method', 'allow_single_cluster', 'alpha', 'approx_min_span_tree', 'gen_min_span_tree', 'core_dist_n_jobs', 'cluster_selection_epsilon'},
        'gmm': {'n_components', 'covariance_type', 'tol', 'reg_covar', 'max_iter', 'n_init', 'init_params', 'weights_init', 'means_init', 'precisions_init', 'random_state', 'warm_start', 'verbose', 'verbose_interval'},
        'bgmm': {'n_components', 'covariance_type', 'tol', 'reg_covar', 'max_iter', 'n_init', 'init_params', 'weight_concentration_prior_type', 'weight_concentration_prior', 'mean_precision_prior', 'mean_prior', 'degrees_of_freedom_prior', 'covariance_prior', 'random_state', 'warm_start', 'verbose', 'verbose_interval'}
    }

    # Select and run the specified algorithm
    if cluster_algorithm == 'kmeans':
        n_clusters = kwargs.get('n_clusters', 3)
        algo_kwargs = {k: v for k, v in kwargs.items() if k in allowed_params['kmeans'] and k != 'n_clusters'}
        result = kmeans_clustering(X_input, n_clusters=n_clusters, **algo_kwargs)
    elif cluster_algorithm == 'dbscan':
        eps = kwargs.get('eps', 0.5)
        min_samples = kwargs.get('min_samples', 5)
        metric = 'precomputed' if is_distance_matrix else 'euclidean'
        algo_kwargs = {k: v for k, v in kwargs.items() if k in allowed_params['dbscan'] and k not in ['eps', 'min_samples', 'metric']}
        result = dbscan_clustering(X_input, eps=eps, min_samples=min_samples, metric=metric, **algo_kwargs)
    elif cluster_algorithm == 'hdbscan':
        min_cluster_size = kwargs.get('min_cluster_size', 15)
        min_samples = kwargs.get('min_samples', 5)
        metric = 'precomputed' if is_distance_matrix else 'euclidean'
        algo_kwargs = {k: v for k, v in kwargs.items() if k in allowed_params['hdbscan'] and k not in ['min_cluster_size', 'min_samples', 'metric']}
        result = hdbscan_clustering(X_input, min_cluster_size=min_cluster_size, min_samples=min_samples, metric=metric, **algo_kwargs)
    elif cluster_algorithm == 'gmm':
        n_components = kwargs.get('n_components', 3)
        covariance_type = kwargs.get('covariance_type', 'full')
        algo_kwargs = {k: v for k, v in kwargs.items() if k in allowed_params['gmm'] and k not in ['n_components', 'covariance_type']}
        result = gaussian_mixture_clustering(X_input, n_components=n_components, covariance_type=covariance_type, **algo_kwargs)
    elif cluster_algorithm == 'bgmm':
        n_components = kwargs.get('n_components', 3)
        covariance_type = kwargs.get('covariance_type', 'full')
        weight_concentration_prior_type = kwargs.get('weight_concentration_prior_type', 'dirichlet_process')
        weight_concentration_prior = kwargs.get('weight_concentration_prior', 0.1)
        algo_kwargs = {k: v for k, v in kwargs.items() if k in allowed_params['bgmm'] and k not in ['n_components', 'covariance_type', 'weight_concentration_prior_type', 'weight_concentration_prior']}
        result = bayesian_gaussian_mixture_clustering(
            X_input, n_components=n_components, covariance_type=covariance_type,
            weight_concentration_prior_type=weight_concentration_prior_type,
            weight_concentration_prior=weight_concentration_prior, **algo_kwargs)
    else:
        raise ValueError(f"Unknown clustering algorithm: {cluster_algorithm}")
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Add filenames to the result dictionary
    result['filenames'] = filenames
    
    save_clustering_results({cluster_algorithm: result}, output_dir, timestamp)
    
    return {cluster_algorithm: result} 