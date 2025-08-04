# CLARA CLI - Astronomical Data Processing Pipeline

A comprehensive command-line interface for downloading, processing, and analyzing TESS (Transiting Exoplanet Survey Satellite) light curve data with anomaly detection and similarity matching capabilities.

## Overview

CLARA CLI is a multi-functional astronomical data processing pipeline designed for:
- **Data Download**: Automated TESS light curve downloads with multi-threaded processing
- **Anomaly Detection**: Machine learning-based anomaly scoring using multiple models (P3, P5, P9)
- **Similarity Matching**: Cosine similarity analysis for astronomical object classification
- **System Monitoring**: Real-time system statistics logging during processing

## Features

### Core Functionality
- **Multi-threaded TESS data downloads** with configurable sectors and batch sizes
- **Anomaly detection** using URF (Unsupervised Random Forest) models
- **Cosine similarity matching** for astronomical object classification
- **Real-time system monitoring** with CPU, RAM, disk, and network statistics
- **Comprehensive logging** with timestamped entries and error handling
- **Configurable processing pipelines** via JSON configuration files

### Technical Capabilities
- **Parallel processing** with multiprocessing support
- **Memory-efficient streaming** for large datasets
- **Automatic directory creation** and file management
- **Robust error handling** with graceful degradation
- **Cross-platform compatibility** (Linux, macOS, Windows)
- **Comprehensive help system** with modular documentation organization

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Sufficient disk space for TESS data (recommended: 100GB+)
- Internet connection for data downloads

### Setup
1. Clone the repository:
```bash
git clone <repository-url>
cd clara-cli
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create necessary directories:
```bash
mkdir -p logs data/fits_files data/catalogues output results
```

## Usage

### Command Structure
```bash
python -m clara_benchmark.cli <command> [options] [--meta_message "description"]
```

### Available Commands

#### 1. Download TESS Data
Download light curve data from TESS sectors:
```bash
python -m clara_benchmark.cli download --config ./configs/download_config.json
```

**Configuration Example:**
```json
{
    "1": {
        "sector_number": 1,
        "catalogues_folder": "./data/catalogues/tess_download_scripts/",
        "out_dir_tess": "./data/fits_files/",
        "start_index": 0,
        "max_downloads": 1500,
        "log_file": "./logs/download_tess_lc.log",
        "num_threads": 4
    }
}
```

#### 2. Anomaly Scoring
Score astronomical objects for anomalies using machine learning models:
```bash
python -m clara_benchmark.cli score --config ./configs/urf_anomaly_scoring_config.json
```

**Configuration Example:**
```json
{
    "input_dirs": ["./data/fits_files/1/"],
    "output_dir": "./results/anomaly_scores/",
    "max_workers": 4,
    "use_p3": true,
    "use_p5": true,
    "use_p9": true,
    "calculate_wrss_p3p9": true,
    "p3model_path": "./models/p3_model.pkl",
    "p5model_path": "./models/p5_model.pkl",
    "p9model_path": "./models/p9_model.pkl"
}
```

#### 3. Cosine Similarity Matching
Match astronomical objects by similarity to known classifications:
```bash
python -m clara_benchmark.cli sim --config ./configs/cosine_similarity_config.json
```

**Configuration Example:**
```json
{
    "fits_dir": "./data/fits_files/1/",
    "input_df_path": "./results/anomaly_scores/wrss_p3p9_20240101_120000.csv",
    "input_df_sql": "SELECT filename, score_weighted_root_sumnorm FROM input_df WHERE score_weighted_root_sumnorm >= 0.0001 ORDER BY score_weighted_root_sumnorm DESC",
    "output_dir": "./results/similarity_scores/",
    "min_similarity_threshold": 0.6,
    "save_label_groups": ["planet_like", "binary_star"],
    "simbad_labelled_pca_model_path": "./models/simbad_pca_model.pkl",
    "simbad_labelled_pca_features_path": "./data/simbad_features.csv",
    "max_workers": 4
}
```

#### 4. Feature Extraction
Extract comprehensive feature sets from TESS light curves for machine learning applications:
```bash
python -m clara_benchmark.cli features --config ./configs/feature_extraction_config.json
```

**Configuration Example:**
```json
{
    "fits_dir": "./data/fits_files/1/",
    "input_df_path": "./results/anomaly_scores/wrss_p3p9_20250719_123442.csv",
    "input_df_sql": "SELECT * FROM input_df WHERE score_weighted_root_sumnorm >= 0.0001",
    "output_dir": "./results/features/",
    "max_workers": 4,
    "features": "fluxpowerstack",
    "reduce_features": "true",
    "reduce_features_n_features": 50,
    "filename_column_in_df": "filename"
}
```

**Available Feature Extraction Methods:**

1. **FLUXPOWERSTACK** (Recommended for ML):
   - Extracts 4000-dimensional feature vectors
   - Combines flux time series (3000 points) + Lomb-Scargle power spectrum (1000 points)
   - Optimized for machine learning applications
   - Supports PCA reduction for dimensionality reduction

2. **BIN_MEANS** (Phase-folded features):
   - Extracts phase-folded binned flux features
   - Useful for periodic pattern analysis
   - Configurable bin count and desired length
   - Normalized feature vectors

3. **ADVANCED_10_FEATURE_SET** (Astronomical features):
   - Extracts 10 sophisticated astronomical features:
     * Transit depth and width
     * Baseline standard deviation
     * Asymmetry and sharpness indices
     * Autocorrelation strength
     * Transit count and BLS parameters
   - Optimized for astronomical classification

**Feature Extraction Examples:**
```bash
# Basic fluxpowerstack extraction
python -m clara_benchmark.cli features --config ./configs/feature_extraction_config.json

# Advanced features with custom SQL filtering
python -m clara_benchmark.cli features --config ./configs/feature_extraction_config.json --features "advanced_10_feature_set" --input_df_sql "SELECT filename FROM input_df WHERE score_weighted_root_sumnorm >= 0.001 LIMIT 100"

# Fluxpowerstack with PCA reduction
python -m clara_benchmark.cli features --config ./configs/feature_extraction_config.json --features "fluxpowerstack" --reduce_features true --reduce_features_n_features 25

# Bin means with custom length
python -m clara_benchmark.cli features --config ./configs/feature_extraction_config.json --features "bin_means" --desired_length 100 --max_workers 2
```

#### 5. Phase Folding
Perform phase folding analysis on TESS light curves:
```bash
python -m clara_benchmark.cli fold --config ./configs/phase_folding_config.json
```

### Global Options
- `--meta_message`: Add a descriptive message to the run logs
- `--config`: Specify custom configuration file path

### Command-Line Overrides
All configuration parameters can be overridden via command-line arguments. The system automatically generates `--parameter_name` arguments for every key in the config file.

**Examples:**
```bash
# Override specific parameters for download
python -m clara_benchmark.cli download --num_threads 8 --max_downloads 2000

# Override parameters for scoring
python -m clara_benchmark.cli score --max_workers 6 --use_p3 true

# Override parameters for similarity matching
python -m clara_benchmark.cli sim --input_df_sql "SELECT * FROM input_df WHERE score_weighted_root_sumnorm >= 0.001" --min_similarity_threshold 0.7
```

**Automatic Type Detection:**
- Boolean values: `--use_p3 true`
- Integer values: `--max_workers 4`
- Float values: `--min_similarity_threshold 0.6`
- String values: `--output_dir "./custom_output/"`

## Configuration

### Download Configuration
| Parameter | Description | Default |
|-----------|-------------|---------|
| `sector_number` | TESS sector to download | Required |
| `catalogues_folder` | Path to catalog scripts | `./data/catalogues/` |
| `out_dir_tess` | Output directory for FITS files | `./data/fits_files/` |
| `start_index` | Starting index for downloads | 0 |
| `max_downloads` | Maximum number of files to download | Required |
| `log_file` | Log file path | `./logs/download_tess_lc.log` |
| `num_threads` | Number of download threads | 4 |

### Anomaly Scoring Configuration
| Parameter | Description | Default |
|-----------|-------------|---------|
| `input_dirs` | List of input directories | Required |
| `output_dir` | Output directory for results | Required |
| `max_workers` | Number of worker processes | 4 |
| `use_p3` | Enable P3 model scoring | false |
| `use_p5` | Enable P5 model scoring | false |
| `use_p9` | Enable P9 model scoring | false |
| `calculate_wrss_p3p9` | Calculate WRSS between P3/P9 | false |

### Similarity Matching Configuration
| Parameter | Description | Default |
|-----------|-------------|---------|
| `fits_dir` | Directory containing FITS files | Required |
| `input_df_path` | Path to input CSV file | Required |
| `input_df_sql` | Full SQL query to select and filter input data | Required |
| `output_dir` | Output directory for results | Required |
| `min_similarity_threshold` | Minimum similarity score | 0.6 |
| `save_label_groups` | Label groups to save (array, "all", or "none") | Required |
| `simbad_labelled_pca_model_path` | Path to PCA model file | Required |
| `simbad_labelled_pca_features_path` | Path to labeled features CSV | Required |
| `max_workers` | Number of worker processes | 4 |

### Feature Extraction Configuration
| Parameter | Description | Default |
|-----------|-------------|---------|
| `fits_dir` | Directory containing FITS files | Required |
| `input_df_path` | Path to input CSV file | Required |
| `input_df_sql` | Full SQL query to select and filter input data | Required |
| `output_dir` | Output directory for results | Required |
| `max_workers` | Number of worker processes | 4 |
| `features` | Feature extraction method ("fluxpowerstack", "bin_means", "advanced_10_feature_set") | Required |
| `reduce_features` | Enable PCA reduction (only for fluxpowerstack) | "false" |
| `reduce_features_n_features` | Number of PCA components to keep | 50 |
| `desired_length` | Desired feature vector length (for bin_means) | null |
| `filename_column_in_df` | Column name containing filenames | Required |

**Feature Extraction Methods:**

**FLUXPOWERSTACK:**
- **Output**: 4000-dimensional feature vectors (.npy format)
- **Features**: Flux time series (3000 points) + Lomb-Scargle power spectrum (1000 points)
- **PCA Support**: Yes (recommended for large datasets)
- **Use Case**: Machine learning applications, anomaly detection

**BIN_MEANS:**
- **Output**: Phase-folded binned flux features (.npy format)
- **Features**: Normalized binned flux values from phase folding
- **PCA Support**: No
- **Use Case**: Periodic pattern analysis, transit detection

**ADVANCED_10_FEATURE_SET:**
- **Output**: 10 astronomical features (.npy format)
- **Features**: Transit depth, width, baseline std, asymmetry, sharpness, autocorr strength, transit count, BLS parameters
- **PCA Support**: No
- **Use Case**: Astronomical classification, object characterization

**Output File Structure:**
```
./results/features/
├── fluxpowerstack/
│   ├── fluxpowerstack_features_YYYYMMDD_HHMMSS.npy
│   ├── fluxpowerstack_features_YYYYMMDD_HHMMSS.txt
│   └── fluxpowerstack_features_YYYYMMDD_HHMMSS_pca_25.npy (if PCA enabled)
├── bin_means/
│   ├── bin_means_features_YYYYMMDD_HHMMSS.npy
│   └── bin_means_filenames_YYYYMMDD_HHMMSS.txt
└── advanced_features/
    ├── advanced_features_YYYYMMDD_HHMMSS.npy
    └── advanced_filenames_YYYYMMDD_HHMMSS.txt
```

**SQL Query Capabilities:**
The `input_df_sql` parameter supports full SQL queries including:
- **SELECT**: Choose specific columns or use `SELECT *`
- **WHERE**: Filter rows based on conditions
- **ORDER BY**: Sort results by columns
- **LIMIT**: Limit number of results
- **Aggregate functions**: COUNT, SUM, AVG, etc.
- **Subqueries**: Nested SELECT statements

**Label Group Filtering Options:**
The `save_label_groups` parameter supports three filtering modes:
- **Array of labels**: `["planet_like", "binary_star"]` - Save only objects matching specific labels
- **"all"**: Save all matched objects regardless of their label group
- **"none"**: Save only unmatched objects (where label_group is null/NaN)

**Example SQL Queries:**
```sql
-- Select all columns with filtering
SELECT * FROM input_df WHERE score_weighted_root_sumnorm >= 0.0001

-- Select specific columns with ordering
SELECT filename, score_weighted_root_sumnorm FROM input_df 
WHERE score_weighted_root_sumnorm >= 0.0001 
ORDER BY score_weighted_root_sumnorm DESC

-- Limit results
SELECT filename FROM input_df 
WHERE score_weighted_root_sumnorm >= 0.0001 
LIMIT 1000

-- Complex filtering
SELECT filename, score_weighted_root_sumnorm FROM input_df 
WHERE score_weighted_root_sumnorm >= 0.0001 
AND filename LIKE '%sector1%'
ORDER BY score_weighted_root_sumnorm DESC
```

## System Monitoring

The CLI automatically monitors system resources during execution:
- **CPU Usage**: Percentage, core count, frequency
- **Memory Usage**: Total, used, available RAM
- **Disk Usage**: Space utilization and free space
- **Network I/O**: Bytes sent and received
- **System Load**: 1, 5, and 15-minute averages
- **Temperature**: CPU temperature (if available)

System statistics are logged every 5 seconds to `./logs/system_status.log`.

## Output Files

### Download Output
- FITS files organized by sector in `./data/fits_files/<sector>/`
- Download logs in `./logs/download_tess_lc.log`

### Anomaly Scoring Output
- CSV files with anomaly scores: `anomaly_scores_p3_YYYYMMDD_HHMMSS.csv`
- WRSS calculations: `wrss_p3p9_YYYYMMDD_HHMMSS.csv`
- Results stored in configured output directory

### Similarity Matching Output
- CSV files with similarity scores: `cosine_similarity_scores_YYYYMMDD_HHMMSS.csv`
- Filtered by label groups and WRSS threshold
- Results stored in configured output directory

### Feature Extraction Output
- **Fluxpowerstack**: `.npy` feature files and `.txt` filename lists in `./results/features/fluxpowerstack/`
- **Bin Means**: `.npy` feature files and `.txt` filename lists in `./results/features/bin_means/`
- **Advanced Features**: `.npy` feature files and `.txt` filename lists in `./results/features/advanced_features/`
- **PCA-reduced features**: Available for fluxpowerstack only with `_pca_N.npy` suffix
- **Feature extraction logs**: `./logs/feature_extraction*.log` and `./logs/advanced_features.log`

## Logging

### Log Files
- **Main Log**: `./logs/main.log` - General application events
- **System Status**: `./logs/system_status.log` - System resource monitoring
- **Download Log**: `./logs/download_tess_lc.log` - TESS download operations
- **Cosine Similarity**: `./logs/cosine_sim.log` - Similarity matching details
- **Feature Extraction**: `./logs/feature_extraction*.log` - Feature extraction details
- **Advanced Features**: `./logs/advanced_features.log` - Advanced feature extraction details

### Log Format
```
YYYY-MM-DD HH:MM:SS [LEVEL] Message
```

### Log Levels
- **MAIN**: General application messages
- **SYSTEM**: System resource information
- **ERROR**: Error messages and exceptions
- **LOG**: Detailed operation logs

## Performance Optimization

### Recommended Settings
- **Download**: Use 4-8 threads depending on network bandwidth
- **Scoring**: Use 4-8 workers based on CPU cores
- **Similarity**: Use 4-8 workers for multiprocessing
- **Feature Extraction**: Use 4-8 workers for multiprocessing
- **Memory**: Ensure sufficient RAM for large datasets (8GB+ recommended)

### Scaling Considerations
- **Large Datasets**: Process in batches using `max_downloads` and `start_index`
- **High Performance**: Use SSD storage for faster I/O operations
- **Network**: Ensure stable internet connection for downloads
- **Storage**: Monitor disk space during large downloads

## Troubleshooting

### Common Issues

**Download Failures**
- Check internet connection
- Verify TESS data availability
- Ensure sufficient disk space
- Check log files for specific errors

**Memory Issues**
- Reduce `max_workers` in configuration
- Process smaller batches
- Monitor system memory usage

**Performance Issues**
- Adjust thread/worker counts
- Use SSD storage
- Close unnecessary applications
- Monitor system resources

### Error Handling
- All operations include comprehensive error handling
- Failed operations are logged with detailed error messages
- System continues processing other files when individual failures occur
- Graceful degradation for missing dependencies

## Development

### Project Structure
```
clara-cli/
├── clara_benchmark/
│   ├── cli.py                 # Main CLI interface (clean, focused logic)
│   ├── tess_spoc_download/    # TESS data download modules
│   ├── urf_scoring/          # Anomaly detection modules
│   ├── cosine_similarity/    # Similarity matching modules
│   └── utils/                # Utility functions
│       ├── help_text.py      # Comprehensive CLI help documentation
│       ├── logging.py        # Logging utilities
│       └── system_stats.py   # System monitoring utilities
├── configs/                  # Configuration files
├── data/                     # Data storage
├── logs/                     # Log files
├── models/                   # Machine learning models
├── output/                   # Output files
└── tests/                    # Test files
```

### Adding New Commands
1. Add parser in `parse_args()` function
2. Implement command logic in `main()` function
3. Create corresponding configuration template
4. Add documentation to README

### Testing
- Unit tests available in `tests/` directory
- Run tests with: `python -m pytest tests/`
- Integration tests for full pipeline workflows

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

The MIT License is a permissive license that allows for:
- Commercial use
- Modification
- Distribution
- Private use

The only requirement is that the license and copyright notice be included in all copies or substantial portions of the software.



## Support

For issues and questions:
1. Check the troubleshooting section
2. Review log files for error details
3. Create an issue with detailed information
4. Include configuration files and error messages

## Version History

- **v1.0.0**: Initial release with download, scoring, and similarity matching
- **v1.1.0**: Added system monitoring and improved error handling
- **v1.2.0**: Enhanced multiprocessing and configuration management
- **v1.3.0**: Added dynamic config override system and comprehensive help documentation
