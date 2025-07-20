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
    "input_df_path": "./results/anomaly_scores/anomaly_scores_p3_20240101_120000.csv",
    "output_dir": "./results/similarity_scores/",
    "min_similarity_threshold": 0.3,
    "gt_wrss_threshold": 0.5,
    "save_label_groups": ["planet_like", "binary_star", "stellar"],
    "simbad_labelled_pca_model_path": "./models/simbad_pca_model.pkl",
    "simbad_labelled_pca_features_path": "./data/simbad_features.csv",
    "max_workers": 4
}
```

### Global Options
- `--meta_message`: Add a descriptive message to the run logs
- `--config`: Specify custom configuration file path

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
| `input_df_path` | Path to input DataFrame | Required |
| `output_dir` | Output directory for results | Required |
| `min_similarity_threshold` | Minimum similarity score | 0.3 |
| `gt_wrss_threshold` | Ground truth WRSS threshold | Required |
| `save_label_groups` | Label groups to save | Required |
| `max_workers` | Number of worker processes | 4 |

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

## Logging

### Log Files
- **Main Log**: `./logs/main.log` - General application events
- **System Status**: `./logs/system_status.log` - System resource monitoring
- **Download Log**: `./logs/download_tess_lc.log` - TESS download operations
- **Cosine Similarity**: `./logs/cosine_sim.log` - Similarity matching details

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
│   ├── cli.py                 # Main CLI interface
│   ├── tess_spoc_download/    # TESS data download modules
│   ├── urf_scoring/          # Anomaly detection modules
│   ├── cosine_similarity/    # Similarity matching modules
│   └── utils/                # Utility functions
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

[Add license information here]

## Contributing

[Add contribution guidelines here]

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
