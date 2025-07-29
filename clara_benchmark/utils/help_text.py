"""
Help text module for CLARA CLI
Contains all help documentation and descriptions for the command-line interface.
"""

# Main parser help content
MAIN_DESCRIPTION = "CLARA CLI - Astronomical Data Processing Pipeline"

MAIN_EPILOG = """
Examples:
  # Download TESS data with custom config
  python -m clara_benchmark.cli download --config ./configs/download_config.json
  
  # Score anomalies with overridden parameters
  python -m clara_benchmark.cli score --max_workers 8 --use_p3 true --use_p9 true
  
  # Similarity matching with custom SQL query
  python -m clara_benchmark.cli sim --input_df_sql "SELECT * FROM input_df WHERE score_weighted_root_sumnorm >= 0.001"
  
  # Add metadata to any command
  python -m clara_benchmark.cli download --meta_message "Production run 2024-01-15"

Configuration Override System:
  The CLI automatically generates command-line arguments for every parameter in your config files.
  This allows you to override any config value without editing files.

  Dynamic Override Examples:
    # Override download parameters
    python -m clara_benchmark.cli download --num_threads 8 --max_downloads 2000 --start_index 100
    
    # Override scoring parameters  
    python -m clara_benchmark.cli score --max_workers 6 --use_p3 true --use_p5 false --calculate_wrss_p3p9 true
    
    # Override similarity parameters with complex SQL
    python -m clara_benchmark.cli sim --input_df_sql "SELECT filename FROM input_df WHERE score_weighted_root_sumnorm >= 0.001 LIMIT 100" --min_similarity_threshold 0.7 --max_workers 8
    
    # Mix config file and command-line overrides
    python -m clara_benchmark.cli sim --config ./custom_config.json --input_df_sql "SELECT * FROM input_df WHERE filename LIKE '%sector1%'"

  Automatic Type Detection:
    - Boolean values: --use_p3 true/false, --save false, --show true (also accepts: yes/no, 1/0, on/off)
    - Integer values: --max_workers 4, --num_threads 8
    - Float values: --min_similarity_threshold 0.6
    - String values: --output_dir "./custom_output/", --input_df_sql "SELECT * FROM input_df"

  Benefits:
    - No need to edit config files for quick parameter changes
    - Override any parameter from any config file
    - Maintains config file defaults when not overridden
    - Automatic type validation and conversion
    - Clear logging of which parameters were overridden

For more information, see the README.md file or visit the project documentation.
"""

# Subparser help content
SUBCOMMANDS_HELP = "Available commands"
SUBCOMMANDS_TITLE = "Commands"
SUBCOMMANDS_DESCRIPTION = "Choose one of the following commands to execute:"

# Download command help content
DOWNLOAD_HELP = "Download TESS light curve data from specified sectors"
DOWNLOAD_DESCRIPTION = """
Download TESS (Transiting Exoplanet Survey Satellite) light curve data with multi-threaded processing.
This command downloads FITS files from the TESS archive based on sector configurations.

Features:
  - Multi-threaded downloads for improved performance
  - Configurable sector parameters (start_index, max_downloads, num_threads)
  - Automatic directory creation and file management
  - Comprehensive logging and error handling
  - Real-time system monitoring during downloads

Configuration Override Support:
  All sector configuration parameters can be overridden via command-line arguments.
  For nested sector configs, overrides apply to all sectors in the configuration.

  Example overrides:
    --num_threads 8          # Override thread count for all sectors
    --max_downloads 2000     # Override download limit for all sectors
    --start_index 100        # Override starting index for all sectors
"""

# Score command help content
SCORE_HELP = "Score astronomical objects for anomalies using machine learning models"
SCORE_DESCRIPTION = """
Analyze TESS light curve data for anomalies using URF (Unsupervised Random Forest) models.
This command processes FITS files and generates anomaly scores using multiple model variants.

Features:
  - Multiple model support (P3, P5, P9 variants)
  - Parallel processing with configurable worker count
  - Memory-efficient streaming for large datasets
  - WRSS (Weighted Root Sum Square) calculations
  - Comprehensive result logging and file management
  - Real-time system monitoring during processing

Models:
  - P3: Primary anomaly detection model
  - P5: Secondary model variant
  - P9: Tertiary model variant
  - WRSS: Weighted comparison between P3 and P9 results

Configuration Override Support:
  All scoring parameters can be overridden via command-line arguments.

  Example overrides:
    --max_workers 8          # Override parallel worker count
    --use_p3 true           # Enable P3 model scoring
    --use_p5 false          # Disable P5 model scoring
    --use_p9 true           # Enable P9 model scoring
    --calculate_wrss_p3p9 true  # Enable WRSS calculations
"""

# Similarity command help content
SIM_HELP = "Match astronomical objects by cosine similarity to known classifications"
SIM_DESCRIPTION = """
Perform cosine similarity analysis on astronomical objects using PCA-reduced feature vectors.
This command matches unknown objects to known classifications based on light curve characteristics.

Features:
  - Pure SQL query support for flexible data filtering
  - PCA-based feature reduction for efficient comparison
  - Configurable similarity thresholds
  - Multi-threaded processing for large datasets
  - Label group filtering and classification
  - Real-time system monitoring during processing

Process:
  1. Load and filter input data using SQL queries
  2. Extract Lomb-Scargle features from light curves
  3. Apply PCA transformation to feature vectors
  4. Calculate cosine similarity with labeled reference data
  5. Classify objects based on similarity scores
  6. Filter results by label groups and thresholds

Configuration Override Support:
  All similarity matching parameters can be overridden via command-line arguments.
  SQL queries can be customized for complex filtering and selection.

  Example overrides:
    --input_df_sql "SELECT * FROM input_df WHERE score_weighted_root_sumnorm >= 0.001"
    --min_similarity_threshold 0.7
    --max_workers 8
    --fits_dir "./custom_fits_directory/"
    --output_dir "./custom_output_directory/"

  Label Group Filtering:
    --save_label_groups ["planet_like", "binary_star"]  # Save specific label groups
    --save_label_groups "all"                          # Save all matched objects regardless of label
    --save_label_groups "none"                         # Save only unmatched objects (label_group is null)

  SQL Query Examples:
    --input_df_sql "SELECT filename FROM input_df WHERE score_weighted_root_sumnorm >= 0.001 LIMIT 100"
    --input_df_sql "SELECT * FROM input_df WHERE filename LIKE '%sector1%' AND score_weighted_root_sumnorm >= 0.0001"
    --input_df_sql "SELECT filename, score_weighted_root_sumnorm FROM input_df WHERE score_weighted_root_sumnorm >= 0.001 ORDER BY score_weighted_root_sumnorm DESC"
"""

# Phase folding command help content
PHASE_FOLD_HELP = "Phase fold light curves for periodic analysis"
PHASE_FOLD_DESCRIPTION = """
Perform phase folding analysis on TESS light curves to identify periodic patterns.
This command folds light curves at specified periods to reveal periodic variations.

Features:
  - Phase folding at user-specified periods
  - Multiple period analysis capabilities
  - Configurable phase binning and smoothing
  - Output generation for further analysis
  - Multi-threaded processing for large datasets
  - Real-time system monitoring during processing

Process:
  1. Load light curve data from FITS files
  2. Calculate phase values for specified periods
  3. Bin and fold light curves at each period
  4. Generate phase-folded light curve plots
  5. Save results for further analysis

Configuration Override Support:
  All phase folding parameters can be overridden via command-line arguments.

  Example overrides:
    --period 2.5                    # Override folding period
    --phase_bins 50                 # Override number of phase bins
    --output_dir "./folded_curves/" # Override output directory
    --max_workers 8                 # Override parallel worker count
"""


# Argument help content
CONFIG_HELP = "Path to JSON configuration file containing {command} parameters"
META_MESSAGE_HELP = "Add a descriptive message to the run logs for tracking and documentation purposes"

def get_config_help(command_name):
    """Get config help text for a specific command"""
    return CONFIG_HELP.format(command=command_name)

def get_override_help_text(key, value):
    """Generate help text for dynamic config override arguments"""
    if isinstance(value, bool):
        return f"Override {key} from config (boolean: true/false, default: {value})"
    elif isinstance(value, int):
        return f"Override {key} from config (integer, default: {value})"
    elif isinstance(value, float):
        return f"Override {key} from config (float, default: {value})"
    else:
        return f"Override {key} from config (string, default: {value})" 