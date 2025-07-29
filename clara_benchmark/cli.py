import os
import argparse
import json
import threading
import time
from clara_benchmark.tess_spoc_download.download_fits import *
from clara_benchmark.utils.logging import *
from datetime import datetime
from clara_benchmark.urf_scoring.score import *
from clara_benchmark.urf_scoring.wrss import *
import clara_benchmark.utils.suppress_warnings as suppress_warnings
from clara_benchmark.cosine_similarity.cos_score import *
from tqdm import tqdm
from clara_benchmark.utils.help_text import *
from clara_benchmark.phase_folding.pf import *
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(
        description=MAIN_DESCRIPTION,
        epilog=MAIN_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(
        dest="command", 
        help=SUBCOMMANDS_HELP,
        title=SUBCOMMANDS_TITLE,
        description=SUBCOMMANDS_DESCRIPTION
    )

    # Download command
    run_parser = subparsers.add_parser(
        "download", 
        help=DOWNLOAD_HELP,
        description=DOWNLOAD_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    run_parser.add_argument(
        "--config", 
        type=str, 
        required=False, 
        default="./configs/download_config.json", 
        help=get_config_help("download")
    )
    add_dynamic_config_arguments(run_parser, "./configs/download_config.json")

    # Score command
    anomaly_parser = subparsers.add_parser(
        "score", 
        help=SCORE_HELP,
        description=SCORE_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    anomaly_parser.add_argument(
        "--config", 
        type=str, 
        required=False, 
        default="./configs/urf_anomaly_scoring_config.json", 
        help=get_config_help("scoring")
    )
    add_dynamic_config_arguments(anomaly_parser, "./configs/urf_anomaly_scoring_config.json")

    # Similarity command
    sim_score_parser = subparsers.add_parser(
        "sim", 
        help=SIM_HELP,
        description=SIM_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sim_score_parser.add_argument(
        "--config", 
        type=str, 
        required=False, 
        default="./configs/cosine_similarity_config.json", 
        help=get_config_help("similarity matching")
    )
    add_dynamic_config_arguments(sim_score_parser, "./configs/cosine_similarity_config.json")
    
    phase_fold_parser = subparsers.add_parser(
        "fold", 
        help=PHASE_FOLD_HELP,
        description=PHASE_FOLD_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    phase_fold_parser.add_argument( 
        "--config", 
        type=str, 
        required=False, 
        default="./configs/phase_folding_config.json", 
        help=get_config_help("phase folding")
    )
    add_dynamic_config_arguments(phase_fold_parser, "./configs/phase_folding_config.json")


    # Global meta message
    parser.add_argument(
        "--meta_message", 
        type=str, 
        required=False, 
        default="", 
        help=META_MESSAGE_HELP
    )

    return parser.parse_args()


def load_and_filter_dataframe(csv_path, sql_query, command_name="Command"):
    """
    Load CSV data into SQLite database and apply SQL filtering
    
    Args:
        csv_path: Path to the CSV file
        sql_query: SQL query to filter the data
        command_name: Name of the command for logging
    
    Returns:
        pandas.DataFrame: Filtered DataFrame
    
    Raises:
        Exception: If SQL query execution fails
    """
    # Read the input df from CSV file and load into SQLite for pure SQL operations
    input_df = pd.read_csv(csv_path)
    log_main(f"{command_name} - Original input df shape: {input_df.shape}")
    
    # Create in-memory SQLite database and load DataFrame
    import sqlite3
    conn = sqlite3.connect(':memory:')
    input_df.to_sql('input_df', conn, index=False, if_exists='replace')
    
    # Execute pure SQL query
    log_main(f"{command_name} - Executing SQL query: {sql_query}")
    
    try:
        filtered_df = pd.read_sql(sql_query, conn)
        log_main(f"{command_name} - Filtered input df shape: {filtered_df.shape}")
        return filtered_df
    except Exception as e:
        log_main(f"{command_name} - Error executing SQL query: {e}", error=True)
        raise
    finally:
        conn.close()


def str2bool(v):
    """
    Convert string to boolean for argparse
    
    Args:
        v: String value to convert
    
    Returns:
        bool: True for 'true', 'yes', '1', 'on'; False for 'false', 'no', '0', 'off'
    
    Raises:
        argparse.ArgumentTypeError: If value cannot be converted to boolean
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1', 'on'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', 'off'):
        return False
    else:
        import argparse
        raise argparse.ArgumentTypeError(f'Boolean value expected, got {v}')


def add_dynamic_config_arguments(parser, config_path):
    """
    Dynamically add command-line arguments for all keys in a config file
    
    Args:
        parser: ArgumentParser object to add arguments to
        config_path: Path to the config file to read keys from
    """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Add arguments for each config key
        for key, value in config.items():
            # Determine argument type based on value type
            if isinstance(value, bool):
                arg_type = str2bool
            elif isinstance(value, int):
                arg_type = int
            elif isinstance(value, float):
                arg_type = float
            else:
                arg_type = str
            
            # Add the argument with help text from help_text module
            parser.add_argument(
                f"--{key}", 
                type=arg_type, 
                required=False, 
                help=get_override_help_text(key, value)
            )
    except (FileNotFoundError, json.JSONDecodeError) as e:
        # If config file doesn't exist or is invalid, skip adding arguments
        pass


def validate_config_keys(config, required_keys, command_name):
    """
    Validate that all required keys are present in the config
    
    Args:
        config: Configuration dictionary
        required_keys: List of required keys
        command_name: Name of the command for logging
    
    Returns:
        bool: True if all keys are present, False otherwise
    """
    if all(key in config for key in required_keys):
        return True
    else:
        missing_keys = [key for key in required_keys if key not in config]
        log_main(f"{command_name} config is missing required keys: {missing_keys}", error=True)
        return False


def load_config_with_overrides(config_path, args):
    """
    Load config from file and override with command-line arguments
    
    Args:
        config_path: Path to config file
        args: Parsed arguments object
    
    Returns:
        dict: Updated config dictionary
    """
    # Load base config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Override with command-line arguments if provided
    for key, value in vars(args).items():
        if value is not None and key in config:
            config[key] = value
            log_main(f"Overriding config {key} with command-line value: {value}")
        elif value is not None and args.command == "download":
            # For download command, check if the key exists in any sector config
            for sector_key, sector_config in config.items():
                if key in sector_config:
                    sector_config[key] = value
                    log_main(f"Overriding sector {sector_key} config {key} with command-line value: {value}")
    
    return config


def system_stats_logger(stop_event, interval=5):
    """
    Background thread function that logs system stats at regular intervals
    
    Args:
        stop_event: threading.Event to signal when to stop logging
        interval: Logging interval in seconds (default 5)
    """
    log_system_status("System stats logging started", stdout=False)
    
    while not stop_event.is_set():
        try:
            log_system_status("Periodic system status check", stdout=False)
            # Wait for the specified interval or until stop is signaled
            if stop_event.wait(interval):
                break
        except Exception as e:
            log_main(f"Error in system stats logging: {e}", error=True)
            # Continue logging even if there's an error
            if stop_event.wait(interval):
                break
    
    log_system_status("System stats logging stopped", stdout=False)


def main():
    
    args = parse_args()

    # if meta message is provided, add it to the log file
    if args.meta_message:
        log_main(f"Meta message: {args.meta_message}", stdout=True)

    # Start system stats logging in background thread for entire main function duration
    stop_stats_logging = threading.Event()
    stats_thread = threading.Thread(
        target=system_stats_logger, 
        args=(stop_stats_logging, 5),  # Log every 5 seconds
        daemon=True  # Daemon thread will be terminated when main thread exits
    )
    stats_thread.start()
    
    try:
        if args.command == "download":
            log_main(f"Downloading with config: {args.config}")
            
            # Load config with command-line overrides
            config = load_config_with_overrides(args.config, args)
            
            for sector_key, sector_config in config.items():
                required_keys = ['sector_number', 'catalogues_folder', 'out_dir_tess', 'start_index', 'max_downloads', 'log_file', 'num_threads']
                if not validate_config_keys(sector_config, required_keys, f"Sector {sector_key}"):
                    continue
                log_main(f"Downloading sector {sector_config['sector_number']} with config: {sector_config} ", stdout=True)
                download_tess_sector_threaded(sector_number=sector_config['sector_number'],
                                              catalogues_folder=sector_config['catalogues_folder'],
                                              out_dir_tess=sector_config['out_dir_tess'],
                                              start_index=sector_config['start_index'],
                                              max_downloads=sector_config['max_downloads'],
                                              log_file=sector_config['log_file'],
                                              num_threads=sector_config['num_threads'])

        elif args.command == "score":
            log_main(f"Scoring anomalies with config: {args.config}")

            # Load config with command-line overrides
            config = load_config_with_overrides(args.config, args)
            
            # check if all config keys are present with valid values
            required_keys = ['input_dirs', 'output_dir', 'max_workers', 'use_p3', 'use_p5', 'use_p9', 'calculate_wrss_p3p9']
            if not validate_config_keys(config, required_keys, "Score"):
                return

            log_main(f"Config details: {config}")

            for input_dir in config['input_dirs']:
                log_main(f"Scoring anomalies in {input_dir}")
                
                os.makedirs(config['output_dir'], exist_ok=True)
                
                if config['use_p3']:
                    log_main(f"Scoring anomalies in {input_dir} with p3 model")
                    save_csv_fname_p3 = config['output_dir'] + 'anomaly_scores_p3_' + datetime.now().strftime("%Y%m%d_%H%M%S") + '.csv' 
                    get_anomaly_scores_from_folder_parallelized_streamed_mp(folder_path=input_dir,
                                                                            model_path=config['p3model_path'],
                                                                            save_csv=save_csv_fname_p3,
                                                                            max_workers=config['max_workers'],
                                                                            )
                if config['use_p5']:
                    log_main(f"Scoring anomalies in {input_dir} with p5 model")
                    save_csv_fname_p5 = config['output_dir'] + 'anomaly_scores_p5_' + datetime.now().strftime("%Y%m%d_%H%M%S") + '.csv' 
                    get_anomaly_scores_from_folder_parallelized_streamed_mp(folder_path=input_dir,
                                                                            model_path=config['p5model_path'],
                                                                            save_csv=save_csv_fname_p5,
                                                                            max_workers=config['max_workers'],
                                                                            )
                if config['use_p9']:
                    log_main(f"Scoring anomalies in {input_dir} with p9 model")
                    save_csv_fname_p9 = config['output_dir'] + 'anomaly_scores_p9_' + datetime.now().strftime("%Y%m%d_%H%M%S") + '.csv' 
                    get_anomaly_scores_from_folder_parallelized_streamed_mp(folder_path=input_dir,
                                                                            model_path=config['p9model_path'],
                                                                            save_csv=save_csv_fname_p9,
                                                                            max_workers=config['max_workers'],
                                                                            )
                if config['calculate_wrss_p3p9']:
                    log_main(f"Calculating WRSS for p3 and p9 models")
                    wrss_csv_fname = config['output_dir'] + 'wrss_p3p9_' + datetime.now().strftime("%Y%m%d_%H%M%S") + '.csv' 
                    calculate_wrss_p3p9(p3_csv_fname=save_csv_fname_p3,
                                        p9_csv_fname=save_csv_fname_p9,
                                        save_csv=wrss_csv_fname)
                
                log_main(f"All tasks complete")
                log_main(f"Results saved in {config['output_dir']}")


        elif args.command == "sim":
            log_main(f"Matching fits by cosine similarity with config: {args.config}")

            # Load config with command-line overrides
            config = load_config_with_overrides(args.config, args)

            # check if all config keys are present with valid values
            required_keys = ['fits_dir', 'input_df_path', 'input_df_sql', 'output_dir', 'min_similarity_threshold', 'simbad_labelled_pca_model_path', 'simbad_labelled_pca_features_path', 'max_workers', 'save_label_groups']
            if not validate_config_keys(config, required_keys, "Similarity"):
                return            
            log_main(f"Config details: {config}")

            # make sure all directories exist
            os.makedirs(config['output_dir'], exist_ok=True)
            os.makedirs(os.path.dirname(config['simbad_labelled_pca_model_path']), exist_ok=True)
            os.makedirs(os.path.dirname(config['simbad_labelled_pca_features_path']), exist_ok=True)

                        # Load and filter data using SQL query
            try:
                filtered_df = load_and_filter_dataframe(
                    csv_path=config['input_df_path'],
                    sql_query=config['input_df_sql'],
                    command_name="Similarity"
                )
                print(filtered_df.head())
            except Exception as e:
                return

            # match the fits by cosine similarity using multiprocessing run cosine matching batch function
            results = run_cosine_matching_batch(filenames=list(set(filtered_df['filename'].tolist())),
                                                fits_dir=config['fits_dir'],
                                                simbad_labelled_pca_model_path=config['simbad_labelled_pca_model_path'],
                                                simbad_labelled_pca_features_path=config['simbad_labelled_pca_features_path'],
                                                compute_func=computeLombScargleTessLC,
                                                min_similarity_threshold=config['min_similarity_threshold'],
                                                workers=config['max_workers'])
            
            # add the results to the filtered df
            filtered_df = pd.concat([filtered_df, results], axis=1)

            # Apply label group filtering
            # add option to check for "all" or "none" in save label groups
            if config['save_label_groups'] == "all":
                filtered_df = filtered_df
            elif config['save_label_groups'] == "none":
                filtered_df = filtered_df[filtered_df['label_group'].isna()]
            else:
                filtered_df = filtered_df[filtered_df['label_group'].isin(config['save_label_groups'])]
            
            # save the output df
            outfile = config['output_dir'] + 'cosine_similarity_scores_' + datetime.now().strftime("%Y%m%d_%H%M%S") + '.csv'
            filtered_df.to_csv(outfile, index=False)

            log_main(f"Cosine similarity matching complete. Results saved in {outfile} with {len(results)} matches. Label groups: {filtered_df['label_group'].value_counts()}")

        elif args.command == "fold":
            log_main(f"Phase folding with config: {args.config}")
            # Load config with command-line overrides
            config = load_config_with_overrides(args.config, args)
            # check if all config keys are present with valid values
            required_keys = ['input_df_path', 'output_dir', 'max_workers']
            if not validate_config_keys(config, required_keys, "Phase folding"):
                return

            # Load and filter data using SQL query
            try:
                filtered_df = load_and_filter_dataframe(
                    csv_path=config['input_df_path'],
                    sql_query=config['input_df_sql'],
                    command_name="Phase folding"
                )
                print(filtered_df.head())
            except Exception as e:
                return



            # phase fold the fits
            phase_fold_batch(input_df=filtered_df,
                             fits_dir=config['fits_dir'],
                             output_dir=config['output_dir'],
                             min_power_threshold=config['min_power_threshold'],
                             min_period_threshold=config['min_period_threshold'],
                             max_period_threshold=config['max_period_threshold'],
                             bins=config['bins'],
                             show=config['show'],
                             phase_shift=config['phase_shift'],
                             column_name=config['column_name'],
                             save=bool(config['save'])
                             )

            log_main(f"Phase folding complete. Results saved in {config['output_dir']}")

        else:
            print("No command specified. Use -h for help.")
    
    finally:
        # Stop the system stats logging thread
        stop_stats_logging.set()
        if stats_thread.is_alive():
            stats_thread.join(timeout=2)  # Wait up to 2 seconds for thread to finish
            if stats_thread.is_alive():
                log_main("System stats thread did not stop gracefully", error=True)