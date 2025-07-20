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

def parse_args():
    parser = argparse.ArgumentParser(description="CLARA Benchmark CLI")

    subparsers = parser.add_subparsers(dest="command", help="Sub-commands")

    # run command
    run_parser = subparsers.add_parser("download", help="Run benchmark pipeline")
    run_parser.add_argument("--config", type=str, required=False, default="./configs/download_config.json", help="Path to JSON config file")

    anomaly_parser = subparsers.add_parser("score", help="Score anomalies")
    anomaly_parser.add_argument("--config", type=str, required=False, default="./configs/urf_anomaly_scoring_config.json", help="Path to JSON config file")

    sim_score_parser = subparsers.add_parser("sim", help="Score anomalies")
    sim_score_parser.add_argument("--config", type=str, required=False, default="./configs/cosine_similarity_config.json", help="Path to JSON config file")

    
    parser.add_argument("--meta_message", type=str, required=False, default="", help="Meta message for the run")

    return parser.parse_args()


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
            
            with open(args.config, 'r') as f:
                config = json.load(f)
            for _, sector_config in config.items():
                keys = sector_config.keys()
                required_keys = ['sector_number', 'catalogues_folder', 'out_dir_tess', 'start_index', 'max_downloads', 'log_file', 'num_threads']
                if all(key in keys for key in required_keys):
                    pass
                else:
                    log_main(f"Sector {sector_config['sector_number']} is missing required keys", error=True)
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

            with open(args.config, 'r') as f:
                config = json.load(f)
            
            # check if all config keys are present with valid values
            required_keys = ['input_dirs', 'output_dir', 'max_workers', 'use_p3', 'use_p5', 'use_p9', 'calculate_wrss_p3p9']
            if all(key in config for key in required_keys):
                pass
            else:
                log_main(f"Config is missing required keys", error=True)
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

            with open(args.config, 'r') as f:
                config = json.load(f)

            # check if all config keys are present with valid values
            required_keys = ['fits_dir', 'input_df_path', 'output_dir', 'min_similarity_threshold', 'simbad_labelled_pca_model_path', 'simbad_labelled_pca_features_path', 'max_workers', 'gt_wrss_threshold', "save_label_groups"]
            if all(key in config for key in required_keys):
                pass
            else:
                log_main(f"Config is missing required keys", error=True)
                return
            
            log_main(f"Config details: {config}")

            # make sure all directories exist
            os.makedirs(config['output_dir'], exist_ok=True)
            os.makedirs(os.path.dirname(config['simbad_labelled_pca_model_path']), exist_ok=True)
            os.makedirs(os.path.dirname(config['simbad_labelled_pca_features_path']), exist_ok=True)

            # read the input df
            input_df = pd.read_csv(config['input_df_path'])

            # match the fits by cosine similarity using multiprocessing run cosine matching batch function
            results = run_cosine_matching_batch(filenames=list(set(input_df[input_df['score_weighted_root_sumnorm'] >= config['gt_wrss_threshold']]['filename'].tolist())),
                                                fits_dir=config['fits_dir'],
                                                simbad_labelled_pca_model_path=config['simbad_labelled_pca_model_path'],
                                                simbad_labelled_pca_features_path=config['simbad_labelled_pca_features_path'],
                                                compute_func=computeLombScargleTessLC,
                                                min_similarity_threshold=config['min_similarity_threshold'],
                                                workers=config['max_workers'])
            # add the results to the input df
            input_df = pd.concat([input_df, results], axis=1)

            input_df = input_df[input_df['label_group'].isin(config['save_label_groups'])]
            
            # save the output df
            outfile = config['output_dir'] + 'cosine_similarity_scores_' + datetime.now().strftime("%Y%m%d_%H%M%S") + '.csv'
            input_df.to_csv(outfile, index=False)

            log_main(f"Cosine similarity matching complete. Results saved in {outfile} with {len(results)} matches. Label groups: {input_df['label_group'].value_counts()}. GT WRSS threshold: {config['gt_wrss_threshold']}")


        else:
            print("No command specified. Use -h for help.")
    
    finally:
        # Stop the system stats logging thread
        stop_stats_logging.set()
        if stats_thread.is_alive():
            stats_thread.join(timeout=2)  # Wait up to 2 seconds for thread to finish
            if stats_thread.is_alive():
                log_main("System stats thread did not stop gracefully", error=True)