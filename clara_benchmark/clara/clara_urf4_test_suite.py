import glob
import os
import traceback
import test_helpers.gen_rep_test_set as grts
import test_helpers.toi_importance as timp
import clara_toi_functions as ctoi
import clara_urf_predictor as cupred
import pandas as pd
from sklearn.metrics import auc
import random
from IPython.display import clear_output
import time as tm

def test_urf4_subvariant_runmodel(model_path, sector_data_path, sector_no, toi_fits_files=None, test_results_save_dir=None, gen_sample_subsets=True, test_sets_catalogues_save_dir=None, n_test_samples=None):

    try:
        all_fits_fnames = [x.split("/")[-1] for x in glob.glob(os.path.join(sector_data_path, '*.fits'))]
        if not all_fits_fnames:
            print("[-] No FITS files found in the sector data path.")
            return
    except Exception as e:
        print(f"[-] Error gathering FITS files: {e}")
        return

    if toi_fits_files is None:
        try:
            print("[+] Gathering Sector TOI Data\n")
            sector_toi_tics = ctoi.get_sector_tic_ids(sector=sector_no)
            toi_fits_files = [x for x in all_fits_fnames if x.endswith(".fits") and int(x.split("-")[2].lstrip("0")) in sector_toi_tics]
            print(f"[+] Number of TOI FITS in whole sector: {len(toi_fits_files)}")
        except Exception as e:
            print(f"[-] Error fetching TOI FITS files: {e}")
            return

    test_sample_size = 4000
    if n_test_samples is None:
        n_test_samples = 10

    if test_sets_catalogues_save_dir is None:
        test_sets_catalogues_save_dir = "../test/test_set_catalogues/"
    if test_results_save_dir is None:
        test_results_save_dir = "../test/results/"
    test_logs_save_dir = "../test/logs/"
    os.makedirs(test_sets_catalogues_save_dir, exist_ok=True)
    os.makedirs(test_results_save_dir, exist_ok=True)
    os.makedirs(test_logs_save_dir, exist_ok=True)
    
    error_log_path = os.path.join(test_logs_save_dir, "error_log.txt")

    max_workers = 4

    if gen_sample_subsets == True:

        # Step 1: Generate test sets
        for i in range(n_test_samples):
            try:
                seed = random.randint(0, 10000)
                this_test_sample = grts.generate_representative_test_set(
                    all_fits_fnames, toi_fits_files, sample_size=test_sample_size, seed=seed)
                this_test_sample_catalogue_file = os.path.join(test_sets_catalogues_save_dir, f"test_set_{i}_fits_files.txt")
                with open(this_test_sample_catalogue_file, "w") as f:
                    for fname in this_test_sample:
                        f.write(fname + "\n")
            except Exception as e:
                with open(error_log_path, "a") as elog:
                    elog.write(f"[Set Generation Error] Test Set {i}: {e}\n")
                    elog.write(traceback.format_exc())
                continue
    
        print(f"[+] Generated {n_test_samples} test sets in {test_sets_catalogues_save_dir}\n")

    # Step 2: Run model on each test set
    sample_files = glob.glob(os.path.join(test_sets_catalogues_save_dir, "*.txt"))
    c=0
    for test_sample in sample_files:
        c+=1
        clear_output(wait=True)
        print(f"{c}/{len(sample_files)}")
        try:
            save_csv_path = os.path.join(
                test_results_save_dir, os.path.basename(test_sample).replace(".txt", "_results.csv"))

            with open(test_sample, "r") as f:
                test_sample_fits_list = [line.strip() for line in f.readlines()]
            
            test_sample_fits_list = [x for x in test_sample_fits_list if x in all_fits_fnames]

            print(f"[+] Running model on {test_sample} ({len(test_sample_fits_list)} files)...")

            cupred.get_anomaly_scores_from_folder_parallelized_streamed_mp(
                folder_path=sector_data_path,
                model_path=model_path,
                save_csv=save_csv_path,
                save_dir=None,
                max_workers=max_workers,
                subset=test_sample_fits_list
            )
        except Exception as e:
            with open(error_log_path, "a") as elog:
                elog.write(f"[Model Run Error] Test Set File: {test_sample}\nError: {e}\n")
                elog.write(traceback.format_exc())
            print(f"[-] Error running model on {test_sample}, logged.")
            continue
        tm.sleep(300)
        
def calculate_auc_metrics_from_results(
    results_dir,
    sector_no,
    threshold_steps=100,
    first_n = None
):

    all_fits_fnames = [x.split("/")[-1] for x in glob.glob(os.path.join(f"../../downloaded_lc/tess_lc/{sector_no}/", '*.fits'))]
    print("Accessing TOI data from: https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv")
    sector_toi_tics = ctoi.get_sector_tic_ids(sector=sector_no)
    toi_filenames = [x for x in all_fits_fnames if x.endswith(".fits") and int(x.split("-")[2].lstrip("0")) in sector_toi_tics]
    
    all_results = []
    for path in glob.glob(os.path.join(results_dir, "*_results.csv")):
        try:
            df = pd.read_csv(path)
            all_results.append([df, os.path.basename(path)])
        except Exception as e:
            print(f"Skipping {path}: {e}")
    
    if not all_results:
        raise RuntimeError("No valid result CSVs found.")

    # df_all = pd.concat(all_results, ignore_index=True)
    recall_auc_list = []
    importance_auc_list = []
    toi_recall_list = []

    if first_n:
        all_results = all_results[:first_n]
    for result, sample_name in all_results:
        print(f"Calculating auc metrics for result: {sample_name}")
        result['is_toi'] = result['filename'].isin(toi_filenames)

        sample_tois = result[result["filename"].isin(toi_filenames)]["filename"].tolist()
        sample_tic_ids = [int(x.split("-")[2].lstrip("0")) for x in sample_tois]
        # result_toi_df = timp.get_toi_importance_scores_from_filenames(
                            #     toi_filenames=sample_tois,
                            #     sector_tic_ids=sample_tic_ids
                            # )
    
        total_tois = result['is_toi'].sum()
        total_files = len(result)

        if total_tois == 0:
            raise ValueError("No TOIs found in the combined results.")

        thresholds = sorted(result['anomaly_score'].quantile(q=i/threshold_steps) for i in range(threshold_steps + 1))

        recall_values = []
        importance_values = []

        for thresh in thresholds:
            df_thresh = result[result['anomaly_score'] >= thresh]
    
            if len(df_thresh) == 0:
                continue
            
            n_flagged = len(df_thresh)
            n_flagged_toi = df_thresh['is_toi'].sum()
    
            recall = n_flagged_toi / total_tois
            thresh_tois_df = df_thresh[df_thresh["filename"].isin(sample_tois)]
            # importance_norm = result_toi_df[result_toi_df["filename"].isin(thresh_tois_df["filename"])]["normalized_score"].mean()
    
            recall_values.append(recall)
            # importance_values.append(importance_norm)
    
        # recall_auc = auc(thresholds[:len(recall_values)], recall_values)
        # importance_auc = auc(thresholds[:len(importance_values)], importance_values)

        # recall_auc_list.append([recall_auc, sample_name])
        # importance_auc_list.append([importance_auc, sample_name])
        avg_rec = 0
        for rec in recall_values:
            avg_rec += rec
        avg_rec = avg_rec/len(recall_values)
        print(avg_rec)
    # return {
    #     "toi_recall_auc_list": recall_auc_list,
    #     "toi_importance_auc": importance_auc_list,
    # }

def calculate_auc_metrics_from_results_v3(
    results_dir,
    sector_no,
    fits_dir = None,
    first_n=None,
    top_percent=20,
    csv_endswith=None
):
    import glob
    import os
    import traceback
    import test_helpers.gen_rep_test_set as grts
    import test_helpers.toi_importance as timp
    import clara_toi_functions as ctoi
    import clara_urf_predictor as cupred
    import pandas as pd
    from scipy.integrate import trapezoid
    from IPython.display import clear_output
    import numpy as np

    def mean_importance_in_top_n(df, toi_filenames, importance_df, top_percent=10):
        n = int(len(df) * top_percent / 100)
        df_top = df.sort_values("anomaly_score", ascending=False).head(n)
        df_top = df_top[df_top["filename"].isin(toi_filenames)]
        return df_top["normalized_score"].mean()

    def anomaly_score_percentile_range_for_top_important_tois(
        result_df,
        toi_filenames,
        top_n_percent=10
    ):
        # Filter to TOIs with valid importance scores
        df = result_df[result_df["filename"].isin(toi_filenames)].copy()
        df = df.dropna(subset=["normalized_score"])
    
        if df.empty:
            raise ValueError("No valid TOIs with importance scores found in result_df.")
    
        # Get top-N% TOIs by importance
        n_top = max(1, int(len(df) * top_n_percent / 100))
        df_top = df.sort_values("normalized_score", ascending=False).head(n_top)
    
        # Rank entire dataset by anomaly score to get percentiles
        df_all = result_df.copy()
        df_all = df_all.sort_values("anomaly_score", ascending=False).reset_index(drop=True)
        df_all["anomaly_percentile"] = df_all.index / len(df_all) * 100
    
        # Merge top important TOIs with their anomaly percentiles
        df_top = df_top.merge(
            df_all[["filename", "anomaly_percentile"]],
            on="filename",
            how="left"
        )
    
        lower = df_top["anomaly_percentile"].min()
        upper = df_top["anomaly_percentile"].max()
    
        return lower, upper, df_top.sort_values("anomaly_percentile")

    if fits_dir is None:
        all_fits_fnames = [x.split("/")[-1] for x in glob.glob(os.path.join(f"../../downloaded_lc/tess_lc/{sector_no}/", '*.fits'))]
    else:
        all_fits_fnames = [x.split("/")[-1] for x in glob.glob(os.path.join(fits_dir, '*.fits'))]
    print("Accessing TOI data from: https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv")
    sector_toi_tics = ctoi.get_sector_tic_ids(sector=sector_no)
    toi_filenames = [x for x in all_fits_fnames if x.endswith(".fits") and int(x.split("-")[2].lstrip("0")) in sector_toi_tics]

    all_results = []
    if csv_endswith is None:
        for path in glob.glob(os.path.join(results_dir, "*_results.csv")):
            try:
                df = pd.read_csv(path)
                all_results.append((df, os.path.basename(path)))
            except Exception as e:
                print(f"Skipping {path}: {e}")

    else:
        for path in glob.glob(os.path.join(results_dir, "*"+csv_endswith)):
            try:
                df = pd.read_csv(path)
                all_results.append((df, os.path.basename(path)))
            except Exception as e:
                print(f"Skipping {path}: {e}")

    
    if not all_results:
        raise RuntimeError("No valid result CSVs found.")

    auc_records = []

    if first_n:
        all_results = all_results[:first_n]

    for result, variant_name in all_results:
        print(f"Calculating auc metrics for result: {variant_name}")
        result = result.copy()
        result['is_toi'] = result['filename'].isin(toi_filenames)

        sample_tois = result[result["filename"].isin(toi_filenames)]["filename"].tolist()
        sample_tic_ids = [int(x.split("-")[2].lstrip("0")) for x in sample_tois]

        result_toi_df = timp.get_toi_importance_scores_from_filenames(
            toi_filenames=sample_tois,
            sector_tic_ids=sample_tic_ids
        )

        print(result_toi_df.columns.tolist())

        result["tic_id"] = result["filename"].apply(lambda f: int(f.split("-")[2].lstrip("0")))
        result = result.merge(result_toi_df[["tic_id", "normalized_score"]], how="left", left_on="tic_id", right_on="tic_id")

        n_total = len(result)
        n_tois_total = result["is_toi"].sum()
        total_importance = result.loc[result["is_toi"], "normalized_score"].sum()

        if n_tois_total == 0:
            print(f"[!] No TOIs found in result: {variant_name}. Skipping.")
            continue

        thresholds = list(range(5, 101, 5))
        recall_values = []
        importance_values = []
        threshold_fracs = []
        tot_anomaly_toi = result[result["is_toi"]]["anomaly_score"].gt(0).sum()
        n_anomaly = result["anomaly_score"].gt(0).sum()
        anomaly_rate = n_anomaly / n_total
        toi_recall_tot = tot_anomaly_toi / n_tois_total
        result = result.sort_values("anomaly_score", ascending=False).reset_index(drop=True)

        for threshold in thresholds:
            top_n = int(n_total * threshold / 100)
            df_thresh = result.iloc[:top_n]
            n_flagged_toi = df_thresh["is_toi"].sum()
            toi_recall = n_flagged_toi / n_tois_total
            importance_flagged = df_thresh.loc[df_thresh["is_toi"], "normalized_score"].mean() if not df_thresh["is_toi"].empty else 0.0
            recall_values.append(toi_recall)
            importance_values.append(importance_flagged)
            threshold_fracs.append(threshold / 100)

        recall_auc = np.trapezoid(recall_values, threshold_fracs)
        importance_auc = np.trapezoid(importance_values, threshold_fracs)

        mean_top_n_importance = mean_importance_in_top_n(result[result["anomaly_score"] > 0], toi_filenames, result_toi_df, top_percent=top_percent)

        lower_10, upper_10, ranked_top_tois_10 = anomaly_score_percentile_range_for_top_important_tois(
            result_df=result[result["anomaly_score"] > 0],
            toi_filenames=toi_filenames,
            top_n_percent=10
        )

        lower_20, upper_20, ranked_top_tois_20 = anomaly_score_percentile_range_for_top_important_tois(
            result_df=result[result["anomaly_score"] > 0],
            toi_filenames=toi_filenames,
            top_n_percent=20
        )

        auc_records.append({
            "variant": variant_name,
            "auc_toi_recall": recall_auc,
            "auc_importance_recall": importance_auc,
            "mean_importance_top{}p".format(top_percent): mean_top_n_importance,
            "toi_recall": toi_recall_tot,
            "top_10p_by_importance_percentile_range_lower":lower_10,
            "top_10p_by_importance_percentile_range_upper":upper_10,
            "top_10p_by_importance_percentile_range_ranked_top_tois":ranked_top_tois_10,
            "top_20p_by_importance_percentile_range_lower":lower_20,
            "top_20p_by_importance_percentile_range_upper":upper_20,
            "top_20p_by_importance_percentile_range_ranked_top_tois":ranked_top_tois_20,
            "anomaly_rate": anomaly_rate
        })

    return pd.DataFrame(auc_records)

