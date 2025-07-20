import os
import pandas as pd
from IPython.display import clear_output
import time
import joblib
import os
from tqdm import tqdm
import numpy as np
from astropy.timeseries import LombScargle
import matplotlib.pyplot as plt
from astropy.table import Table
from lightkurve import LightCurve
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from astropy.stats import sigma_clip
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import clara_benchmark.clara.clara_utils as clara_utils
import clara_benchmark.clara.clara_feature_extraction_parallel as cfep
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager
from clara_benchmark.utils.logging import log

def load_nonzero_anomaly_scores(csv_path):
    """
    Load a saved anomaly score CSV and return rows with score > 0.

    Args:
        csv_path (str): Path to the CSV file.

    Returns:
        valid_rows (list): List of dicts with non-zero anomaly scores.
        count (int): Number of valid rows.
    """
    df = pd.read_csv(csv_path)

    # Clean and filter
    df = df.dropna(subset=["anomaly_score"])
    df = df[df["anomaly_score"] > 0]

    valid_rows = df.to_dict(orient="records")
    count = len(valid_rows)

    print(f"✅ {count} non-zero anomaly scores found in: {csv_path}")
    return valid_rows, count

def get_anomaly_score(fits_path, model_path="./saved_URF_models/mg21_urfparams_model.pkl"):
    try:
        clf = joblib.load(model_path)
        table = Table.read(fits_path, hdu=1)
        lc = LightCurve(time=table["TIME"], flux=table["PDCSAP_FLUX"])
        feature_vector, _, _ = cfep.computeLombScargleTessLC(lc)
        if feature_vector.shape[0] != 4000:
            return None
        feature_vector = feature_vector.reshape(1, -1)
        prob_real = clf.predict_proba(feature_vector)[0, 1]
        return 1 - prob_real
    except Exception as e:
        print(e)
        return None

def score_one(args):
    fpath, model_path, save_csv, save_dir, lock, counter, counter_lock = args
    try:
        import os
        from astropy.table import Table
        from lightkurve import LightCurve
        import joblib
        import numpy as np
        import pandas as pd
        import clara_benchmark.clara.clara_utils as clara_utils
        import clara_benchmark.clara.clara_feature_extraction_parallel as cfep

        # Load model and light curve
        clf = joblib.load(model_path)
        table = Table.read(fpath, hdu=1)
        lc = LightCurve(time=table["TIME"], flux=table["PDCSAP_FLUX"])
        feature_vector, _, _ = cfep.computeLombScargleTessLC(lc)

        if feature_vector.shape[0] != 4000:
            return

        feature_vector = feature_vector.reshape(1, -1)
        prob_real = clf.predict_proba(feature_vector)[0, 1]
        anomaly_score = 1 - prob_real
        tic = clara_utils.tic_from_fits_fname(os.path.basename(fpath))
        row = f"{tic},{os.path.basename(fpath)},{anomaly_score}\n"

        if save_dir and tic is not None:
            df = pd.DataFrame([[tic, os.path.basename(fpath), anomaly_score]],
                              columns=["tic", "filename", "anomaly_score"])
            df.to_csv(os.path.join(save_dir, f"{anomaly_score}_{tic}_{os.path.basename(fpath)}_score.csv"),
                      index=False)

        if save_csv and tic is not None:
            with lock:
                with open(save_csv, "a") as f:
                    f.write(row)

        # ✅ Update and print live counter
        if anomaly_score != 0:
            with counter_lock:
                counter.value += 1
                # clear_output(wait=True)
                log(f"Non zero anomaly scores: {counter.value}", log_file="./logs/anomaly_scoring.log", call_log_main=True, overwrite_last_line=True, stdout=False)

    except Exception as e:
        with lock:
            with open(save_csv, "a") as f:
                f.write(f"None,{os.path.basename(fpath)},None\n")


def get_anomaly_scores_from_folder_parallelized_streamed_mp(folder_path,
                                                             model_path,
                                                             save_csv,
                                                             save_dir=None,
                                                             max_workers=8,
                                                             start_index=None,
                                                             limit=None,
                                                            subset=None):
    from multiprocessing import Manager
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import os

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    if save_csv:
        os.makedirs(os.path.dirname(save_csv), exist_ok=True)

    manager = Manager()
    csv_lock = manager.Lock()
    counter_lock = manager.Lock()
    processed_counter = manager.Value('i', 0)  # shared integer

    if save_csv and not os.path.exists(save_csv):
        with open(save_csv, "w") as f:
            f.write("tic,filename,anomaly_score\n")

    fits_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".fits")])
    if start_index:
        fits_files = fits_files[start_index:]
    if limit:
        fits_files = fits_files[:limit]
    if subset:
        fits_files = [x for x in fits_files if x in subset]
    fits_paths = [os.path.join(folder_path, f) for f in fits_files]

    # Build argument tuples
    args_list = [(f, model_path, save_csv, save_dir, csv_lock, processed_counter, counter_lock) for f in fits_paths]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(score_one, args) for args in args_list]
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Scoring light curves (MP)"):
            pass

    log(f"Multiprocessing scoring complete. Results in: {save_csv}", log_file="./logs/anomaly_scoring.log", call_log_main=True)



# tess_sector = 1

