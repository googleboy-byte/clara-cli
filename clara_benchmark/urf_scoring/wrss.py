import pandas as pd
import numpy as np
import os
from clara_benchmark.utils.logging import log
from clara_benchmark.utils.logging import log_main
from clara_benchmark.utils.logging import log_system_status

def calc_wrss(interdf, weight_p3=0.8, weight_p9=0.2):
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    root_p3 = np.sqrt(interdf['anomaly_score_p3'])
    root_p9 = np.sqrt(interdf['anomaly_score_p9'])
    interdf['score_weighted_root_sum'] = weight_p3 * root_p3 + weight_p9 * root_p9
    interdf['score_weighted_root_sumnorm'] = scaler.fit_transform(interdf[['score_weighted_root_sum']])
    return interdf

def calculate_wrss_p3p9(p3_csv_fname, p9_csv_fname, save_csv):
    log(f"Calculating WRSS for p3 and p9 models", log_file="./logs/anomaly_scoring.log", call_log_main=True)
    make_dir_if_not_exists(os.path.dirname(save_csv))
    log_system_status(f"Calculating WRSS for p3 and p9 models", stdout=False)
    p3_scores = pd.read_csv(p3_csv_fname)
    p9_scores = pd.read_csv(p9_csv_fname)
    p3p9_inter = pd.merge(p3_scores[p3_scores["anomaly_score"]>0].copy(),
                           p9_scores[p9_scores["anomaly_score"]>0].copy(), on="filename", suffixes=('_p3', '_p9'))
    p3p9_inter = calc_wrss(p3p9_inter)
    p3p9_inter.to_csv(save_csv, index=False)
    return p3p9_inter

def make_dir_if_not_exists(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)