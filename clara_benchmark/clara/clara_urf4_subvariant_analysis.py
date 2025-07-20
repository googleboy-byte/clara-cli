import os
import pandas as pd

# urf4 subvariants analysis helper functions

def parse_variant_params(filename):
    """Extracts n, duration, cadence, and noise level from filename."""
    parts = filename.split('_')
    params = {
        "n_transits": int(parts[3][1:]),
        "duration_hr": int(parts[4][1:]),
        "cadence_min": int(parts[5][1:]),
        "noise_ppm": int(parts[6][1:-4])  # strip 'ppm.csv'
    }
    return params

def evaluate_urf4_variants(
    csv_folder,
    toi_filename_list,
    output_csv_path=None,
    threshold_percent=None  # Optional: top n% thresholding
):
    # Load TOI filename list
    
    toi_filenames = set(toi_filename_list)

    results = []

    for fname in os.listdir(csv_folder):
        if not fname.endswith(".csv"):
            continue
        
        # Parse variant parameters
        params = parse_variant_params(fname)
        full_path = os.path.join(csv_folder, fname)
        df = pd.read_csv(full_path)

        # All predicted anomalies (non-zero score entries)
        total_lcs = len(df)
        nonzero_anomalies = df[df["anomaly_score"] > 0]
        num_anomalies = len(nonzero_anomalies)

        # TOIs flagged
        df["is_toi"] = df["filename"].isin(toi_filenames)
        num_tois_total = df["is_toi"].sum()
        num_tois_flagged = df[df["anomaly_score"] > 0]["is_toi"].sum()

        row = {
            "variant": fname,
            **params,
            "n_total": total_lcs,
            "n_anomalies": num_anomalies,
            "n_tois_total": num_tois_total,
            "n_tois_flagged": num_tois_flagged,
            "anomaly_rate": num_anomalies / total_lcs,
            "toi_recall": num_tois_flagged / num_tois_total if num_tois_total > 0 else 0
        }

        # Optional thresholding
        if threshold_percent is not None:
            cutoff = df["anomaly_score"].quantile(1 - threshold_percent / 100)
            thresholded_df = df[df["anomaly_score"] >= cutoff]
            thresholded_tois = thresholded_df["is_toi"].sum()
            row.update({
                "threshold_percent": threshold_percent,
                "n_thresholded": len(thresholded_df),
                "n_tois_thresholded": thresholded_tois,
                "thresholded_toi_recall": thresholded_tois / num_tois_total if num_tois_total > 0 else 0,
                "thresholded_anomaly_rate": len(thresholded_df) / total_lcs
            })

        results.append(row)

    summary_df = pd.DataFrame(results)
    
    if output_csv_path:
        summary_df.to_csv(output_csv_path, index=False)
        print(f"✅ Summary saved to: {output_csv_path}")
    
    return summary_df