import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
from lightkurve import LightCurve
from scipy.stats import binned_statistic

def plot_bls_and_folded_from_list(
    fits_files,
    output_dir="./bls_folded_output",
    bins=150,
    first_n_points=8000,
    min_points=100,
    flatten_window=401,
    bls_min_period=0.5,
    bls_max_period=20.0,
    bls_period_samples=5000,
    save_plots=True,
    show=False
):
    os.makedirs(output_dir, exist_ok=True)

    for file_path in fits_files:
        file_name = os.path.basename(file_path)

        try:
            with fits.open(file_path) as hdul:
                data = Table(hdul[1].data)
                time = np.asarray(data['TIME'])
                flux = np.asarray(data['PDCSAP_FLUX'])

            # Clean
            mask = np.isfinite(time) & np.isfinite(flux)
            time = time[mask]
            flux = flux[mask]

            if len(time) < min_points:
                print(f"⚠️ Skipping {file_name}: not enough points")
                continue

            # Normalize and flatten
            flux = (flux - np.nanmean(flux)) / np.nanstd(flux)
            lc = LightCurve(time=time, flux=flux).remove_nans().flatten(window_length=flatten_window)

            # BLS periodogram
            bls = lc.to_periodogram(method="bls", period=np.linspace(bls_min_period, bls_max_period, bls_period_samples))
            period = bls.period_at_max_power.value
            t0 = bls.transit_time_at_max_power.value
            duration = bls.duration_at_max_power.value

            # Phase fold
            folded = lc.fold(period=period, epoch_time=t0)
            phase = folded.phase.value
            folded_flux = folded.flux.value

            # Sort for plotting
            idx = np.argsort(phase)
            phase = phase[idx]
            folded_flux = folded_flux[idx]
            flux_clip = np.clip(folded_flux, -6, 3)

            # Bin for smoothing
            bin_means, bin_edges, _ = binned_statistic(phase, flux_clip, bins=bins, statistic='mean')
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

            # PLOT
            fig, axs = plt.subplots(3, 1, figsize=(10, 8))

            # 1. BLS Periodogram
            axs[0].plot(bls.period.value, bls.power.value, 'k')
            axs[0].axvline(period, color='red', linestyle='--', label=f"Best P = {period:.4f} d")
            axs[0].set_xlabel("Period (days)")
            axs[0].set_ylabel("BLS Power")
            axs[0].set_title(f"BLS Periodogram: {file_name}")
            axs[0].legend()
            axs[0].grid(True)

            # 2. Phase Folded LC
            axs[1].scatter(phase, flux_clip, s=4, alpha=0.3, color="steelblue", label="Raw")
            axs[1].plot(bin_centers, bin_means, color="red", lw=2, label="Binned")
            axs[1].set_xlabel("Phase")
            axs[1].set_ylabel("Normalized Flux")
            axs[1].set_title("Phase Folded Light Curve")
            axs[1].grid(True)
            axs[1].legend()

            # 3. Raw flux vector (first N points)
            axs[2].plot(flux[:first_n_points], lw=1, color='navy')
            axs[2].set_ylabel("Flux")
            axs[2].set_title(f"First {first_n_points} Flux Points")
            axs[2].grid(True)

            plt.tight_layout()

            if save_plots:
                out_path = os.path.join(output_dir, file_name.replace(".fits", "_summary.png"))
                plt.savefig(out_path)
            if show:
                plt.show()
            else:
                plt.close()

        except Exception as e:
            print(f"❌ Error processing {file_name}: {e}")
