import os
import numpy as np
from astropy.table import Table
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle
from lightkurve import LightCurve

def phase_fold(lc_fits_path,
                min_period=0.2, 
                max_period=20, 
                ls_power_thresh=0.05, 
                save_path=None,
                bins=100,
                show=False,
                phase_shift=0.0
):
    tic = os.path.basename(lc_fits_path).split("-")[2].lstrip("0")
    table = Table.read(lc_fits_path, hdu=1)    
    lc = LightCurve(time=table["TIME"], flux=table["PDCSAP_FLUX"])
    
    time=lc.time.value
    flux=lc.flux.value

    # strip units
    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)
    
    # --- Clean NaNs ---
    mask = np.isfinite(time) & np.isfinite(flux)
    time = time[mask]
    flux = flux[mask]
    
    if len(time) < 100:
        print(f"⚠️ Not enough valid points: {fits_file_path}")
        return

    # --- Normalize Flux ---
    flux = (flux - np.nanmean(flux)) / np.nanstd(flux)


    # --- Lomb-Scargle ---
    freq, power = LombScargle(time, flux).autopower(
        minimum_frequency=1 / max_period,
        maximum_frequency=1 / min_period,
        samples_per_peak=10,
    )
    # best_period = (1 / freq[np.argmax(power)]) * 2
    best_period = 1 / freq[np.argmax(power)]
    peak_power = np.max(power)

    if peak_power < ls_power_thresh:
        print(f"⚠️ Skipping {os.path.basename(lc_fits_path)} due to low power: {peak_power:.4f}")
        return

    # --- Phase Fold ---
    phase = (time % best_period) / best_period
    phase = (phase - phase_shift) % 1.0
    sorted_indices = np.argsort(phase)
    phase = phase[sorted_indices]
    folded_flux = flux[sorted_indices]

    # --- Optional: Bin phase curve ---
    from scipy.stats import binned_statistic
    bin_means, bin_edges, _ = binned_statistic(phase, folded_flux, bins=bins, statistic='mean')
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    fig, axs = plt.subplots(2, 1, figsize=(10, 5))
    axs[1].plot(flux[:8000], lw=1, color='navy')
    axs[1].set_title(f"Normalized Flux Vector - First 8000 Data Points")
    axs[1].set_ylabel("Flux (normalized)")
    axs[1].grid(True)
    
    axs[0].plot(phase, folded_flux, '.', alpha=0.2, label="Raw folded flux")
    axs[0].plot(bin_centers, bin_means, 'r-', lw=2, label="Binned")
    axs[0].set_title(f"Phase Folded Light Curve\nPeriod = {best_period:.4f} d | LS Power = {peak_power:.4f} | TIC: {tic}")
    axs[0].set_xlabel("Phase")
    axs[0].set_ylabel("Normalized Flux")
    axs[0].grid(True)
    
    # plt.figure(figsize=(10, 5))
    # plt.plot(phase, folded_flux, '.', alpha=0.2, label="Raw folded flux")
    # plt.plot(bin_centers, bin_means, 'r-', lw=2, label="Binned")
    # plt.title(f"Phase Folded Light Curve\nPeriod = {best_period:.4f} d | LS Power = {peak_power:.4f}")
    # plt.xlabel("Phase")
    # plt.ylabel("Normalized Flux")
    # plt.legend()
    plt.tight_layout()
    
    if save_path != None:
        plt.savefig(save_path + f"_period_{best_period}_days.png")
        print(f"✅ Saved: {save_path}")
    if show:
        plt.show()
    else:
        plt.close()
    return phase, folded_flux, bin_centers, bin_means, flux[:5000]