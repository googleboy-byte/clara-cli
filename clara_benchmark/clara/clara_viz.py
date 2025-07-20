plt_style = "seaborn-v0_8-darkgrid"

def set_clara_dark_theme():
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    
    plt.style.use("default")  # Start from clean baseline
    rcParams.update({
        "axes.facecolor": "#1e1e1e",      # Dark gray background
        "figure.facecolor": "#1e1e1e",
        "axes.edgecolor": "white",
        "axes.labelcolor": "white",
        "xtick.color": "white",
        "ytick.color": "white",
        "text.color": "white",
        "grid.color": "#444444",
        "axes.grid": True,
        "grid.linestyle": "--",
        "lines.color": "#1f77b4",         # Optional: default line color
    })


def viz_LS_feature_vec(flux_vector, power_vector, nfluxpoints=3000, save_path_this=None, show=True):
    import matplotlib.pyplot as plt
    import os
    import numpy as np

    plt_style = "seaborn-v0_8-darkgrid"

    plt.style.use(plt_style)

    # --- Define frequency range: 4 hours to 27 days ---
    num_flux_points = 3000 # MG21 3000 flux light curve points
    num_freq_points = 1000 # MG21 1000 points corresponding to the periodogram
    min_period = 4 / 24  # 4 hours in days
    max_period = 27      # 27 days
    frequencies = np.logspace(np.log10(1 / max_period), np.log10(1 / min_period), num_freq_points)
    
    fig, axs = plt.subplots(2, 1, figsize=(12, 6))
    axs[0].plot(flux_vector, lw=1, color='navy')
    axs[0].set_title(f"Normalized Flux Vector (First {nfluxpoints} Points)")
    axs[0].set_ylabel("Flux (normalized)")
    axs[0].grid(True)
    
    axs[1].semilogx(1 / frequencies, power_vector, lw=1, color='darkred')
    axs[1].set_title("🎚️ Lomb-Scargle Power Spectrum")
    axs[1].set_xlabel("Period (days)")
    axs[1].set_ylabel("Power")
    axs[1].grid(True)
    
    plt.tight_layout()
    if save_path_this:
        os.makedirs(os.path.dirname(save_path_this), exist_ok=True)
        plt.savefig(save_path_this, dpi=200)
    if show==True:
        plt.show()

def viz_lc(lc, title=None, save_path=None, show=True):
    import matplotlib.pyplot as plt
    from lightkurve import LightCurve

    plt.style.use(plt_style)

    plt.figure(figsize=(12, 4))
    plt.plot(lc.time.value, lc.flux.value, color="black", lw=0.5)
    plt.xlabel("Time (JD)")
    plt.ylabel("Flux (e/s)")
    plt.title(title if title else "Raw Light Curve")
    plt.grid(True)

    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
    if show:
        plt.show()
    else:
        plt.close()

        
def viz_pdcsap_flux_tess(fits_path):
    import warnings
    warnings.filterwarnings(
        "ignore",
        message="Numerical value without unit or explicit format passed to TimeDelta, assuming days",
        module="astropy"
    )
    
    from astropy.table import Table
    from lightkurve import LightCurve
    
    table = Table.read(fits_path, hdu=1)
    lc = LightCurve(time=table["TIME"], flux=table["PDCSAP_FLUX"])
    lc.flux = lc.flux.value
    lc.flux_err = lc.flux_err.value
    lc = lc.normalize()
    
    # viz_lc(lc, title="TIC 70442570 Sector 1 RAW")
    viz_lc(lc, title="TIC 70442570 Sector 1 PDC Processed")