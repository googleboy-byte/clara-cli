from astropy.io import fits
import numpy as np
from scipy.stats import pearsonr

def load_centroid_flux(fname):
    with fits.open(fname) as hdul:
        data = hdul[1].data
        time = data['TIME']
        flux = data['PDCSAP_FLUX']
        
        # Local centroid (good for contamination detection)
        mom_x = data['MOM_CENTR1']
        mom_y = data['MOM_CENTR2']

        # Jitter-corrected (for global flux-position correlation)
        pos_x = data['POS_CORR1']
        pos_y = data['POS_CORR2']

        # Filter valid data
        mask = np.isfinite(time) & np.isfinite(flux) & np.isfinite(mom_x) & np.isfinite(mom_y) & np.isfinite(pos_x) & np.isfinite(pos_y)
        return time[mask], flux[mask], mom_x[mask], mom_y[mask], pos_x[mask], pos_y[mask]


def is_multi_source_v3(phase, flux, mom_x, mom_y, pos_x, pos_y,
                       corr_threshold=0.3,
                       shift_threshold=0.005,
                       transit_center=0.75,
                       transit_width=0.05):
    # Normalize flux and centroids for Pearson correlation
    flux_norm = (flux - np.nanmedian(flux)) / np.nanstd(flux)
    pos_x_norm = (pos_x - np.nanmedian(pos_x)) / np.nanstd(pos_x)
    pos_y_norm = (pos_y - np.nanmedian(pos_y)) / np.nanstd(pos_y)

    try:
        corr_x, _ = pearsonr(flux_norm, pos_x_norm)
        corr_y, _ = pearsonr(flux_norm, pos_y_norm)
    except:
        corr_x, corr_y = np.nan, np.nan

    global_corr_flag = max(abs(corr_x), abs(corr_y)) > corr_threshold

    # Local centroid shift using MOM_CENTR*
    in_transit = (phase > transit_center - transit_width) & (phase < transit_center + transit_width)
    out_transit = (phase < transit_center - 2*transit_width) | (phase > transit_center + 2*transit_width)

    delta_x = np.nanmean(mom_x[in_transit]) - np.nanmean(mom_x[out_transit])
    delta_y = np.nanmean(mom_y[in_transit]) - np.nanmean(mom_y[out_transit])
    local_shift = max(abs(delta_x), abs(delta_y))

    # Estimate transit depth
    # Use percentile-based or localized medians
    baseline_flux = np.nanmedian(flux[out_transit])
    dip_flux = np.nanmedian(flux[in_transit])
    depth = baseline_flux - dip_flux
    
    if depth < 1e-4:
        depth = 1e-4  # avoid division issues

    effective_shift = local_shift / depth
    normalized_shift_flag = effective_shift > (shift_threshold / 0.01)  # adaptive threshold

    # Final flag
    is_multi = normalized_shift_flag

    return is_multi, {
        "corr_x": corr_x,
        "corr_y": corr_y,
        "delta_x": delta_x,
        "delta_y": delta_y,
        "local_shift": local_shift,
        "depth": depth,
        "effective_shift": effective_shift
    }
