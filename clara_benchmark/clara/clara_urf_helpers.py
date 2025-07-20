def lc_from_fits_norm(fits_path):
    from astropy.table import Table
    from lightkurve import LightCurve
    
    table = Table.read(fits_path, hdu=1)
    lc = LightCurve(time=table["TIME"], flux=table["PDCSAP_FLUX"])
    lc.flux = lc.flux.value
    lc.flux_err = lc.flux_err.value
    lc.normalize()

    return lc
