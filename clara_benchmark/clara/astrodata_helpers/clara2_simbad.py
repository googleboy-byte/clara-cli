import pandas as pd
import time
from multiprocessing import Pool, cpu_count, Manager
from astroquery.mast import Catalogs
from astroquery.simbad import Simbad
from astropy.coordinates import SkyCoord
import astropy.units as u
from tqdm import tqdm
from IPython.display import clear_output

# Configure SIMBAD to fetch object types
Simbad.reset_votable_fields()
Simbad.add_votable_fields("ra", "dec", "otype")

# Shared list to collect labels across workers
manager = Manager()
collected_labels = manager.list()

# === Worker Function ===
def resolve_tic_to_simbad(tic):
    try:
        mast_result = Catalogs.query_object(f"TIC {tic}", catalog="TIC")
        if mast_result is None or len(mast_result) == 0:
            return None
        ra, dec = mast_result[0]["ra"], mast_result[0]["dec"]

        coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
        simbad_result = Simbad.query_region(coord, radius=5 * u.arcsec)

        if simbad_result and len(simbad_result) > 0:
            row = simbad_result[0]
            main_id = row["main_id"].decode() if hasattr(row["main_id"], 'decode') else row["main_id"]
            otype = row["otype"].decode() if hasattr(row["otype"], 'decode') else row["otype"]

            collected_labels.append(otype)
            clear_output(wait=True)
            print(f"Label frequency so far: {dict(pd.Series(list(collected_labels)).value_counts())}")

            return {
                "tic": tic,
                "ra": ra,
                "dec": dec,
                "simbad_main_id": main_id,
                "simbad_label": otype,
                "is_tic_named": "TIC" in str(main_id)
            }
    except Exception:
        return None

# === Sector-Based Multiprocessing SIMBAD Fetch ===
def fetch_sector_tois_with_simbad_labels_mp(sector, max_tois=None, workers=None, pause=0.1, batch_size=10):
    print(f"Fetching TOI list for Sector {sector} from ExoFOP...")
    toi_url = "https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv"
    toi_df = pd.read_csv(toi_url)

    # Filter TOIs to this sector
    toi_df = toi_df[toi_df["Sectors"].astype(str).str.contains(fr"\b{sector}\b", na=False)]
    toi_tics = toi_df["TIC ID"].dropna().astype(int).unique().tolist()

    if max_tois:
        toi_tics = toi_tics[:max_tois]

    total = len(toi_tics)
    print(f"Found {total} TOIs in Sector {sector}. Starting SIMBAD queries...")

    results = []
    pool = Pool(processes=workers or cpu_count())
    pbar = tqdm(total=total)

    for i in range(0, total, batch_size):
        batch = toi_tics[i:i + batch_size]
        batch_results = pool.map(resolve_tic_to_simbad, batch)
        valid = [r for r in batch_results if r]
        results.extend(valid)
        pbar.update(len(batch))
        time.sleep(pause)

    pool.close()
    pool.join()
    pbar.close()

    return pd.DataFrame(results)
