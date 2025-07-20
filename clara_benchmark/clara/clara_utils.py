def tic_from_fits_fname(fits_fname):
    return int(fits_fname.split("-")[2].lstrip("0"))

def pick_random_fits_file(base_dir):
    import os
    import random

    """
    Recursively selects a random .fits file from the given directory.

    Args:
        base_dir (str): Path to the base directory to search.

    Returns:
        str: Full path to a randomly chosen .fits file, or None if no files found.
    """
    fits_files = []
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".fits"):
                fits_files.append(os.path.join(root, file))

    if not fits_files:
        print("⚠️ No .fits files found.")
        return None

    return random.choice(fits_files)
    
def log(line, log_file="logs.log"):
    import os
    """
    Prepends a given line to the beginning of the log file.
    
    Args:
        line (str): The line to prepend.
        log_file (str): Path to the log file.
    """
    line = line.rstrip("\n") + "\n"
    
    if os.path.exists(log_file):
        with open(log_file, "r+", encoding="utf-8") as f:
            existing = f.read()
            f.seek(0, 0)
            f.write(line + existing)
    else:
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(line)

            
def extract_tic_ids_from_sh(directory):
    import os
    
    tic_ids = set()
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".sh"):
                path = os.path.join(root, file)
                with open(path, "r") as f:
                    for line in f:
                        # Look for 16-digit number that appears in the filename of the FITS file
                        match = re.search(r'tess[0-9a-z\-]+-(\d{16})-', line)
                        if match:
                            tic_id = match.group(1)
                            tic_ids.add(int(tic_id))
                            # print(f"✅ Found TIC ID: {tic_id}")
    return sorted(tic_ids)

def download_tess_sector_threaded(sector_number=1, catalogues_folder="../catalogues/tess_download_scripts/",
                                  out_dir_tess="../../downloaded_lc/tess_lc/",
                                  start_index=0,
                                  max_downloads=None,
                                  log_file="../logs/download_tess_lc.log",
                                  num_threads=8):
    import requests
    import os
    from tqdm import tqdm
    from concurrent.futures import ThreadPoolExecutor

    sh_filename = f"tesscurl_sector_{sector_number}_lc.sh"
    sh_path = os.path.join(catalogues_folder, sh_filename)
    
    out_dir = os.path.join(out_dir_tess, str(sector_number))
    os.makedirs(out_dir, exist_ok=True)

    # Parse .sh file for URLs
    urls = []
    with open(sh_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("curl") or line.startswith("wget"):
                url = line.split()[-1].strip('"')
                urls.append(url)

    if not urls:
        log("⚠️ No download URLs found in script.", log_file)
        print("⚠️ No download URLs found in script.")
        return

    log(f"🌐 Found {len(urls)} FITS files in sector {sector_number}", log_file)
    print(f"🌐 Found {len(urls)} FITS files in sector {sector_number}")
    log(f"🌐 Downloading {max_downloads if max_downloads is not None else len(urls)} of {len(urls)} FITS files...", log_file)

    urls = urls[start_index:]
    if max_downloads is not None:
        urls = urls[:max_downloads]

    def download_single_file(url):
        filename = os.path.join(out_dir, os.path.basename(url))
        if os.path.exists(filename):
            # log(f"✔️ Already exists: {filename}", log_file)
            # print(f"✔️ Already exists: {filename}")
            return None
        try:
            log(f"⬇️ Downloading {os.path.basename(url)}", log_file)
            r = requests.get(url, stream=True, timeout=60)
            r.raise_for_status()
            with open(filename, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            return os.path.basename(url)
        except Exception as e:
            log(f"❌ Failed to download {url}: {e}", log_file)
            return None

    last_downloaded = None
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = list(tqdm(executor.map(download_single_file, urls),
                            total=len(urls),
                            desc=f"Downloading sector {sector_number}",
                            unit="file"))
        for res in results:
            if res:
                last_downloaded = res

    if last_downloaded:
        log(f"✅ Last successfully downloaded: {last_downloaded}", log_file)
        print(f"✅ Last successfully downloaded: {last_downloaded}")